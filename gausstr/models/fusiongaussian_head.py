import torch
import torch.nn.functional as F
from mmdet.models import MLP
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

from .gsplat_rasterization import rasterize_gaussians
from .utils import (cam2world, flatten_bsn_forward, get_covariance,
                    inverse_sigmoid, rotmat_to_quat)


def prompt_denoising(logits, logit_scale=100, pd_threshold=0.1):
    probs = logits.softmax(-1)
    probs_ = F.softmax(logits * logit_scale, -1)
    max_cls_conf = probs_.flatten(1, 3).max(1).values
    selected_cls = (max_cls_conf < pd_threshold)[:, None, None,
                                                 None].expand(*probs.shape)
    probs[selected_cls] = 0
    return probs


@MODELS.register_module()
class FusionGaussianHead_VoxelSup(BaseModule):
    """Predicts 3D Gaussians from the decoder queries.

    Training losses:
        * 2D: rendered depth vs. Metric3D/LiDAR depth, and rendered
          (PCA-reduced) features vs. DINOv2 image features.
        * 3D: Gaussians are voxelized by the CUDA voxelizer; voxel density is
          supervised by the binary pseudo occupancy (BCE) and voxel features
          by the DINOv2 pseudo labels (cosine loss).
    Inference: voxel features are matched with the text embeddings to
    predict semantic occupancy (zero-shot).
    """

    def __init__(self,
                 opacity_head,
                 feature_head,
                 scale_head,
                 regress_head,
                 reduce_dims,
                 image_shape,
                 patch_size,
                 voxelizer_train,
                 text_protos=None,
                 depth_limit=54.0,
                 depth_limit_min=3.0,
                 prompt_denoising=True,
                 pred_logits_thres=0.5,
                 ego_point_range=None,
                 ego_grid_shape=None,
                 ego_grid_size=None,
                 use_ce_loss=True,
                 use_feat_loss=True,
                 loss_bce_weight=1.0,
                 loss_feat_weight=1.0):
        super().__init__()
        self.opacity_head = MODELS.build(opacity_head)
        self.feature_head = MODELS.build(feature_head)
        self.scale_head = MODELS.build(scale_head)
        self.regress_head = MODELS.build(regress_head)

        self.reduce_dims = reduce_dims
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.depth_limit = depth_limit
        self.depth_limit_min = depth_limit_min
        self.prompt_denoising = prompt_denoising
        self.use_ce_loss = use_ce_loss
        self.use_feat_loss = use_feat_loss
        self.loss_bce_weight = loss_bce_weight
        self.loss_feat_weight = loss_feat_weight
        # sigmoid threshold on the predicted occupancy that selects the voxels
        # supervised by the feature loss
        self.pred_logits_thres = pred_logits_thres

        self.g2v_aggregator = MODELS.build(voxelizer_train)
        self.ego_point_range = ego_point_range
        self.ego_grid_shape = ego_grid_shape
        self.ego_grid_size = ego_grid_size

        if text_protos is not None:
            self.register_buffer('text_proto_embeds',
                                 torch.load(text_protos, map_location='cpu'))

        self.silog_loss = MODELS.build(dict(type='SiLogLoss', _scope_='mmseg'))
        self.density_to_logits = MLP(1, 16, 1, 2)

    def forward(self,
                x,
                ref_pts,
                depth,
                cam2img,
                cam2ego,
                img_shape,
                mode='tensor',
                feats=None,
                img_aug_mat=None,
                occ_gt_mask_camera=None,
                pseudo_binary_gt=None,
                pseudo_dino_gt=None,
                pseudo_dino_gt_v=None,
                **kwargs):
        bs, n = cam2img.shape[:2]
        x = x.reshape(bs, n, *x.shape[1:])  # (B, N, num_queries, C)

        # Gaussian centers: refine the reference points in the image frame
        # (u, v, normalized depth) and lift them to the ego frame.
        deltas = self.regress_head(x)  # (B, N, num_queries, 3)
        ref_pts = (deltas + inverse_sigmoid(ref_pts)).sigmoid()  # [0, 1]
        learned_depth = self.depth_limit_min + (
            self.depth_limit - self.depth_limit_min) * ref_pts[..., 2]
        learned_depth = learned_depth.unsqueeze(-1)
        points = torch.cat([
            ref_pts[..., :2] * torch.tensor(self.image_shape[::-1]).to(x),
            learned_depth
        ], -1)
        means3d = cam2world(points, cam2img, cam2ego,
                            img_aug_mat)  # (B, N, num_queries, 3)

        opacities = self.opacity_head(x).float()  # (B, N, num_queries, 1)
        features = self.feature_head(x).float()  # (B, N, num_queries, 768)
        scales = self.scale_head(x) * self.scale_transform(
            learned_depth, cam2img[..., 0, 0]).clamp(1e-6)  # camera frame

        covariances = flatten_bsn_forward(
            get_covariance, scales,
            cam2ego[..., None, :3, :3])  # (B, N, num_queries, 3, 3), ego
        rotations = flatten_bsn_forward(rotmat_to_quat, cam2ego[..., :3, :3])
        rotations = rotations.unsqueeze(2).expand(-1, -1, x.size(2), -1)

        if mode == 'predict':
            # (768, num_classes) text embeddings -> per-Gaussian logits
            features = features @ self.text_proto_embeds
            density, grid_feats = self.g2v_aggregator(
                means3d=means3d.flatten(0, 2).float(),
                opacities=opacities.flatten(0, 3).float(),
                covariances=covariances.flatten(0, 2).flatten(1, 2).float(),
                features=features.flatten(0, 2).float().softmax(-1))
            density = density.unsqueeze(0).unsqueeze(-1)
            grid_feats = grid_feats.unsqueeze(0)

            if self.prompt_denoising:
                probs = prompt_denoising(grid_feats)
            else:
                probs = grid_feats.softmax(-1)
            preds = probs.argmax(-1)
            # low-density voxels are free (class 17)
            preds = torch.where(density.squeeze(-1) > 4e-2, preds, 17)
            return preds

        losses = {}
        # 3D voxel supervision
        assert means3d.shape[0] == 1, 'only batch size 1 per GPU is supported'
        grid_density, grid_feats = self.g2v_aggregator(
            means3d=means3d.flatten(0, 2).float(),
            opacities=opacities.flatten(0, 3).float(),
            covariances=covariances.flatten(0, 2).flatten(1, 2).float(),
            # project features onto the PCA basis of the pseudo labels
            features=(features.flatten(0, 2) @ pseudo_dino_gt_v.squeeze(0)
                      ).float())  # (200, 200, 16), (200, 200, 16, C)
        voxels_pred = grid_density.flatten(0, 2).unsqueeze(-1)
        voxels_pred_logits = self.density_to_logits(voxels_pred).squeeze(-1)

        voxels_label_full = pseudo_binary_gt.to(dtype=torch.bool).squeeze(0)
        if self.use_ce_loss:
            voxels_label = voxels_label_full.view(-1)
            voxels_pred = voxels_pred_logits.reshape(
                1, *self.ego_grid_shape)[occ_gt_mask_camera].flatten()
            voxels_label = voxels_label.reshape(
                1, *self.ego_grid_shape)[occ_gt_mask_camera].flatten()
            pos_weight = (voxels_label == 0).sum() / (voxels_label == 1).sum()
            losses['loss_bce_logits'] = self.bce_loss_with_logits(
                voxels_pred, voxels_label,
                pos_weight=pos_weight) * self.loss_bce_weight

        if self.use_feat_loss:
            occ_mask = voxels_label_full * (
                voxels_pred_logits.reshape(*self.ego_grid_shape).sigmoid() >
                self.pred_logits_thres)
            losses['loss_feat'] = self.g2v_aggregator.get_feat_loss(
                occ_mask.float(),
                occ_gt_mask_camera.squeeze(0).bool(),
                grid_feats,
                pseudo_dino_gt.squeeze(0)) * self.loss_feat_weight

        # 2D rendering supervision
        tgt_feats = feats.flatten(-2).mT  # (bsn, h * w, C)
        u, s, v = torch.pca_lowrank(
            tgt_feats.flatten(0, 2).double(), q=self.reduce_dims, niter=4)
        tgt_feats = tgt_feats @ v.to(tgt_feats)
        features = features @ v.to(features)
        features = features.float()

        rendered = rasterize_gaussians(
            means3d.flatten(1, 2),
            features.flatten(1, 2),
            opacities.squeeze(-1).flatten(1, 2),
            scales.flatten(1, 2),
            rotations.flatten(1, 2),
            cam2img,
            cam2ego,
            covariances=covariances.flatten(1, 2),
            img_aug_mats=img_aug_mat,
            image_size=(900, 1600),
            near_plane=0.1,
            far_plane=100,
            render_mode='RGB+D',
            channel_chunk=32).flatten(0, 1)
        rendered_depth = rendered[:, -1]
        rendered = rendered[:, :-1]

        depth = torch.where(depth < self.depth_limit, depth,
                            1e-3).flatten(0, 1)
        losses['loss_depth'] = self.depth_loss(rendered_depth, depth)
        losses['mae_depth'] = self.depth_loss(
            rendered_depth, depth, criterion='l1')

        bsn, c, h, w = rendered.shape
        tgt_feats = tgt_feats.mT.reshape(bsn, c, h // self.patch_size,
                                         w // self.patch_size)
        tgt_feats = F.interpolate(
            tgt_feats, scale_factor=self.patch_size, mode='bilinear')
        rendered = rendered.flatten(2).mT
        tgt_feats = tgt_feats.flatten(2).mT.flatten(0, 1)
        losses['loss_cosine'] = F.cosine_embedding_loss(
            rendered.flatten(0, 1), tgt_feats,
            torch.ones_like(tgt_feats[:, 0])) * 5
        return losses

    def depth_loss(self, pred, target, criterion='silog_l1'):
        loss = 0
        if 'silog' in criterion:
            loss += self.silog_loss(pred, target)
        if 'l1' in criterion:
            target = target.flatten()
            pred = pred.flatten()[target != 0]
            l1_loss = F.l1_loss(pred, target[target != 0])
            if loss != 0:
                l1_loss *= 0.2
            loss += l1_loss
        return loss

    def bce_loss_with_logits(self,
                             pred,
                             target,
                             pos_weight=None,
                             ignore_index=255):
        target = target.view(-1)
        pred = pred.view(-1)
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight],
                                      device=pred.device,
                                      dtype=pred.dtype)
        return F.binary_cross_entropy_with_logits(
            pred, target.float(), pos_weight=pos_weight)

    def scale_transform(self, depth, focal, multiplier=7.5):
        return depth * multiplier / focal.reshape(*depth.shape[:2], 1, 1)
