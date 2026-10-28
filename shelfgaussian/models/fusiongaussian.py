from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from mmcv.ops import Voxelization
from mmdet3d.registry import MODELS
from mmengine.model import BaseModel, ModuleList
from mmengine.runner.checkpoint import (_load_checkpoint_with_prefix,
                                        load_state_dict)
from torch.nn import functional as F
from torch.nn.init import normal_

from .utils import flatten_bsn_forward


@MODELS.register_module()
class FusionGaussian(BaseModel):
    """ShelfGaussian: a multi-modal Gaussian transformer.

    Each camera holds `num_queries` Gaussian queries with 3D reference points
    in its image frame. A deformable transformer decoder refines them by
    attending to DINOv2 image features and, optionally, LiDAR / radar BEV
    features; a Gaussian head after every decoder layer predicts the
    Gaussians and their losses.

    Args:
        use_lidar (bool): Encode LiDAR points (SECOND backbone).
        use_radar (bool): Encode radar points (PointPillars-style encoder).
        img_backbone (dict): `dict(type='TorchHubModel', repo_or_dir,
            model_name)`; the backbone is frozen.
        num_queries (int): Number of Gaussian queries per camera.
        depth_limit_min, depth_limit (float): Depth range of the queries.
    """

    def __init__(self,
                 use_lidar=False,
                 use_radar=False,
                 # lidar
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_backbone=None,
                 pts_neck=None,
                 load_lidar_path=None,
                 freeze_lidar_backbone=False,
                 # camera
                 img_backbone=None,
                 img_neck=None,
                 # radar
                 radar_voxel_layer=None,
                 radar_voxel_encoder=None,
                 radar_middle_encoder=None,
                 load_radar_path=None,
                 freeze_radar_backbone=False,
                 # decoder
                 decoder=None,
                 num_queries=300,
                 embed_dims=256,
                 depth_limit=54.0,
                 depth_limit_min=3.0,
                 # head
                 gauss_head=None,
                 num_feature_levels=4,
                 **kwargs):
        super().__init__(**kwargs)
        # camera (frozen DINOv2 from torch.hub)
        assert img_backbone.type == 'TorchHubModel'
        self.img_backbone = torch.hub.load(img_backbone.repo_or_dir,
                                           img_backbone.model_name)
        self.patch_size = self.img_backbone.patch_size
        self.img_backbone.requires_grad_(False)
        self.img_backbone.eval()
        self.img_backbone.is_init = True  # skip re-initialization by mmengine
        for param in self.img_backbone.parameters():
            param.requires_grad = False
        self.img_neck = MODELS.build(img_neck)

        # lidar
        self.use_lidar = use_lidar
        if use_lidar:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
            self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
            self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
            self.pts_backbone = MODELS.build(pts_backbone)
            self.pts_neck = MODELS.build(pts_neck)
        self.load_lidar_path = load_lidar_path
        self.freeze_lidar_backbone = freeze_lidar_backbone
        if load_lidar_path is not None:
            self.load_lidar_pretrained()

        # radar
        self.use_radar = use_radar
        if use_radar:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
            self.radar_voxel_encoder = MODELS.build(radar_voxel_encoder)
            self.radar_middle_encoder = MODELS.build(radar_middle_encoder)
        self.load_radar_path = load_radar_path
        self.freeze_radar_backbone = freeze_radar_backbone
        if load_radar_path is not None:
            self.load_radar_pretrained()

        # decoder
        self.decoder = MODELS.build(decoder)
        self.num_queries = num_queries
        self.embed_dims = embed_dims
        self.depth_limit = depth_limit
        self.depth_limit_min = depth_limit_min
        self.query_embeds = nn.Embedding(num_queries, self.embed_dims)

        # one Gaussian head per decoder layer
        self.gauss_heads = ModuleList(
            [MODELS.build(gauss_head) for _ in range(decoder.num_layers)])

        # Level embeddings of the BEV features. They are not used in the
        # forward pass but kept for compatibility with the checkpoints.
        self.num_feature_levels = num_feature_levels
        if self.use_lidar or self.use_radar:
            self.level_embeds = nn.Parameter(
                torch.Tensor(self.num_feature_levels, self.embed_dims))

        self.init_weights()

    def init_weights(self):
        if self.use_lidar or self.use_radar:
            normal_(self.level_embeds)

    def load_lidar_pretrained(self):
        """Load the pretrained FUTR3D LiDAR backbone."""
        checkpoint_path = self.load_lidar_path
        pts_middle_encoder_ckpt = _load_checkpoint_with_prefix(
            prefix='pts_middle_encoder',
            filename=checkpoint_path,
            map_location='cpu')
        load_state_dict(self.pts_middle_encoder, pts_middle_encoder_ckpt)

        pts_backbone_ckpt = _load_checkpoint_with_prefix(
            prefix='pts_backbone', filename=checkpoint_path,
            map_location='cpu')
        load_state_dict(self.pts_backbone, pts_backbone_ckpt)

        pts_neck_ckpt = _load_checkpoint_with_prefix(
            prefix='pts_neck', filename=checkpoint_path, map_location='cpu')
        load_state_dict(self.pts_neck, pts_neck_ckpt)

        if self.freeze_lidar_backbone:
            for module in (self.pts_middle_encoder, self.pts_backbone,
                           self.pts_neck):
                module.requires_grad_(False)

    def load_radar_pretrained(self):
        """Load the pretrained FUTR3D radar encoder."""
        checkpoint_path = self.load_radar_path
        prefixes = [
            'encoders_student.radar.backbone.pts_voxel_encoder',
            'encoders.radar.backbone.pts_voxel_encoder', 'radar_voxel_encoder'
        ]

        radar_voxel_encoder_ckpt = None
        for prefix in prefixes:
            try:
                radar_voxel_encoder_ckpt = _load_checkpoint_with_prefix(
                    prefix=prefix,
                    filename=checkpoint_path,
                    map_location='cpu')
                break
            except AssertionError:
                continue

        if radar_voxel_encoder_ckpt is None:
            raise ValueError(
                f'None of the expected prefixes {prefixes} were found in '
                f'checkpoint {checkpoint_path}')

        load_state_dict(self.radar_voxel_encoder, radar_voxel_encoder_ckpt)

        if self.freeze_radar_backbone:
            self.radar_voxel_encoder.requires_grad_(False)

    def prepare_inputs(self, inputs_dict, data_samples):
        """Collect and batch the meta information and labels."""
        num_views = data_samples[0].num_views
        self.num_views = num_views
        img_shape = data_samples[0].resize_img_shape
        device = inputs_dict['imgs'].device

        lidar2cam = []
        cam2img = []
        lidar2img = []
        lidar2ego = []
        cam2ego = []
        ego2global = []
        img_aug_mat = []
        depth = []
        occ_gt_mask_camera = []
        pseudo_binary_gt = []
        pseudo_dino_gt = []
        pseudo_dino_gt_v = []

        for i in range(len(data_samples)):
            data_samples[i].set_metainfo(
                {'cam2img': data_samples[i].cam2img[:num_views]})
            cam2img.append(data_samples[i].cam2img)
            data_samples[i].set_metainfo(
                {'cam2ego': data_samples[i].cam2ego[:num_views]})
            cam2ego.append(data_samples[i].cam2ego)
            data_samples[i].set_metainfo(
                {'lidar2cam': data_samples[i].lidar2cam[:num_views]})
            lidar2cam.append(data_samples[i].lidar2cam)
            data_samples[i].set_metainfo(
                {'lidar2img': data_samples[i].lidar2img[:num_views]})
            lidar2img.append(data_samples[i].lidar2img)
            if self.use_lidar:
                lidar2ego.append(data_samples[i].lidar2ego)
            ego2global.append(data_samples[i].ego2global)
            if hasattr(data_samples[i], 'img_aug_mat'):
                data_samples[i].set_metainfo(
                    {'img_aug_mat': data_samples[i].img_aug_mat[:num_views]})
                img_aug_mat.append(data_samples[i].img_aug_mat)
            depth.append(data_samples[i].depth)
            if hasattr(data_samples[i], 'mask_camera'):
                occ_gt_mask_camera.append(data_samples[i].mask_camera)
            if hasattr(data_samples[i], 'pseudo_binary_gt'):
                pseudo_binary_gt.append(data_samples[i].pseudo_binary_gt)
            if hasattr(data_samples[i], 'pseudo_dino_gt'):
                pseudo_dino_gt.append(data_samples[i].pseudo_dino_gt)
            if hasattr(data_samples[i], 'pseudo_dino_gt_v'):
                pseudo_dino_gt_v.append(data_samples[i].pseudo_dino_gt_v)

        data_samples = dict(
            depth=depth,
            lidar2cam=lidar2cam,
            cam2img=cam2img,
            lidar2img=lidar2img,
            cam2ego=cam2ego,
            lidar2ego=lidar2ego if self.use_lidar else None,
            num_views=num_views,
            ego2global=ego2global,
            img_shape=img_shape,
            img_aug_mat=img_aug_mat if img_aug_mat else None)
        if occ_gt_mask_camera:
            data_samples['occ_gt_mask_camera'] = occ_gt_mask_camera
        if pseudo_binary_gt:
            data_samples['pseudo_binary_gt'] = pseudo_binary_gt
        if pseudo_dino_gt:
            data_samples['pseudo_dino_gt'] = pseudo_dino_gt
        if pseudo_dino_gt_v:
            data_samples['pseudo_dino_gt_v'] = pseudo_dino_gt_v

        for k, v in data_samples.items():
            if isinstance(v, torch.Tensor) or not isinstance(v, Iterable):
                continue
            if isinstance(v[0], torch.Tensor):
                data_samples[k] = torch.stack(v).to(device)
            else:
                data_samples[k] = torch.from_numpy(np.stack(v)).to(device)
        return inputs_dict, data_samples

    def extract_img_feat(self, inputs, data_samples):
        """Extract DINOv2 patch features and the image FPN features."""
        bs, n = inputs.shape[:2]  # (B, N, C, H, W)
        inputs = inputs.flatten(0, 1)
        if self.img_backbone.training:
            self.img_backbone.eval()
        with torch.no_grad():
            x = self.img_backbone.forward_features(
                inputs)['x_norm_patchtokens']
            x = x.mT.reshape(bs * n, -1, inputs.shape[-2] // self.patch_size,
                             inputs.shape[-1] // self.patch_size)
        # DINOv2 features, also the targets of the 2D rendering loss
        data_samples['feats'] = x.reshape(bs, n, *x.shape[1:])

        img_feats = self.img_neck(x)  # list of (B*N, C, H_l, W_l)
        return [
            feat.reshape(bs, n, *feat.shape[1:]).flatten(0, 1)
            for feat in img_feats
        ]

    @torch.no_grad()
    def voxelize(self, points, voxel_layer):
        """Hard voxelization of a batch of point clouds."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_pts_feat(self, pts):
        """Extract multi-level LiDAR BEV features."""
        voxels, num_points, coors = self.voxelize(pts, self.pts_voxel_layer)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        return x

    def extract_radar_feat(self, radar):
        """Extract the radar BEV feature."""
        voxels, num_points, coors = self.voxelize(radar,
                                                  self.radar_voxel_layer)
        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.radar_middle_encoder(voxel_features, coors, batch_size)
        return [x]

    def forward(self, inputs, data_samples, mode='loss'):
        inputs_dict, data_samples = self.prepare_inputs(inputs, data_samples)
        img_feats = self.extract_img_feat(inputs_dict['imgs'], data_samples)
        pts_feats = self.extract_pts_feat(
            inputs_dict['points']) if self.use_lidar else None
        radar_feats = self.extract_radar_feat(
            inputs_dict['radar']) if self.use_radar else None

        decoder_inputs = self.pre_transformer(img_feats, pts_feats,
                                              radar_feats)
        decoder_inputs.update(self.pre_decoder(decoder_inputs, **data_samples))
        decoder_outputs = self.forward_decoder(
            reg_branches=[h.regress_head for h in self.gauss_heads],
            **decoder_inputs,
            **data_samples)

        query = decoder_outputs['hidden_states']
        reference_points = decoder_outputs['references']

        if mode == 'predict':
            return self.gauss_heads[-1](
                query[-1], reference_points[-1], mode=mode, **data_samples)

        losses = {}
        for i, gauss_head in enumerate(self.gauss_heads):
            loss = gauss_head(
                query[i], reference_points[i], mode=mode, **data_samples)
            for k, v in loss.items():
                losses[f'{k}/{i}'] = v
        return losses

    def get_masks(self, mlvl_feats):
        """All-valid padding masks of the multi-level BEV features."""
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = mlvl_feats[0].shape[-2:]
        img_masks = mlvl_feats[0].new_zeros(
            (batch_size, input_img_h, input_img_w))
        return [
            F.interpolate(img_masks[None],
                          size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            for feat in mlvl_feats
        ]

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def feats_prepare_point(self, mlvl_feats):
        """Flatten multi-level BEV features (LiDAR or radar)."""
        mlvl_masks = self.get_masks(mlvl_feats)
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask) in enumerate(zip(mlvl_feats, mlvl_masks)):
            bs, c, h, w = feat.shape
            spatial_shapes.append((h, w))
            feat_flatten.append(feat.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
        feat_flatten = torch.cat(feat_flatten, 1)  # (bs, sum(h*w), C)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        return (feat_flatten, mask_flatten, spatial_shapes, level_start_index,
                valid_ratios)

    def feats_prepare_cam(self, mlvl_feats):
        """Flatten multi-level image features, list of (B*N, C, H_l, W_l)."""
        batch_size = mlvl_feats[0].size(0)
        feat_flatten = []
        spatial_shapes = []
        for feat in mlvl_feats:
            batch_size, c, h, w = feat.shape
            spatial_shapes.append(torch._shape_as_tensor(feat)[2:].to(
                feat.device))
            feat_flatten.append(feat.view(batch_size, c, -1).permute(0, 2, 1))
        feat_flatten = torch.cat(feat_flatten, 1)  # (B*N, sum(h*w), C)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = mlvl_feats[0].new_ones(batch_size, len(mlvl_feats), 3)
        return feat_flatten, None, spatial_shapes, level_start_index, \
            valid_ratios

    def pre_transformer(self, mlvl_feats, mlvl_feats_lidar, mlvl_feats_radar):
        (feat_flatten, mask_flatten, spatial_shapes, level_start_index,
         valid_ratios) = self.feats_prepare_cam(mlvl_feats)

        lidar = [None] * 5
        if self.use_lidar:
            lidar = self.feats_prepare_point(mlvl_feats_lidar)
        radar = [None] * 5
        if self.use_radar:
            radar = self.feats_prepare_point(mlvl_feats_radar)

        return dict(
            # camera
            memory=feat_flatten,
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # lidar
            memory_lidar=lidar[0],
            memory_mask_lidar=lidar[1],
            spatial_shapes_lidar=lidar[2],
            level_start_index_lidar=lidar[3],
            valid_ratios_lidar=lidar[4],
            # radar
            memory_radar=radar[0],
            memory_mask_radar=radar[1],
            spatial_shapes_radar=radar[2],
            level_start_index_radar=radar[3],
            valid_ratios_radar=radar[4])

    def pre_decoder(self, decoder_inputs_dict, depth=None, **kwargs):
        """Initialize the queries and their reference points.

        Reference points are random image positions (u, v); their normalized
        depth is sampled from the input depth map (Metric3D depth fused with
        projected LiDAR depth).
        """
        bsn = decoder_inputs_dict['memory'].shape[0]  # B*N
        bs = bsn // self.num_views

        query = self.query_embeds.weight.unsqueeze(0).expand(bsn, -1, -1)
        reference_points = torch.rand((bsn, query.size(1), 3)).to(query)
        reference_points_sample = reference_points.view(
            bs, self.num_views, query.shape[1], -1)[..., :2]
        depth = depth.clamp(max=self.depth_limit, min=self.depth_limit_min)
        depth_normalized = (depth - self.depth_limit_min) / (
            self.depth_limit - self.depth_limit_min)
        sample_depth = flatten_bsn_forward(
            F.grid_sample, depth_normalized[:, :self.num_views, None],
            reference_points_sample.unsqueeze(2) * 2 - 1)
        sample_depth = sample_depth[:, :, 0, 0, :, None]
        reference_points[..., 2] = sample_depth.view(bsn, -1)
        return dict(
            query=query, query_pos=None, reference_points=reference_points)

    def forward_decoder(self,
                        query,
                        query_pos,
                        reference_points,
                        memory=None,
                        memory_mask=None,
                        spatial_shapes=None,
                        level_start_index=None,
                        valid_ratios=None,
                        memory_lidar=None,
                        memory_mask_lidar=None,
                        spatial_shapes_lidar=None,
                        level_start_index_lidar=None,
                        valid_ratios_lidar=None,
                        memory_radar=None,
                        memory_mask_radar=None,
                        spatial_shapes_radar=None,
                        level_start_index_radar=None,
                        valid_ratios_radar=None,
                        **kwargs):
        inter_states, references = self.decoder(
            query=query,
            query_pos=query_pos,
            value=memory,
            value_lidar=memory_lidar,
            value_radar=memory_radar,
            key_padding_mask=memory_mask,
            key_padding_mask_lidar=memory_mask_lidar,
            key_padding_mask_radar=memory_mask_radar,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_lidar=spatial_shapes_lidar,
            spatial_shapes_radar=spatial_shapes_radar,
            level_start_index=level_start_index,
            level_start_index_lidar=level_start_index_lidar,
            level_start_index_radar=level_start_index_radar,
            valid_ratios=valid_ratios,
            valid_ratios_lidar=valid_ratios_lidar,
            valid_ratios_radar=valid_ratios_radar,
            **kwargs)
        return dict(hidden_states=inter_states, references=list(references))
