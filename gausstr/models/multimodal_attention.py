"""Multi-modal deformable cross-attention.

Modified from FUTR3D (https://github.com/Tsinghua-MARS-Lab/futr3d).
"""
import math
import warnings

import torch
import torch.nn as nn
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule, constant_init, xavier_init


def deformable_attn(value, spatial_shapes, level_start_index,
                    sampling_locations, attention_weights, im2col_step):
    if ((IS_CUDA_AVAILABLE and value.is_cuda)
            or (IS_MLU_AVAILABLE and value.is_mlu)):
        return MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, im2col_step)
    return multi_scale_deformable_attn_pytorch(value, spatial_shapes,
                                               sampling_locations,
                                               attention_weights)


@MODELS.register_module()
class MultiModalAttention(BaseModule):
    """Deformable cross-attention from per-camera Gaussian queries to camera,
    LiDAR BEV and radar BEV features. The outputs of the used modalities are
    concatenated and fused by an MLP.

    The queries carry 3D reference points in the image frame of their camera
    (normalized u, v and depth). They are sampled directly on the image
    features, and lifted to the LiDAR frame to sample the BEV features.

    Args:
        use_lidar, use_camera, use_radar (bool): Modalities to attend to.
        embed_dims (int): Embedding dimension. Defaults to 256.
        radar_dims (int): Channels of the radar BEV features. Defaults to 64.
        num_cams (int): Number of cameras. Defaults to 6.
        num_heads (int): Number of attention heads. Defaults to 8.
        num_levels (int): Number of feature levels. Defaults to 4.
        num_points (int): Sampling points per head and level. Defaults to 4.
        pc_range (list[float]): LiDAR BEV range.
        radar_pc_range (list[float]): Radar BEV range.
        depth_limit_min, depth_limit (float): Depth range of the normalized
            reference point depth.
    """

    def __init__(self,
                 use_lidar=True,
                 use_camera=False,
                 use_radar=False,
                 embed_dims=256,
                 radar_dims=64,
                 num_cams=6,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 value_proj_ratio=1.0,
                 depth_limit=54.0,
                 depth_limit_min=3.0,
                 pc_range=None,
                 radar_pc_range=None,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        assert use_lidar or use_camera, 'radar-only is not supported'
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.use_radar = use_radar
        self.pc_range = pc_range
        self.radar_pc_range = radar_pc_range
        self.num_cams = num_cams
        self.depth_limit = depth_limit
        self.depth_limit_min = depth_limit_min

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.fused_embed = 0
        if self.use_lidar:
            self.sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_levels * num_points * 2)
            self.attention_weights = nn.Linear(
                embed_dims, num_heads * num_levels * num_points)
            self.value_proj = nn.Linear(embed_dims, embed_dims)
            self.output_proj = nn.Linear(embed_dims, embed_dims)
            self.fused_embed += embed_dims

        if self.use_camera:
            self.img_sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_levels * num_points * 2)
            self.img_attention_weights = nn.Linear(
                embed_dims, num_heads * num_levels * num_points)
            value_proj_size = int(embed_dims * value_proj_ratio)
            self.img_value_proj = nn.Linear(embed_dims, value_proj_size)
            self.img_output_proj = nn.Linear(value_proj_size, embed_dims)
            self.fused_embed += embed_dims

        if self.use_radar:
            self.rad_sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_points * 2)
            self.rad_attention_weights = nn.Linear(embed_dims,
                                                   num_heads * num_points)
            self.rad_value_proj = nn.Linear(radar_dims, radar_dims)
            self.rad_output_proj = nn.Linear(radar_dims, radar_dims)
            self.fused_embed += radar_dims

        if self.fused_embed > embed_dims:
            self.modality_fusion_layer = nn.Sequential(
                nn.Linear(self.fused_embed, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=False),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
            )

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        if self.use_lidar:
            constant_init(self.sampling_offsets, 0.)
            self.sampling_offsets.bias.data = grid_init.view(-1)
            constant_init(self.attention_weights, val=0., bias=0.)
            xavier_init(self.value_proj, distribution='uniform', bias=0.)
            xavier_init(self.output_proj, distribution='uniform', bias=0.)
        if self.use_camera:
            constant_init(self.img_sampling_offsets, 0.)
            self.img_sampling_offsets.bias.data = grid_init.view(-1)
            constant_init(self.img_attention_weights, val=0., bias=0.)
            xavier_init(self.img_value_proj, distribution='uniform', bias=0.)
            xavier_init(self.img_output_proj, distribution='uniform', bias=0.)
        if self.use_radar:
            constant_init(self.rad_sampling_offsets, 0.)
            self.rad_sampling_offsets.bias.data = grid_init[:, 0].reshape(-1)
            constant_init(self.rad_attention_weights, val=0., bias=0.)
            xavier_init(self.rad_value_proj, distribution='uniform', bias=0.)
            xavier_init(self.rad_output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                reference_points=None,
                pts_feats=None,
                key_padding_mask_lidar=None,
                spatial_shapes_lidar=None,
                level_start_index_lidar=None,
                img_feats=None,
                key_padding_mask=None,
                spatial_shapes=None,
                level_start_index=None,
                rad_feats=None,
                key_padding_mask_radar=None,
                spatial_shapes_radar=None,
                level_start_index_radar=None,
                **kwargs):
        """
        Args:
            query (Tensor): (B*N, num_query, embed_dims).
            reference_points (Tensor): (B*N, num_query, num_levels, 3),
                normalized (u, v, depth) in the image frame of each camera.
            pts_feats / img_feats / rad_feats (Tensor): Flattened multi-level
                LiDAR BEV / image / radar BEV features.
            kwargs: Must contain 'lidar2img', 'img_aug_mat' and 'img_shape'
                when LiDAR or radar is used.

        Returns:
            Tensor: (B*N, num_query, embed_dims).
        """
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bsn, num_query, _ = query.shape
        if self.use_lidar:
            # All queries of the N cameras attend to the shared LiDAR BEV.
            value = pts_feats
            bs, num_value, _ = value.shape
            num_query_all = self.num_cams * num_query

            assert (spatial_shapes_lidar[:, 0] *
                    spatial_shapes_lidar[:, 1]).sum() == num_value

            value = self.value_proj(value)
            if key_padding_mask_lidar is not None:
                value = value.masked_fill(key_padding_mask_lidar[..., None],
                                          0.0)
            value = value.view(bs, num_value, self.num_heads, -1)
            query_lidar = query.view(bs, self.num_cams, num_query, -1)
            query_lidar = query_lidar.permute(0, 2, 1, 3).reshape(
                bs, num_query_all, -1)
            sampling_offsets = self.sampling_offsets(query_lidar).view(
                bs, num_query_all, self.num_heads, self.num_levels,
                self.num_points, 2)
            attention_weights = self.attention_weights(query_lidar).view(
                bs, num_query_all, self.num_heads,
                self.num_levels * self.num_points)
            attention_weights = attention_weights.softmax(-1)
            attention_weights = attention_weights.view(bs, num_query_all,
                                                       self.num_heads,
                                                       self.num_levels,
                                                       self.num_points)

            # image-frame reference points -> normalized LiDAR frame
            ref_points = image_to_lidar_normalized(
                reference_points[..., :3], self.pc_range, kwargs['lidar2img'],
                kwargs['img_aug_mat'], kwargs['img_shape'],
                self.depth_limit_min, self.depth_limit)
            ref_points = ref_points.view(bs, self.num_cams, num_query,
                                         self.num_levels, 3)
            ref_points = ref_points.permute(0, 2, 1, 3, 4).reshape(
                bs, num_query_all, self.num_levels, 3)
            ref_points = ref_points[..., :2]
            offset_normalizer = torch.stack(
                [spatial_shapes_lidar[..., 1], spatial_shapes_lidar[..., 0]],
                -1)
            sampling_locations = ref_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
            output = deformable_attn(value, spatial_shapes_lidar,
                                     level_start_index_lidar,
                                     sampling_locations, attention_weights,
                                     self.im2col_step)
            pts_output = self.output_proj(output)
            pts_output = pts_output.view(bs, num_query, self.num_cams, -1)
            pts_output = pts_output.permute(0, 2, 1, 3).reshape(
                bs * self.num_cams, num_query, -1)

        if self.use_camera:
            # Each camera's queries attend to the features of that camera.
            value = img_feats
            bs, num_value, _ = value.shape  # bs = B*N
            value = self.img_value_proj(value)
            if key_padding_mask is not None:
                value = value.masked_fill(key_padding_mask[..., None], 0.0)
            value = value.view(bs, num_value, self.num_heads, -1)

            sampling_offsets = self.img_sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels,
                self.num_points, 2)
            attention_weights = self.img_attention_weights(query).view(
                bs, num_query, self.num_heads,
                self.num_levels * self.num_points)
            attention_weights = attention_weights.softmax(-1)
            attention_weights = attention_weights.view(bs, num_query,
                                                       self.num_heads,
                                                       self.num_levels,
                                                       self.num_points)

            reference_points_2d = reference_points[..., :2]
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points_2d[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
            output = deformable_attn(value, spatial_shapes, level_start_index,
                                     sampling_locations, attention_weights,
                                     self.im2col_step)
            img_output = self.img_output_proj(output)

        if self.use_radar:
            # All queries of the N cameras attend to the shared radar BEV.
            value = rad_feats
            bs, num_value, _ = value.shape
            num_query_all = self.num_cams * num_query

            assert (spatial_shapes_radar[:, 0] *
                    spatial_shapes_radar[:, 1]).sum() == num_value

            value = self.rad_value_proj(value)
            if key_padding_mask_radar is not None:
                value = value.masked_fill(key_padding_mask_radar[..., None],
                                          0.0)
            value = value.view(bs, num_value, self.num_heads, -1)

            query_radar = query.view(bs, self.num_cams, num_query, -1)
            query_radar = query_radar.permute(0, 2, 1, 3).reshape(
                bs, num_query_all, -1)
            sampling_offsets = self.rad_sampling_offsets(query_radar).view(
                bs, num_query_all, self.num_heads, 1, self.num_points, 2)
            # NOTE: the released models were trained with the attention
            # weights computed from `query` (camera-major order).
            attention_weights = self.rad_attention_weights(query).view(
                bs, num_query_all, self.num_heads, self.num_points)
            attention_weights = attention_weights.softmax(-1)
            attention_weights = attention_weights.view(bs, num_query_all,
                                                       self.num_heads, 1,
                                                       self.num_points)

            ref_points = image_to_lidar_normalized(
                reference_points[..., :3], self.radar_pc_range,
                kwargs['lidar2img'], kwargs['img_aug_mat'],
                kwargs['img_shape'], self.depth_limit_min, self.depth_limit)
            ref_points = ref_points[..., 0, :]
            ref_points = ref_points.view(bs, self.num_cams, num_query, 1, 3)
            ref_points = ref_points.permute(0, 2, 1, 3, 4).reshape(
                bs, num_query_all, 1, 3)
            ref_points = ref_points[..., :2]
            offset_normalizer = torch.stack(
                [spatial_shapes_radar[..., 1], spatial_shapes_radar[..., 0]],
                -1)
            sampling_locations = ref_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
            output = deformable_attn(value, spatial_shapes_radar,
                                     level_start_index_radar,
                                     sampling_locations, attention_weights,
                                     self.im2col_step)
            radar_output = self.rad_output_proj(output)
            radar_output = radar_output.view(bs, num_query, self.num_cams, -1)
            radar_output = radar_output.permute(0, 2, 1, 3).reshape(
                bs * self.num_cams, num_query, -1)

        if self.use_lidar and self.use_camera and self.use_radar:
            output = torch.cat((img_output, pts_output, radar_output), dim=2)
            output = self.modality_fusion_layer(output)
        elif self.use_lidar and self.use_camera:
            output = torch.cat((img_output, pts_output), dim=2)
            output = self.modality_fusion_layer(output)
        elif self.use_camera and self.use_radar:
            output = torch.cat((img_output, radar_output), dim=2)
            output = self.modality_fusion_layer(output)
        elif self.use_lidar:
            output = pts_output
        else:
            output = img_output

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


def image_to_lidar_normalized(reference_points, pc_range, lidar2img,
                              img_aug_mat, img_shape, depth_min, depth_max):
    """Convert normalized image-frame (u, v, depth) to normalized LiDAR-frame
    coordinates.

    Args:
        reference_points (Tensor): (B*N, num_query, num_levels, 3).
        pc_range (list[float]): [x_min, y_min, z_min, x_max, y_max, z_max].
        lidar2img (Tensor): (B, N, 4, 4).
        img_aug_mat (Tensor): (B, N, 4, 4).
        img_shape (tuple[int]): (H, W).
        depth_min, depth_max (float): Depth range of the normalized depth.

    Returns:
        Tensor: (B*N, num_query, num_levels, 3) in [0, 1] within pc_range.
    """
    B, N = lidar2img.shape[:2]
    BN, num_query, num_levels, _ = reference_points.shape
    assert B * N == BN, 'Mismatch in B*N and reference_points.shape[0]'

    device = reference_points.device
    H, W = img_shape

    xy = reference_points[..., :2]
    z_norm = reference_points[..., 2:3]
    z_abs = depth_min + (depth_max - depth_min) * z_norm

    x_pix = xy[..., 0:1] * W
    y_pix = xy[..., 1:2] * H

    # homogeneous image coordinates (u * d, v * d, d, 1)
    x_cam = x_pix * z_abs
    y_cam = y_pix * z_abs
    z_cam = z_abs
    ones = torch.ones_like(z_cam)
    points_cam = torch.cat([x_cam, y_cam, z_cam, ones], dim=-1)

    lidar2img = lidar2img.view(B * N, 4, 4).to(device)
    img_aug_mat = img_aug_mat.view(B * N, 4, 4).to(device)
    P_full = torch.bmm(img_aug_mat, lidar2img)
    P_inv = torch.inverse(P_full.float())

    points_cam_flat = points_cam.view(BN * num_query * num_levels, 4, 1)
    P_inv_flat = P_inv.unsqueeze(1).unsqueeze(1).expand(
        BN, num_query, num_levels, 4, 4)
    P_inv_flat = P_inv_flat.contiguous().view(BN * num_query * num_levels, 4,
                                              4)

    points_lidar = torch.bmm(P_inv_flat, points_cam_flat).view(
        BN, num_query, num_levels, 4)
    points_lidar = points_lidar[..., :3] / points_lidar[..., 3:].clamp(min=1e-6)

    pc_range = torch.tensor(pc_range, device=device)
    x_range = pc_range[3] - pc_range[0]
    y_range = pc_range[4] - pc_range[1]
    z_range = pc_range[5] - pc_range[2]

    points_lidar_norm = torch.empty_like(points_lidar)
    points_lidar_norm[..., 0] = (points_lidar[..., 0] - pc_range[0]) / x_range
    points_lidar_norm[..., 1] = (points_lidar[..., 1] - pc_range[1]) / y_range
    points_lidar_norm[..., 2] = (points_lidar[..., 2] - pc_range[2]) / z_range
    return points_lidar_norm
