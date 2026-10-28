import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmdet.models import (MLP, DetrTransformerDecoder,
                          DetrTransformerDecoderLayer)
from mmdet3d.registry import MODELS

from .multimodal_attention import MultiModalAttention
from .utils import gen_sineembed_for_position, inverse_sigmoid


@MODELS.register_module()
class FusionGaussianDecoder(DetrTransformerDecoder):
    """Decoder refining per-camera Gaussian queries.

    Each query has a 3D reference point in its camera's image frame
    (normalized u, v, depth). Its positional embedding is a sine embedding of
    the reference point, and the reference point is refined after every layer
    by the regression branch of the corresponding Gaussian head.
    """

    def __init__(self, return_intermediate=True, anchor_size=3, **kwargs):
        super().__init__(**kwargs)
        self.return_intermediate = return_intermediate
        self.anchor_size = anchor_size
        self.ref_point_head = MLP(self.embed_dims // 2 * self.anchor_size,
                                  self.embed_dims, self.embed_dims, 2)

    def _init_layers(self):
        self.layers = nn.ModuleList([
            FusionGaussianDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self,
                query,
                query_pos,
                value,
                value_lidar,
                value_radar,
                key_padding_mask,
                key_padding_mask_lidar,
                key_padding_mask_radar,
                reference_points,
                spatial_shapes,
                spatial_shapes_lidar,
                spatial_shapes_radar,
                level_start_index,
                level_start_index_lidar,
                level_start_index_radar,
                valid_ratios,
                valid_ratios_lidar,
                valid_ratios_radar,
                reg_branches=None,
                **kwargs):
        """
        Args:
            query (Tensor): (B*N, num_queries, embed_dims).
            reference_points (Tensor): (B*N, num_queries, 3), normalized
                (u, v, depth) in the image frame.
            value / value_lidar / value_radar (Tensor): Flattened image /
                LiDAR BEV / radar BEV features (None if unused).
            reg_branches (list[nn.Module]): Regression heads predicting the
                reference point offsets after each layer.

        Returns:
            tuple[Tensor]: Stacked outputs and reference points of all
            layers if `return_intermediate`, otherwise those of the last one.
        """
        assert query_pos is None
        assert reference_points.shape[-1] == 3

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = \
                reference_points[:, :, None] * valid_ratios[:, None]
            query_sine_embed = gen_sineembed_for_position(
                reference_points.permute(1, 0, 2))
            query_pos = self.ref_point_head(query_sine_embed).permute(1, 0, 2)

            query = layer(
                query, # (B*N, num_queries, embed_dims)
                query_pos=query_pos, # (B*N, num_queries, C)
                value=value, # (bs=B*N, sum(H_l*W_l), C_l)
                value_lidar=value_lidar, # (bs=B*N, sum(H_l*W_l), C_l)
                value_radar=value_radar, # (bs=B*N, sum(H_l*W_l), C_l)
                key_padding_mask=key_padding_mask, # None
                key_padding_mask_lidar=key_padding_mask_lidar, # None
                key_padding_mask_radar=key_padding_mask_radar, # None
                spatial_shapes=spatial_shapes, # (num_level, 2)
                spatial_shapes_lidar=spatial_shapes_lidar, # (num_level, 2)
                spatial_shapes_radar=spatial_shapes_radar, # (num_level, 2)
                level_start_index=level_start_index, # (num_feature_levels,)
                level_start_index_lidar=level_start_index_lidar, # (num_feature_levels,)
                level_start_index_radar=level_start_index_radar, # (num_feature_levels,)
                valid_ratios=valid_ratios, # (bs=B*N, num_level, 2), all 1
                valid_ratios_lidar=valid_ratios_lidar, # (bs=B*N, num_level, 2), all 1
                valid_ratios_radar=valid_ratios_radar, # (bs=B*N, num_level, 2), all 1
                reference_points=reference_points_input, # (B*N, num_queries, num_levels, 2)
                **kwargs)

            if reg_branches is not None:
                tmp_reg_preds = reg_branches[lid](query)[..., :3]
                new_reference_points = tmp_reg_preds + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid() # to [0, 1]
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points


class FusionGaussianDecoderLayer(DetrTransformerDecoderLayer):

    def _init_layers(self) -> None:
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiModalAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = nn.ModuleList(norms_list)

    def forward(self,
                query,
                key=None,
                value=None,
                value_lidar=None,
                value_radar=None,
                query_pos=None,
                key_pos=None,
                self_attn_mask=None,
                cross_attn_mask=None,
                key_padding_mask=None,
                key_padding_mask_lidar=None,
                key_padding_mask_radar=None,
                spatial_shapes=None,
                spatial_shapes_lidar=None,
                spatial_shapes_radar=None,
                level_start_index=None,
                level_start_index_lidar=None,
                level_start_index_radar=None,
                valid_ratios=None,
                valid_ratios_lidar=None,
                valid_ratios_radar=None,
                reference_points=None,
                **kwargs):
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        # only need to change cross attn since only cross attn involves value
        query = self.cross_attn(
            query=query,
            key=key,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            # camera
            img_feats=value,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # lidar
            pts_feats=value_lidar,
            key_padding_mask_lidar=key_padding_mask_lidar,
            spatial_shapes_lidar=spatial_shapes_lidar,
            level_start_index_lidar=level_start_index_lidar,
            valid_ratios_lidar=valid_ratios_lidar,
            # radar
            rad_feats=value_radar,
            key_padding_mask_radar=key_padding_mask_radar,
            spatial_shapes_radar=spatial_shapes_radar,
            level_start_index_radar=level_start_index_radar,
            valid_ratios_radar=valid_ratios_radar,
            # reference points
            reference_points=reference_points,
            **kwargs)
        query = self.norms[0](query)
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query

