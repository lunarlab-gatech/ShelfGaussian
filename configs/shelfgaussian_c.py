_base_ = 'mmdet3d::_base_/default_runtime.py'

custom_imports = dict(imports=['shelfgaussian'])

input_size = (378, 672)
embed_dims = 256
feat_dims = 768
reduce_dims = 128
patch_size = 14
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
use_lidar = False
use_camera = True
use_radar = False
num_queries = 1000
depth_limit = 54.0
depth_limit_min = 3.0
ego_point_range = [-40, -40, -1, 40, 40, 5.4]

find_unused_parameters = True

model = dict(
    type='FusionGaussian',
    use_lidar=use_lidar,
    depth_limit=depth_limit,
    depth_limit_min=depth_limit_min,
    use_radar=use_radar,
    num_queries=num_queries,
    embed_dims=embed_dims,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    img_backbone=dict(
        type='TorchHubModel',
        repo_or_dir='facebookresearch/dinov2',
        model_name='dinov2_vitb14_reg'),
    img_neck=dict(
        type='ViTDetFPN',
        in_channels=feat_dims,
        out_channels=embed_dims,
        norm_cfg=dict(type='LN2d')),
    decoder=dict(
        type='FusionGaussianDecoder',
        num_layers=3,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=embed_dims, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(use_lidar=use_lidar,
                                use_camera=use_camera,
                                use_radar=use_radar,
                                embed_dims=embed_dims,
                                num_cams=6,
                                num_heads=8,
                                num_levels=4,
                                pc_range=point_cloud_range,
                                depth_limit=depth_limit,
                                depth_limit_min=depth_limit_min),
            ffn_cfg=dict(embed_dims=embed_dims, feedforward_channels=2048)),
        post_norm_cfg=None),
    gauss_head=dict(
        type='FusionGaussianHead_VoxelSup',
        opacity_head=dict(
            type='MLP', input_dim=embed_dims, output_dim=1, mode='sigmoid'),
        feature_head=dict(
            type='MLP', input_dim=embed_dims, output_dim=feat_dims),
        scale_head=dict(
            type='MLP',
            input_dim=embed_dims,
            output_dim=3,
            mode='sigmoid',
            range=(1, 16)),
        regress_head=dict(type='MLP', input_dim=embed_dims, output_dim=3),
        text_protos='ckpts/text_proto_embeds_talk2dino_customized.pth',
        reduce_dims=reduce_dims,
        image_shape=input_size,
        patch_size=patch_size,
        depth_limit=depth_limit,
        depth_limit_min=depth_limit_min,
        voxelizer_train=dict(
            type='CudaGaussianVoxelizer',
            voxel_range=[-40, -40, -1, 40, 40, 5.4],
            voxel_size=[0.4, 0.4, 0.4]),
        ego_point_range=ego_point_range,
        ego_grid_shape=[200, 200, 16],
        ego_grid_size=0.4,
        use_ce_loss=True,
        use_feat_loss=True,
        loss_bce_weight=8.0,
        loss_feat_weight=16.0,
        pred_logits_thres=0.4,
            ))

dataset_type = 'NuScenesOccDataset'
data_root = 'data/nuscenes/'
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    sweeps='sweeps/LIDAR_TOP')
input_modality = dict(use_camera=use_camera, use_lidar=use_lidar, use_radar=use_radar)

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        num_views=6),
    dict(
        type='LoadMultiViewDepthFromFiles',
        is_to_depth_map=True,
        map_size=None,
        resize_on_raw=None),
    dict(type='LoadOccFromFile'),
    dict(type='LoadBinaryPseudoOccFromFile',
        occ_data_path='data/nuscenes_pseudo_binary_gt'),
    dict(type='LoadDINOv2PseudoOccFromFilePCA128',
        occ_data_path='data/nuscenes_pseudo_dinov2_gt'),
    dict(
        type='ImageAug3D',
        final_dim=input_size,
        resize_lim=[0.42, 0.42],
        is_train=True,
        transform_dpt=True),
    dict(
        type='LoadFeatMaps',
        data_root='data/nuscenes_metric3d',
        key='depth',
        apply_aug=True),
    dict(
        type='FuseDepthMaps',
        apply_filter=False),
    dict(
        type='Pack3DDetInputs',
        keys=['img'],
        meta_keys=[
            'cam2img', 'cam2ego', 'lidar2cam', 'lidar2img', 'ego2global', 'img_aug_mat', 'sample_idx',
            'num_views', 'img_path', 'depth', 'feats', 'mask_camera', 'resize_img_shape',
            'pseudo_binary_gt', 'pseudo_dino_gt', 'pseudo_dino_gt_v'
        ])
]
test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        num_views=6),
    dict(
        type='LoadMultiViewDepthFromFiles',
        is_to_depth_map=True,
        map_size=None,
        resize_on_raw=None),
    dict(type='LoadOccFromFile'),
    dict(type='ImageAug3D',
         final_dim=input_size,
         resize_lim=[0.42, 0.42],
         transform_dpt=True),
    dict(
        type='LoadFeatMaps',
        data_root='data/nuscenes_metric3d',
        key='depth',
        apply_aug=True),
    dict(
        type='FuseDepthMaps',
        apply_filter=False),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=[
            'cam2img', 'cam2ego', 'lidar2cam', 'lidar2img', 'ego2global', 'img_aug_mat', 'sample_idx',
            'num_views', 'img_path', 'depth', 'feats', 'mask_camera', 'resize_img_shape'
        ])
]

shared_dataset_cfg = dict(
    type=dataset_type,
    data_root=data_root,
    modality=input_modality,
    data_prefix=data_prefix,
    filter_empty_gt=False)

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        ann_file='nuscenes_infos_train_lidar_radar_sweeps.pkl',
        pipeline=train_pipeline,
        **shared_dataset_cfg))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        ann_file='nuscenes_infos_val_lidar_radar_sweeps.pkl',
        pipeline=test_pipeline,
        **shared_dataset_cfg))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='OccMetric',
    num_classes=18,
    use_lidar_mask=False,
    use_image_mask=True)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=5e-3),
    clip_grad=dict(max_norm=35, norm_type=2))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=200, by_epoch=False),
    dict(type='MultiStepLR', milestones=[16], gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=-1))

visualizer = dict(
    type='Visualizer', vis_backends=[dict(type='LocalVisBackend')])
