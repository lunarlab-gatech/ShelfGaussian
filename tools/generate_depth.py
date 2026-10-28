"""Estimate metric depth for every nuScenes camera image with Metric3D v2.

Writes data/nuscenes_metric3d/<image_name>.npy (900 x 1600 float32 depth).

Usage:
    PYTHONPATH=. python tools/generate_depth.py
"""
import os
import os.path as osp

import numpy as np
import torch
from mmengine import FUNCTIONS
from mmengine.registry import init_default_scope
from PIL import ImageFile
from rich.progress import track
from torch.utils.data import DataLoader

try:
    from torchvision.transforms import v2 as T
except ImportError:
    from torchvision import transforms as T

from gausstr import *  # noqa: F401,F403

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT = 'data/nuscenes/'
ANN_FILES = [
    'nuscenes_infos_train_lidar_radar_sweeps.pkl',
    'nuscenes_infos_val_lidar_radar_sweeps.pkl',
]
SAVE_DIR = 'data/nuscenes_metric3d'

DATASET_CFG = dict(
    type='NuScenesOccDataset',
    data_root=DATA_ROOT,
    modality=dict(use_camera=True, use_lidar=False),
    data_prefix=dict(
        CAM_FRONT='samples/CAM_FRONT',
        CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
        CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
        CAM_BACK='samples/CAM_BACK',
        CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
        CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
    filter_empty_gt=False,
    pipeline=[
        dict(
            type='BEVLoadMultiViewImageFromFiles',
            to_float32=True,
            color_type='color',
            num_views=6),
        dict(
            type='Pack3DDetInputs',
            keys=['img'],
            meta_keys=['cam2img', 'img_aug_mat', 'sample_idx', 'img_path'])
    ])


def test_loop(model, dataset_cfg, save_dir):
    from mmdet3d.registry import DATASETS
    dataset = DATASETS.build(dataset_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=FUNCTIONS.get('pseudo_collate'))
    transform = T.Normalize(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

    for data in track(dataloader):
        data_samples = data['data_samples']
        cam2imgs = []
        img_paths = []
        for i in range(len(data_samples)):
            cam2imgs.append(data_samples[i].cam2img)
            img_paths.append(data_samples[i].img_path)
        cam2imgs = torch.from_numpy(np.concatenate(cam2imgs)).cuda()
        img_paths = sum(img_paths, [])
        x = transform(torch.cat(data['inputs']['img']).cuda())

        with torch.no_grad():
            depths = model(x, cam2imgs)
        depths = depths.cpu().numpy()
        for path, depth in zip(img_paths, depths):
            np.save(
                osp.join(save_dir,
                         path.split('/')[-1].split('.')[0]), depth)


if __name__ == '__main__':
    init_default_scope('mmdet3d')
    from mmdet3d.registry import MODELS
    model = MODELS.build(
        dict(type='Metric3D', model_name='metric3d_vit_large')).cuda()
    os.makedirs(SAVE_DIR, exist_ok=True)
    for ann_file in ANN_FILES:
        test_loop(model, dict(DATASET_CFG, ann_file=ann_file), SAVE_DIR)
