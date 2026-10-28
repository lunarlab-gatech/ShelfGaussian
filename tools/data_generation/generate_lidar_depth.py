"""Project key-frame LiDAR points onto the six cameras to get sparse depth.

Adapted from BEVDepth (https://github.com/Megvii-BaseDetection/BEVDepth,
scripts/gen_depth_gt.py, MIT License).

Each output file `<data_root>/depth_gt/<image_name>.jpg.bin` stores a float32
array of shape (N, 3) with rows (u, v, depth) in the original 1600x900 image.

Usage:
    python tools/data_generation/generate_lidar_depth.py --data-root data/nuscenes
"""
import argparse
import os
from multiprocessing import Pool

import mmengine
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

CAM_KEYS = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
    'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
]
IMG_W, IMG_H = 1600, 900


def map_pointcloud_to_image(points, lidar_calibrated_sensor, lidar_ego_pose,
                            cam_calibrated_sensor, cam_ego_pose,
                            min_dist=0.0):
    pc = LidarPointCloud(points.T)
    # lidar -> ego -> global
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))
    # global -> ego (at camera timestamp) -> camera
    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)
    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    depths = pc.points[2, :]
    pts_img = view_points(
        pc.points[:3, :],
        np.array(cam_calibrated_sensor['camera_intrinsic']),
        normalize=True)

    mask = depths > min_dist
    mask &= pts_img[0, :] > 1
    mask &= pts_img[0, :] < IMG_W - 1
    mask &= pts_img[1, :] > 1
    mask &= pts_img[1, :] < IMG_H - 1
    return pts_img[:2, mask], depths[mask]


NUSC = None  # set in main(); shared with forked workers


def worker(args):
    sample_token, data_root, out_dir = args
    nusc = NUSC
    sample = nusc.get('sample', sample_token)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    points = np.fromfile(
        os.path.join(data_root, lidar_sd['filename']),
        dtype=np.float32).reshape(-1, 5)[..., :4]
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    for cam_key in CAM_KEYS:
        cam_sd = nusc.get('sample_data', sample['data'][cam_key])
        cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        cam_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])
        pts_img, depth = map_pointcloud_to_image(points.copy(), lidar_cs,
                                                 lidar_pose, cam_cs, cam_pose)
        file_name = os.path.basename(cam_sd['filename'])
        np.concatenate([pts_img.T, depth[:, None]], axis=1).astype(
            np.float32).flatten().tofile(
                os.path.join(out_dir, f'{file_name}.bin'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/nuscenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    global NUSC
    NUSC = NuScenes(
        version=args.version, dataroot=args.data_root, verbose=True)
    out_dir = os.path.join(args.data_root, 'depth_gt')
    os.makedirs(out_dir, exist_ok=True)

    tasks = [(s['token'], args.data_root, out_dir) for s in NUSC.sample]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for _ in mmengine.track_iter_progress(
                    pool.imap_unordered(worker, tasks), len(tasks)):
                pass
    else:
        for t in mmengine.track_iter_progress(tasks):
            worker(t)


if __name__ == '__main__':
    main()
