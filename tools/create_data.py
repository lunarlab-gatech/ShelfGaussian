"""Add scene info and LiDAR / radar sweeps to the mmdet3d nuScenes infos.

Run mmdet3d's converter first, which writes nuscenes_infos_{train,val}.pkl:
    mim run mmdet3d create_data nuscenes --root-path ./data/nuscenes \
        --out-dir ./data/nuscenes --extra-tag nuscenes
Then:
    python tools/create_data.py --root-path ./data/nuscenes

This writes nuscenes_infos_{train,val}_lidar_radar_sweeps.pkl, adding to each
sample: 'scene_token', 'scene_idx' (scene name, used to locate the Occ3D and
pseudo labels), 'prev_token', 'sweeps' (up to 10 previous LiDAR sweeps) and
'radars' (10 sweeps for each of the 5 radars).
"""
import argparse
import os
from os import path as osp

import mmengine
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion

RADAR_NAMES = [
    'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT',
    'RADAR_BACK_RIGHT'
]


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.
    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def update_nuscenes_infos(pkl_path,
                          out_path,
                          root_path,
                          max_sweeps=10,
                          max_radar_sweeps=10):
    data = mmengine.load(pkl_path)
    nusc = NuScenes(
        version=data['metainfo']['version'], dataroot=root_path, verbose=True)

    print(f'Updating {pkl_path}:')
    for i, info in enumerate(mmengine.track_iter_progress(data['data_list'])):
        sample = nusc.get('sample', info['token'])
        scene = nusc.get('scene', sample['scene_token'])
        info['scene_token'] = sample['scene_token']
        info['scene_idx'] = scene['name']
        info['prev_token'] = sample['prev']

        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, _, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'])
        mmengine.check_file_exist(lidar_path)

        l2e_t = cs_record['translation']
        e2g_t = pose_record['translation']
        l2e_r_mat = Quaternion(cs_record['rotation']).rotation_matrix
        e2g_r_mat = Quaternion(pose_record['rotation']).rotation_matrix

        # previous LiDAR sweeps
        sweeps = []
        while len(sweeps) < max_sweeps:
            if sd_rec['prev'] == '':
                break
            sweeps.append(
                obtain_sensor2top(nusc, sd_rec['prev'], l2e_t, l2e_r_mat,
                                  e2g_t, e2g_r_mat, 'lidar'))
            sd_rec = nusc.get('sample_data', sd_rec['prev'])
        info['sweeps'] = sweeps

        # radar sweeps (current + previous); the earliest sweep is repeated
        # at the beginning of a scene
        radars = dict()
        for radar_name in RADAR_NAMES:
            radar_token = sample['data'][radar_name]
            radar_rec = nusc.get('sample_data', radar_token)
            sweeps = []
            while len(sweeps) < max_radar_sweeps:
                sweeps.append(
                    obtain_sensor2top(nusc, radar_token, l2e_t, l2e_r_mat,
                                      e2g_t, e2g_r_mat, radar_name))
                if radar_rec['prev'] != '':
                    radar_token = radar_rec['prev']
                    radar_rec = nusc.get('sample_data', radar_token)
            radars[radar_name] = sweeps
        info['radars'] = radars

    print(f'Writing to {out_path}')
    mmengine.dump(data, out_path, 'pkl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', default='./data/nuscenes')
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--extra-tag', default='nuscenes')
    parser.add_argument('--max-sweeps', type=int, default=10)
    parser.add_argument('--max-radar-sweeps', type=int, default=10)
    args = parser.parse_args()
    out_dir = args.out_dir or args.root_path

    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')
    for split in ['train', 'val']:
        update_nuscenes_infos(
            pkl_path=osp.join(out_dir, f'{args.extra_tag}_infos_{split}.pkl'),
            out_path=osp.join(
                out_dir,
                f'{args.extra_tag}_infos_{split}_lidar_radar_sweeps.pkl'),
            root_path=args.root_path,
            max_sweeps=args.max_sweeps,
            max_radar_sweeps=args.max_radar_sweeps)


if __name__ == '__main__':
    main()
