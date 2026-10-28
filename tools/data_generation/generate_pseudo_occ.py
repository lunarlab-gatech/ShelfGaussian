"""Generate shelf-supervised 3D pseudo labels on nuScenes.

For every scene, LiDAR key frames are accumulated into a static scene (points
outside annotated boxes) plus per-object point clouds (points inside boxes,
re-posed per frame), following LOcc (https://github.com/pkqbajng/LOcc).
Only points visible in at least one camera are kept. The aggregated cloud is
then voxelized in the ego frame of each key frame:

* ``--type binary``: binary occupancy,
  saved to ``<save-path>/<scene>/<token>/label_binary.npz``.
* ``--type dinov2``: each LiDAR point takes the DINOv2 (ViT-B/14-reg) patch
  feature of the camera it projects to; voxel features are the mean of their
  points, compressed to 128-D with per-sample PCA, saved to
  ``<save-path>/<scene>/<token>/label_dinov2_pca128_tensor_reso04_098.pt``.

Usage (scenes can be split across processes with --start / --end):
    python tools/data_generation/generate_pseudo_occ.py --type binary
    python tools/data_generation/generate_pseudo_occ.py --type dinov2 --start 0 --end 425
"""
import argparse
import os

import numpy as np
import torch
from mmcv.ops.points_in_boxes import points_in_boxes_cpu
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image, ImageFile
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

CAMERA_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK',
    'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT'
]
IMG_W, IMG_H = 1600, 900

# Occ3D-nuScenes grid (ego frame)
PC_RANGE = [-40, -40, -1, 40, 40, 5.4]
VOXEL_SIZE = 0.4
OCC_SIZE = [200, 200, 16]
SELF_RANGE = [3.0, 3.0, 3.0]  # drop LiDAR points on the ego vehicle

# DINOv2 pseudo label settings
DINO_RESIZE = 0.98  # images are resized to (882, 1568) before DINOv2
DINO_DIM = 768
PCA_DIM = 128


def rt2mat(translation, quaternion, inverse=False):
    R = Quaternion(quaternion).rotation_matrix
    T = np.array(translation)
    if inverse:
        R = R.T
        T = -R @ T
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat


def lidar_to_world_to_lidar(pc, lidar_calibrated_sensor, lidar_ego_pose,
                            cam_calibrated_sensor, cam_ego_pose):
    """Transform (N, 4) points from one LiDAR frame to another via world."""
    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)
    return pc


def lidar2vehicle(pc, lidar_calibrated_sensor):
    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))
    return pc


def get_cam_info(nusc, sample_token):
    """Return ego->image projection matrices and image paths of a sample."""
    rec = nusc.get('sample', sample_token)
    proj_matrix, filenames = [], []
    for cam_name in CAMERA_NAMES:
        cam_sample = nusc.get('sample_data', rec['data'][cam_name])
        filenames.append(cam_sample['filename'])
        cs = nusc.get('calibrated_sensor', cam_sample['calibrated_sensor_token'])
        ego2cam = rt2mat(cs['translation'], cs['rotation'],
                         inverse=True).astype(np.float32)
        K = np.eye(4).astype(np.float32)
        K[:3, :3] = cs['camera_intrinsic']
        proj_matrix.append(np.dot(K, ego2cam))
    return proj_matrix, filenames


def vehicle2img(coords, proj_matrix):
    """Assign each ego-frame point to the camera where it projects closest to
    the image center.

    Returns:
        proj_info (N, 3): (camera index, u, v), -1 if not visible.
        proj_mask (N,): whether the point is visible in any camera.
    """
    num_points = coords.shape[0]
    coords = np.concatenate((coords, np.ones((num_points, 1))), axis=-1)
    center = (IMG_W / 2, IMG_H / 2)

    all_uv, all_dist = [], []
    for proj in proj_matrix:
        img_coords = np.dot(proj, coords.T)
        uv = (img_coords[:2, :] / img_coords[2, :]).T
        depth = img_coords[2, :]
        valid = ((uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) &
                 (uv[:, 1] < IMG_H) & (depth > 0))
        dist = (uv[:, 0] - center[0])**2 + (uv[:, 1] - center[1])**2
        all_uv.append(uv)
        all_dist.append(np.where(valid, dist, np.inf))
    all_uv = np.stack(all_uv)  # (num_cams, N, 2)
    all_dist = np.stack(all_dist)  # (num_cams, N)

    # argmin keeps the first camera on ties
    best_cam = np.argmin(all_dist, axis=0)
    visible = np.isfinite(all_dist[best_cam, np.arange(num_points)])

    proj_info = np.full((num_points, 3), -1.0)
    proj_info[visible, 0] = best_cam[visible]
    proj_info[visible, 1:] = all_uv[best_cam[visible], visible]
    proj_mask = ~((proj_info[:, 0] < 0) | (proj_info[:, 1] < 0) |
                  (proj_info[:, 2] < 0))
    return proj_info, proj_mask


@torch.no_grad()
def get_dino_feats(proj_img_index, proj_uv, filenames, dino_backbone,
                   data_root, device):
    """Sample the DINOv2 patch feature at each point's projection."""
    patch_size = dino_backbone.patch_size
    H, W = int(IMG_H * DINO_RESIZE), int(IMG_W * DINO_RESIZE)
    mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

    imgs = []
    for filename in filenames:
        img = Image.open(os.path.join(data_root, filename)).convert('RGB')
        img = img.resize((W, H), Image.BILINEAR)
        img = torch.from_numpy(np.array(img).astype(np.float32)).permute(
            2, 0, 1)
        imgs.append(((img - mean) / std).unsqueeze(0))
    imgs = torch.cat(imgs, dim=0).to(device)

    feats = dino_backbone.forward_features(imgs)['x_norm_patchtokens']
    B, _, C = feats.shape
    feats = feats.transpose(1, 2).reshape(B, C, H // patch_size,
                                          W // patch_size)
    feats = feats.permute(0, 2, 3, 1).cpu().numpy()  # (B, Hp, Wp, C)

    u = proj_uv[:, 0] * (W / IMG_W)
    v = proj_uv[:, 1] * (H / IMG_H)
    patch_u = (u // patch_size).astype(np.int32)
    patch_v = (v // patch_size).astype(np.int32)
    return feats[proj_img_index, patch_v, patch_u]  # (N, C)


def save_pca(occ_feats, save_dir, device):
    """Compress (X, Y, Z, C) voxel features with PCA and save them."""
    data = torch.from_numpy(occ_feats).float().to(device)
    X, Y, Z, C = data.shape
    feats_2d = data.flatten(0, 2)
    mean_per_dim = feats_2d.mean(dim=0, keepdim=True)
    centered = feats_2d - mean_per_dim
    _, _, v = torch.pca_lowrank(
        centered.double(), q=PCA_DIM, niter=4, center=False)
    compressed = (centered @ v.to(data)).view(X, Y, Z, PCA_DIM)
    torch.save(
        {
            'pca_data': compressed.detach().cpu().contiguous(),
            'pca_v': v.detach().cpu().contiguous(),
            'pca_mean': mean_per_dim.detach().cpu().contiguous()
        },
        os.path.join(save_dir,
                     f'label_dinov2_pca{PCA_DIM}_tensor_reso04_098.pt'))


def get_boxes(nusc, lidar_token):
    """Annotated boxes as (x, y, z_bottom, w, l, h, yaw), slightly enlarged."""
    _, boxes, _ = nusc.get_sample_data(lidar_token)
    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    rots = np.array([b.orientation.yaw_pitch_roll[0]
                     for b in boxes]).reshape(-1, 1)
    gt_bbox_3d = np.concatenate([locs, dims, rots],
                                axis=1).astype(np.float32)
    gt_bbox_3d[:, 6] += np.pi / 2.
    gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
    gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # move boxes slightly down
    gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1  # wrap all object points
    return boxes, gt_bbox_3d


def process_scene(nusc, scene, args, dino_backbone=None, device=None):
    use_dino = args.type == 'dinov2'
    lidar_data = nusc.get(
        'sample_data',
        nusc.get('sample', scene['first_sample_token'])['data']['LIDAR_TOP'])
    lidar_ego_pose0 = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    lidar_calibrated_sensor0 = nusc.get('calibrated_sensor',
                                        lidar_data['calibrated_sensor_token'])

    # 1. Collect per-key-frame static points and object points
    frames = []
    while True:
        if lidar_data['is_key_frame']:
            boxes, gt_bbox_3d = get_boxes(nusc, lidar_data['token'])
            object_tokens = [
                nusc.get('sample_annotation', box.token)['instance_token']
                for box in boxes
            ]

            pc0 = np.fromfile(
                os.path.join(args.data_root, lidar_data['filename']),
                dtype=np.float32).reshape(-1, 5)[..., :4]
            # (x, y, z, 0) in float64; the 4th column is only a placeholder
            pc = np.concatenate([pc0[:, :3], np.zeros_like(pc0[:, :1])],
                                axis=1).astype(np.float64)

            # points in any box are dynamic objects, the rest is static
            points_in_boxes = points_in_boxes_cpu(
                torch.from_numpy(pc0[:, :3][np.newaxis, :, :]),
                torch.from_numpy(gt_bbox_3d[np.newaxis, :]))
            each_box_masks = points_in_boxes[0]  # (N, M)
            points_mask = ~(points_in_boxes.sum(dim=-1).bool()[0])

            oneself_mask = torch.from_numpy(
                (np.abs(pc0[:, 0]) > SELF_RANGE[0]) |
                (np.abs(pc0[:, 1]) > SELF_RANGE[1]) |
                (np.abs(pc0[:, 2]) > SELF_RANGE[2]))

            lidar_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            lidar_calibrated_sensor = nusc.get(
                'calibrated_sensor', lidar_data['calibrated_sensor_token'])

            # keep only points visible in some camera
            vehicle_coords = lidar2vehicle(
                pc.copy(), lidar_calibrated_sensor).points.T[:, :3]
            sample_token = lidar_data['sample_token']
            proj_matrix, filenames = get_cam_info(nusc, sample_token)
            proj_info, proj_mask = vehicle2img(vehicle_coords, proj_matrix)
            proj_mask = torch.from_numpy(proj_mask)
            if use_dino:
                dino_feats = get_dino_feats(
                    proj_info[:, 0].astype(np.int16), proj_info[:, 1:],
                    filenames, dino_backbone, args.data_root, device)

            points_mask = points_mask & oneself_mask & proj_mask

            # static points, expressed in the first frame's LiDAR coordinates
            static_pc = lidar_to_world_to_lidar(
                pc[points_mask].copy(), lidar_calibrated_sensor.copy(),
                lidar_ego_pose.copy(), lidar_calibrated_sensor0,
                lidar_ego_pose0)

            object_points_list, object_feats_list = [], []
            for j in range(each_box_masks.shape[-1]):
                box_mask = each_box_masks[:, j].bool() & oneself_mask & proj_mask
                object_points_list.append(pc[box_mask])
                if use_dino:
                    object_feats_list.append(dino_feats[box_mask])

            frames.append(
                dict(
                    object_tokens=object_tokens,
                    object_points_list=object_points_list,
                    object_feats_list=object_feats_list,
                    static_points=static_pc.points,
                    static_feats=dino_feats[points_mask] if use_dino else None,
                    lidar_ego_pose=lidar_ego_pose,
                    lidar_calibrated_sensor=lidar_calibrated_sensor,
                    lidar_token=lidar_data['token'],
                    gt_bbox_3d=gt_bbox_3d,
                    occ_token=sample_token,
                    scene_name=scene['name']))

        if lidar_data['next'] == '':
            break
        lidar_data = nusc.get('sample_data', lidar_data['next'])

    # 2. Accumulate the static scene over all key frames
    static_points = np.concatenate([f['static_points'] for f in frames],
                                   axis=1).T  # first-frame LiDAR coords
    if use_dino:
        static_feats = np.concatenate([f['static_feats'] for f in frames],
                                      axis=0)

    # 3. Accumulate each object in its own box frame
    object_token_zoo = []
    for f in frames:
        for i, token in enumerate(f['object_tokens']):
            if token not in object_token_zoo and \
                    f['object_points_list'][i].shape[0] > 0:
                object_token_zoo.append(token)

    object_points_vertice, object_feats_vertice = [], []
    for query_token in object_token_zoo:
        points, feats = [], []
        for f in frames:
            for i, token in enumerate(f['object_tokens']):
                if token != query_token:
                    continue
                object_points = f['object_points_list'][i]
                if object_points.shape[0] > 0:
                    object_points = object_points[:, :3] - f['gt_bbox_3d'][i][:3]
                    rot = Rotation.from_euler(
                        'z', -f['gt_bbox_3d'][i][6], degrees=False)
                    points.append(rot.apply(object_points))
                    if use_dino:
                        feats.append(f['object_feats_list'][i])
        object_points_vertice.append(np.concatenate(points, axis=0)[:, :3])
        if use_dino:
            object_feats_vertice.append(np.concatenate(feats, axis=0))

    # 4. For each key frame, place static scene + objects and voxelize
    for f in frames:
        lidar_calibrated_sensor = f['lidar_calibrated_sensor']
        scene_points = lidar_to_world_to_lidar(
            static_points.copy(), lidar_calibrated_sensor0.copy(),
            lidar_ego_pose0.copy(), lidar_calibrated_sensor,
            f['lidar_ego_pose']).points.T  # (N, 4) in this frame's LiDAR

        _, gt_bbox_3d = get_boxes(nusc, f['lidar_token'])
        rots = gt_bbox_3d[:, 6:7]
        locs = gt_bbox_3d[:, 0:3]

        object_points_list, object_feats_list = [], []
        for j, token in enumerate(f['object_tokens']):
            for k, token_in_zoo in enumerate(object_token_zoo):
                if token != token_in_zoo:
                    continue
                points = object_points_vertice[k]
                if use_dino:
                    feats = object_feats_vertice[k]
                rot = Rotation.from_euler('z', rots[j], degrees=False)
                points = rot.apply(points) + locs[j]
                if points.shape[0] >= 5:
                    in_box = points_in_boxes_cpu(
                        torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                        torch.from_numpy(gt_bbox_3d[j:j + 1][np.newaxis, :]))
                    in_box = in_box[0, :, 0].bool()
                    points = points[in_box]
                    if use_dino:
                        feats = feats[in_box]
                object_points_list.append(
                    np.concatenate(
                        [points[:, :3], np.zeros_like(points[:, :1])],
                        axis=1))
                if use_dino:
                    object_feats_list.append(feats)

        point_feats = static_feats if use_dino else None
        if object_points_list:
            scene_points = np.concatenate(
                [scene_points, np.concatenate(object_points_list)])
            if use_dino:
                point_feats = np.concatenate(
                    [static_feats, np.concatenate(object_feats_list)], axis=0)

        # LiDAR -> ego, crop to the occupancy range
        scene_points = lidar2vehicle(scene_points,
                                     lidar_calibrated_sensor).points.T
        mask = ((np.abs(scene_points[:, 0]) < 40.0) &
                (np.abs(scene_points[:, 1]) < 40.0) &
                (scene_points[:, 2] > -1.0) & (scene_points[:, 2] < 5.4))
        scene_points = scene_points[mask]
        if use_dino:
            point_feats = point_feats[mask]

        voxel_coords = scene_points[:, :3]
        voxel_coords[:, 0] = (voxel_coords[:, 0] - PC_RANGE[0]) / VOXEL_SIZE
        voxel_coords[:, 1] = (voxel_coords[:, 1] - PC_RANGE[1]) / VOXEL_SIZE
        voxel_coords[:, 2] = (voxel_coords[:, 2] - PC_RANGE[2]) / VOXEL_SIZE
        voxel_coords = np.floor(voxel_coords).astype(np.int32)

        save_dir = os.path.join(args.save_path, f['scene_name'],
                                f['occ_token'])
        os.makedirs(save_dir, exist_ok=True)

        if not use_dino:
            occ = np.zeros(OCC_SIZE).astype(np.uint8)
            occ_index = np.unique(voxel_coords, axis=0)
            occ[occ_index[:, 0], occ_index[:, 1], occ_index[:, 2]] = 1
            np.savez_compressed(
                os.path.join(save_dir, 'label_binary.npz'),
                pseudo_binary_gt=occ)
        else:
            # mean-pool point features inside each occupied voxel
            occ_index, inverse_indices = np.unique(
                voxel_coords, axis=0, return_inverse=True)
            occ_feats = np.zeros((*OCC_SIZE, DINO_DIM), dtype=np.float32)
            occ_accum = np.zeros((occ_index.shape[0], DINO_DIM),
                                 dtype=np.float32)
            occ_counts = np.zeros(occ_index.shape[0], dtype=np.int32)
            np.add.at(occ_accum, inverse_indices, point_feats)
            np.add.at(occ_counts, inverse_indices, 1)
            occ_counts = np.maximum(occ_counts, 1)[:, None]
            occ_feats[tuple(occ_index.T)] = occ_accum / occ_counts
            save_pca(occ_feats, save_dir, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['binary', 'dinov2'], required=True)
    parser.add_argument('--data-root', default='data/nuscenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument(
        '--save-path',
        default=None,
        help='defaults to data/nuscenes_pseudo_{binary,dinov2}_gt')
    parser.add_argument('--start', type=int, default=0, help='first scene')
    parser.add_argument('--end', type=int, default=850, help='last scene + 1')
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args()
    if args.save_path is None:
        args.save_path = f'data/nuscenes_pseudo_{args.type}_gt'

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)

    dino_backbone, device = None, None
    if args.type == 'dinov2':
        device = torch.device(
            f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        dino_backbone = torch.hub.load('facebookresearch/dinov2',
                                       'dinov2_vitb14_reg').to(device)
        dino_backbone.requires_grad_(False)
        dino_backbone.eval()

    for i in tqdm(range(args.start, min(args.end, len(nusc.scene)))):
        process_scene(nusc, nusc.scene[i], args, dino_backbone, device)


if __name__ == '__main__':
    main()
