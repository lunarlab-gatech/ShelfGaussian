import copy
import os
from typing import Optional, List, Sequence, Union

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
import mmengine
from mmengine.fileio import get
from PIL import Image

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import BasePoints, get_points_type
from .radar_points import RadarPoints
from nuscenes.utils.data_classes import RadarPointCloud

from mmengine.structures import InstanceData
from numpy import dtype
from mmdet3d.structures import BaseInstance3DBoxes, Det3DDataSample, PointData

from .loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams

# For the error: OSError: image file is truncated (18 bytes not processed)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        if data.dtype is dtype('float64'):
            data = data.astype(np.float32)
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # Support multi-view images with different shapes
        filename, cam2img, lidar2cam, cam2ego = [], [], [], []
        lidar2img = []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(np.array(cam_item['lidar2cam'], dtype=np.float32)) # 4x4

            cam2img_array = np.eye(4).astype(np.float32)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img']).astype(
                np.float32) # 3x3
            cam2img.append(cam2img_array)

            cam2ego_array = np.array(cam_item['cam2ego']).astype(np.float32)
            cam2ego.append(cam2ego_array)

            lidar2img_array = cam2img[-1] @ lidar2cam[-1] # 4x4
            lidar2img.append(lidar2img_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0) # (6, 4, 4)
        results['cam2ego'] = np.stack(cam2ego, axis=0) # (6, 4, 4)
        results['lidar2img'] = np.stack(lidar2img, axis=0)

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        imgs = [
            mmcv.imfrombytes(
                img_byte,
                flag=self.color_type,
                backend='pillow',
                channel_order='rgb') for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        return results


@TRANSFORMS.register_module()
class LoadMultiViewDepthFromFiles(BaseTransform):
    """Load the gound truth depth map generated from BEVDepth using lidar.
    """
    def __init__(self, is_to_depth_map=True, use_multi_sweeps=False, map_size=None, resize_on_raw=None):
        self.is_to_depth_map = is_to_depth_map
        self.map_size = map_size
        self.resize_on_raw = resize_on_raw
        self.use_multi_sweeps = use_multi_sweeps

    def transform(self, results):
        if self.map_size is None:
            self.map_size = results['img'][0].shape[:2] # List of (H, W, C) numpy array, here should be (900, 1600)
        img_paths = results['filename']
        dpt_paths = []
        map_depths = []
        for img_path in img_paths:
            if self.use_multi_sweeps:
                dpt_path = os.path.join(img_path.split("/samples/")[0], "depth_gt_multi_sweeps", img_path.split("/")[-1]+".bin")
            else:
                dpt_path = os.path.join(img_path.split("/samples/")[0], "depth_gt", img_path.split("/")[-1]+".bin")
            point_depth = np.fromfile(dpt_path, dtype=np.float32, count=-1).reshape(-1, 3)
            dpt_paths.append(dpt_path)
            if self.is_to_depth_map:
                map_depth = self.to_depth_map(point_depth)
                map_depths.append(map_depth)
            
        # img is of shape (h, w, c, num_views)
        # map_depths is a List of depth maps, each of shape (H, W)
        # N * (H, W), N = num_views
        results['dpt'] = map_depths
        results['filename_dpt'] = dpt_paths
        return results
    
    def to_depth_map(self, point_depth):
        """Transform depth based on ida augmentation configuration.

        Args:
            cam_depth (np array): Nx3, 3: x,y,d.
            resize (float): Resize factor.
            resize_dims (list): Final dimension.
            crop (list): x1, y1, x2, y2
            flip (bool): Whether to flip.
            rotate (float): Rotation value.

        Returns:
            np array: [h/down_ratio, w/down_ratio, d]
        """

        # point_depth rows are (u, v, depth) in the original 1600x900 image
        depth_coords = point_depth[:, :2].astype(np.int16)

        depth_map = np.zeros(self.map_size) # (H, W)
        valid_mask = ((depth_coords[:, 1] < self.map_size[0])
                    & (depth_coords[:, 0] < self.map_size[1])
                    & (depth_coords[:, 1] >= 0)
                    & (depth_coords[:, 0] >= 0))
        depth_map[depth_coords[valid_mask, 1],
                depth_coords[valid_mask, 0]] = point_depth[valid_mask, 2]
        if self.resize_on_raw is not None:
            depth_map = mmcv.imresize(
                depth_map, self.resize_on_raw, interpolation='nearest')

        return depth_map


@TRANSFORMS.register_module()
class LoadOccFromFile(BaseTransform):

    def transform(self, results):
        occ_path = os.path.join(results['occ_path'], 'labels.npz')
        occ_labels = np.load(occ_path)

        results['gt_semantic_seg'] = occ_labels['semantics'] # (200, 200, 16) Occ3d
        results['mask_lidar'] = occ_labels['mask_lidar'] # (200, 200, 16) Occ3d
        results['mask_camera'] = occ_labels['mask_camera'] # (200, 200, 16) Occ3d
        return results


@TRANSFORMS.register_module()
class LoadBinaryPseudoOccFromFile(BaseTransform):
    def __init__(self, occ_data_path):
        self.occ_data_path = occ_data_path

    def transform(self, results):
        occ_path = os.path.join(self.occ_data_path, results['scene_idx'], results['token'], 'label_binary.npz')
        occ_labels = np.load(occ_path)

        results['pseudo_binary_gt'] = occ_labels['pseudo_binary_gt'] # (200, 200, 16) binary
        return results


@TRANSFORMS.register_module()
class LoadDINOv2PseudoOccFromFile(BaseTransform):
    def __init__(self, occ_data_path):
        self.occ_data_path = occ_data_path

    def transform(self, results):

        occ_path = os.path.join(self.occ_data_path, results['scene_idx'], results['token'], 'label_dinov2_pca32_tensor.pt')
        results['pseudo_dino_gt'] = torch.load(occ_path)['pca_data'].float() # (200, 200, 16, 32)
        results['pseudo_dino_gt_v'] = torch.load(occ_path)['pca_v'].float() # (768, 32)
        return results


@TRANSFORMS.register_module()
class LoadDINOv2PseudoOccFromFilePCA128(BaseTransform):
    def __init__(self, occ_data_path):
        self.occ_data_path = occ_data_path

    def transform(self, results):
        occ_path = os.path.join(self.occ_data_path, results['scene_idx'], results['token'], 'label_dinov2_pca128_tensor_reso04_098.pt')
        results['pseudo_dino_gt'] = torch.load(occ_path)['pca_data'].float() # (200, 200, 16, 128)
        results['pseudo_dino_gt_v'] = torch.load(occ_path)['pca_v'].float() # (768, 128)
        return results


@TRANSFORMS.register_module()
class ImageAug3D(BaseTransform):

    def __init__(self,
                 final_dim,
                 resize_lim,
                 bot_pct_lim=[0.0, 0.0],
                 rot_lim=[0.0, 0.0],
                 rand_flip=False,
                 is_train=False,
                 transform_dpt=False):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train
        self.transform_dpt = transform_dpt

    def sample_augmentation(self, results):
        H, W = results['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    # For the image augmentation
    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    # For the depth map augmentation
    def dpt_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        if img.dtype != np.uint8:
            img = Image.fromarray(img, mode='F')  # keeps float32 depth values
        else:
            img = Image.fromarray(img, mode='L')  # uint8 depth

        img = img.resize(resize_dims, resample=Image.NEAREST)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate, resample=Image.NEAREST)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def transform(self, data):
        imgs = data['img']
        if 'dpt' in data and self.transform_dpt:
            dpts = data['dpt']
            new_dpts = []
        new_imgs = []
        transforms = []
        if 'dpt' in data and self.transform_dpt:
            for img, dpt in zip(imgs, dpts):
                resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                data)
                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)
                new_img, rotation, translation = self.img_transform(
                    img,
                    post_rot,
                    post_tran,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                new_dpt, _, _ = self.dpt_transform(
                    dpt,
                    post_rot,
                    post_tran,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                transform = torch.eye(4)
                transform[:2, :2] = rotation
                transform[:2, 3] = translation
                new_imgs.append(np.array(new_img).astype(np.float32))
                new_dpts.append(np.array(new_dpt).astype(np.float32))
                transforms.append(transform.numpy())
        else:
            for img in imgs:
                resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                    data)
                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)
                new_img, rotation, translation = self.img_transform(
                    img,
                    post_rot,
                    post_tran,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                transform = torch.eye(4)
                transform[:2, :2] = rotation
                transform[:2, 3] = translation
                new_imgs.append(np.array(new_img).astype(np.float32))
                transforms.append(transform.numpy())
        
        data['img'] = new_imgs
        if 'dpt' in data and self.transform_dpt:
            data['dpt'] = new_dpts
        # update the calibration matrices
        data['img_aug_mat'] = transforms
        data['resize_img_shape'] = new_imgs[0].shape[:2]
        return data


@TRANSFORMS.register_module()
class LoadFeatMaps(BaseTransform):
    def __init__(self, data_root, key, apply_aug=False):
        self.data_root = data_root
        self.key = key
        self.apply_aug = apply_aug

    def transform(self, results):
        feats = []
        img_aug_mats = results.get('img_aug_mat')
        for i, filename in enumerate(results['filename']):
            feat = np.load(
                os.path.join(self.data_root,
                             filename.split('/')[-1].split('.')[0] + '.npy'))
            feat = torch.from_numpy(feat)
 
            if self.apply_aug and img_aug_mats is not None:
                post_rot = img_aug_mats[i][:3, :3]
                post_tran = img_aug_mats[i][:3, 3]
                assert post_rot[0, 1] == post_rot[1, 0] == 0  # noqa

                h, w = feat.shape
                mode = 'nearest' if torch.all(feat == feat.floor()) else 'bilinear'
                feat = F.interpolate(
                    feat[None, None], (int(h * post_rot[1, 1] + 0.5),
                                       int(w * post_rot[0, 0] + 0.5)),
                    mode=mode).squeeze()
                feat = feat[int(post_tran[1]):, int(-post_tran[0]):]
            feats.append(feat)

        results[self.key] = torch.stack(feats) # (N, C, H, W)?
        return results


@TRANSFORMS.register_module()
class LoadPointFromFileLiDAR(object):
    """Load LiDAR Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6, # for our case: 5
        use_dim=[0, 1, 2], # for our case: 5
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmengine.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["lidar_points"]["lidar_path"]
        results['lidar2ego'] = np.stack(results['lidar_points']['lidar2ego'], axis=0)
        points = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim)
        # check reduced beams
        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points
        results["points_current"] = copy.deepcopy(results["points"])

        return results
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadPointsFromMultiSweepsLiDAR(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmengine.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["points"]
        points = points[:, self.use_dim]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                points_sweep = points_sweep[:, self.use_dim]
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)

        results["points"] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


@TRANSFORMS.register_module()
class LoadRadarPointsMultiSweeps(object):
    """Load radar points from multiple sweeps.
    This is usually used for nuScenes dataset to utilize previous sweeps.
    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 load_dim=18,
                 use_dim=[0, 1, 2, 3, 4],
                 sweeps_num=3, 
                 file_client_args=dict(backend='disk'),
                 max_num=300,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 
                 compensate_velocity=False, 
                 normalize_dims=[(3, 0, 50), (4, -100, 100), (5, -100, 100)], 
                 filtering='default', 
                 normalize=False, 
                 test_mode=False):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range
        self.compensate_velocity = compensate_velocity
        self.normalize_dims = normalize_dims
        self.filtering = filtering 
        self.normalize = normalize

        self.encoding = [
            (3, 'one-hot', 8), # dynprop
            (11, 'one-hot', 5), # ambig_state
            (14, 'one-hot', 18), # invalid_state
            (15, 'ordinal', 7), # pdh
            (0, 'nusc-filter', 1) # binary feature: 1 if nusc would have filtered it out
        ]


    def perform_encodings(self, points, encoding):
        for idx, encoding_type, encoding_dims in self.encoding:

            assert encoding_type in ['one-hot', 'ordinal', 'nusc-filter']

            feat = points[:, idx]

            if encoding_type == 'one-hot':
                encoding = np.zeros((points.shape[0], encoding_dims))
                encoding[np.arange(feat.shape[0]), np.rint(feat).astype(int)] = 1
            if encoding_type == 'ordinal':
                encoding = np.zeros((points.shape[0], encoding_dims))
                for i in range(encoding_dims):
                    encoding[:, i] = (np.rint(feat) > i).astype(int)
            if encoding_type == 'nusc-filter':
                encoding = np.zeros((points.shape[0], encoding_dims))
                mask1 = (points[:, 14] == 0)
                mask2 = (points[:, 3] < 7)
                mask3 = (points[:, 11] == 3)

                encoding[mask1 & mask2 & mask3, 0] = 1


            points = np.concatenate([points, encoding], axis=1)
        return points

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.
        Args:
            pts_filename (str): Filename of point clouds data.
        Returns:
            np.ndarray: An array containing point clouds data.
            [N, 18]
        """

        invalid_states, dynprop_states, ambig_states = {
            'default': ([0], range(7), [3]), 
            'none': (range(18), range(8), range(5)), 
        }[self.filtering]

        radar_obj = RadarPointCloud.from_file(
            pts_filename, 
            invalid_states, dynprop_states, ambig_states
        )

        #[18, N]
        points = radar_obj.points

        return points.transpose().astype(np.float32)
        

    def _pad_or_drop(self, points):
        '''
        points: [N, 18]
        '''

        num_points = points.shape[0]

        if num_points == self.max_num:
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)

            return points, masks
        
        if num_points > self.max_num:
            points = np.random.permutation(points)[:self.max_num, :]
            masks = np.ones((self.max_num, 1), 
                        dtype=points.dtype)
            
            return points, masks

        if num_points < self.max_num:
            zeros = np.zeros((self.max_num - num_points, points.shape[1]), 
                        dtype=points.dtype)
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)
            
            points = np.concatenate((points, zeros), axis=0)
            masks = np.concatenate((masks, zeros.copy()[:, [0]]), axis=0)

            return points, masks

    def normalize_feats(self, points, normalize_dims):
        for dim, min, max in normalize_dims:
            points[:, dim] -= min 
            points[:, dim] /= (max-min)
        return points

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.
        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.
        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.
                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        radars_dict = results['radars']

        points_sweep_list = []
        for key, sweeps in radars_dict.items():
            if len(sweeps) < self.sweeps_num:
                idxes = list(range(len(sweeps)))
            else:
                idxes = list(range(self.sweeps_num))

            ts = sweeps[0]['timestamp'] * 1e-6
            for idx in idxes:
                sweep = sweeps[idx]

                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                timestamp = sweep['timestamp'] * 1e-6
                time_diff = ts - timestamp
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

                # velocity compensated by the ego motion in sensor frame
                velo_comp = points_sweep[:, 8:10]
                velo_comp = np.concatenate(
                    (velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
                velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
                velo_comp = velo_comp[:, :2]

                # velocity in sensor frame
                velo = points_sweep[:, 6:8]
                velo = np.concatenate(
                    (velo, np.zeros((velo.shape[0], 1))), 1)
                velo = velo @ sweep['sensor2lidar_rotation'].T
                velo = velo[:, :2]

                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']

                if self.compensate_velocity:
                    points_sweep[:, :2] += velo_comp * time_diff

                points_sweep_ = np.concatenate(
                    [points_sweep[:, :6], velo,
                     velo_comp, points_sweep[:, 10:],
                     time_diff], axis=1)
                            #       0 1 2     3    4   5  6  7    8       9            10             11       12   13      14          15   16      17      18 
                # current format is x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms timestamp
                points_sweep_list.append(points_sweep_)
        
        points = np.concatenate(points_sweep_list, axis=0)
        points = self.perform_encodings(points, self.encoding)
        points = points[:, self.use_dim]


        if self.normalize:
            points = self.normalize_feats(points, self.normalize_dims)
        
        points = RadarPoints(
            points, points_dim=points.shape[-1], attribute_dims=None
        )
        
        results['radar'] = points
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@TRANSFORMS.register_module()
class FuseDepthMaps(BaseTransform):
    def __init__(self, apply_filter=False):
        self.apply_filter = apply_filter

    def transform(self, results):
        assert 'depth' in results # dense monocular depth estimation
        assert 'dpt' in results #sparse lidar depth

        depth = results['depth'] # (N, H, W)
        dpt = results['dpt'] # list of (H, W) numpy arrays
        dpt = torch.stack([torch.from_numpy(d) for d in dpt]) # (N, H, W)
        assert depth.shape == dpt.shape

        if not self.apply_filter:
            fused_depth = torch.where(
                dpt > 0.0, dpt, depth)
        results['depth'] = fused_depth

        return results


@TRANSFORMS.register_module()
class Pack3DDetInputsRadar(BaseTransform):
    INPUTS_KEYS = ['points', 'img', 'radar']
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d'
    ]
    INSTANCEDATA_2D_KEYS = [
        'gt_bboxes',
        'gt_bboxes_labels',
    ]

    SEG_KEYS = [
        'gt_seg_map', 'pts_instance_mask', 'pts_semantic_mask',
        'gt_semantic_seg'
    ]

    def __init__(
        self,
        keys: tuple,
        meta_keys: tuple = ('img_path', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
                            'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                            'pcd_rotation_angle', 'lidar_path',
                            'transformation_3d_flow', 'trans_mat',
                            'affine_aug', 'sweep_img_metas', 'ori_cam2img',
                            'cam2global', 'crop_offset', 'img_crop_offset',
                            'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
                            'num_ref_frames', 'num_views', 'ego2global',
                            'axis_align_matrix')
    ) -> None:
        self.keys = keys
        self.meta_keys = meta_keys

    def _remove_prefix(self, key: str) -> str:
        if key.startswith('gt_'):
            key = key[3:]
        return key

    def transform(self, results: Union[dict,
                                       List[dict]]) -> Union[dict, List[dict]]:
        """Method to pack the input data. when the value in this dict is a
        list, it usually is in Augmentations Testing.

        Args:
            results (dict | list[dict]): Result dict from the data pipeline.

        Returns:
            dict | List[dict]:

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info of
              the sample.
        """
        # augtest
        if isinstance(results, list):
            if len(results) == 1:
                # simple test
                return self.pack_single_results(results[0])
            pack_results = []
            for single_result in results:
                pack_results.append(self.pack_single_results(single_result))
            return pack_results
        # norm training and simple testing
        elif isinstance(results, dict):
            return self.pack_single_results(results)
        else:
            raise NotImplementedError

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        # Format 3D data
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor
        
        if 'radar' in results:
            if isinstance(results['radar'], RadarPoints):
                results['radar'] = results['radar'].tensor

        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = np.stack(results['img'], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(
                        np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                results['img'] = imgs
            else:
                img = results['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # To improve the computational speed by by 3-5 times, apply:
                # `torch.permute()` rather than `np.transpose()`.
                # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
                # for more details
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(
                        np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['img'] = img

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_bboxes_labels', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers_2d', 'depths', 'gt_labels_3d'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])
        if 'gt_bboxes_3d' in results:
            if not isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(
                results['gt_semantic_seg'][None])
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][None, ...]

        data_sample = Det3DDataSample()
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()
        gt_pts_seg = PointData()

        data_metas = {}
        for key in self.meta_keys:
            if key in results:
                data_metas[key] = results[key]
            elif 'images' in results:
                if len(results['images'].keys()) == 1:
                    cam_type = list(results['images'].keys())[0]
                    # single-view image
                    if key in results['images'][cam_type]:
                        data_metas[key] = results['images'][cam_type][key]
                else:
                    # multi-view image
                    img_metas = []
                    cam_types = list(results['images'].keys())
                    for cam_type in cam_types:
                        if key in results['images'][cam_type]:
                            img_metas.append(results['images'][cam_type][key])
                    if len(img_metas) > 0:
                        data_metas[key] = img_metas
            elif 'lidar_points' in results:
                if key in results['lidar_points']:
                    data_metas[key] = results['lidar_points'][key]
        data_sample.set_metainfo(data_metas)

        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == 'gt_bboxes_labels':
                        gt_instances['labels'] = results[key]
                    else:
                        gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                else:
                    raise NotImplementedError(f'Please modified '
                                              f'`Pack3DDetInputs` '
                                              f'to put {key} to '
                                              f'corresponding field')

        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        if 'eval_ann_info' in results:
            data_sample.eval_ann_info = results['eval_ann_info']
        else:
            data_sample.eval_ann_info = None

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        return packed_results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
