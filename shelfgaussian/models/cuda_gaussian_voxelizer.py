import torch.nn as nn
from mmdet3d.registry import MODELS
from CudaGaussianVoxelizer import PyCudaGaussianVoxelizer, PyCudaFeatsCosineLoss

@MODELS.register_module()
class CudaGaussianVoxelizer(nn.Module):

    def __init__(self,
                 voxel_size,
                 voxel_range):
        super().__init__()
        self.voxel_size = voxel_size
        self.voxel_range = voxel_range
        self.voxelizer = PyCudaGaussianVoxelizer(voxel_size=voxel_size, 
                                                voxel_range=voxel_range)
        self.voxelizer_loss = PyCudaFeatsCosineLoss(voxel_size=voxel_size, 
                                                voxel_range=voxel_range)
    
    def forward(self, means3d, opacities, covariances, features=None):

        density, grid_feats, _ = self.voxelizer(means3d.contiguous(), 
                                                opacities.contiguous(), 
                                                covariances.contiguous(),
                                                features.contiguous() if features is not None else None)
        return density, grid_feats
    
    def get_feat_loss(self, voxels_label_full, occ_gt_mask_camera, grid_feats, pseudo_dino_gt):
        loss_cuda, stats = self.voxelizer_loss(
            voxels_label_full,            # [X,Y,Z] float
            occ_gt_mask_camera,             # [X,Y,Z] bool
            grid_feats,                       # [X,Y,Z,C] float32
            pseudo_dino_gt                      # [X,Y,Z,C] float32
        )
        return loss_cuda
