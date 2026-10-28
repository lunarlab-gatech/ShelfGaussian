# CudaGaussianVoxelizer

CUDA kernels used by ShelfGaussian to voxelize 3D Gaussians (density + feature splatting)
and to compute the voxel-level feature cosine loss, with forward and backward passes.
Written by [Yandong Luo](https://github.com/Yandong-Luo).

## Install

Requires a CUDA toolkit matching your PyTorch build.

```bash
cd third_party/CudaGaussianVoxelizer
pip install -e . --no-build-isolation
```

## Usage

```python
from CudaGaussianVoxelizer import PyCudaGaussianVoxelizer, PyCudaFeatsCosineLoss

voxelizer = PyCudaGaussianVoxelizer(voxel_size=[0.4, 0.4, 0.4],
                                    voxel_range=[-40, -40, -1, 40, 40, 5.4])
# means3d: (N, 3), opacities: (N, 1), covariances: (N, 3, 3), features: (N, C) or None
density, grid_feats, _ = voxelizer(means3d, opacities, covariances, features)
```

In ShelfGaussian it is wrapped by `gausstr/models/cuda_gaussian_voxelizer.py`.
