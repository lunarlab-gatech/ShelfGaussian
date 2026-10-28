
import numpy as np
import torch
from pyquaternion import Quaternion
from torch.cuda.amp import autocast
import math


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse function of sigmoid, has the same
        shape with input.
        Approximately: [-11.5, 11.5]
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
pc_range_2 = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
pc_range_3 = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]


def gen_sineembed_for_position(pos_tensor):
    '''
    Transformer models are permutation-invariant, so they need positional encodings to understand spatial structure.
    '''
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    elif pos_tensor.size(-1) == 3:
        z_embed = pos_tensor[:, :, 2] * scale
        pos_z = z_embed[:, :, None] / dim_t
        pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_z), dim=2)
    elif pos_tensor.size(-1) == 6:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        z_embed = pos_tensor[:, :, 4] * scale
        pos_z = z_embed[:, :, None] / dim_t
        pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)

        l_embed = pos_tensor[:, :, 5] * scale
        pos_l = l_embed[:, :, None] / dim_t
        pos_l = torch.stack((pos_l[:, :, 0::2].sin(), pos_l[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h, pos_z, pos_l), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def cam2world(points, cam2img, cam2ego, img_aug_mat=None):
    if img_aug_mat is not None:
        post_rots = img_aug_mat[..., :3, :3]
        post_trans = img_aug_mat[..., :3, 3]
        points = points - post_trans.unsqueeze(-2)
        points = (torch.inverse(post_rots).unsqueeze(2)
                  @ points.unsqueeze(-1)).squeeze(-1)

    cam2img = cam2img[..., :3, :3]
    with autocast(enabled=False):
        combine = cam2ego[..., :3, :3] @ torch.inverse(cam2img)
        points = points.float()
        points = torch.cat(
            [points[..., :2] * points[..., 2:3], points[..., 2:3]], dim=-1)
        points = combine.unsqueeze(2) @ points.unsqueeze(-1)
    points = points.squeeze(-1) + cam2ego[..., None, :3, 3]
    return points


def world2cam(points, cam2img, cam2ego, img_aug_mat=None, eps=1e-6):
    points = points - cam2ego[..., None, :3, 3]
    points = torch.inverse(cam2ego[..., None, :3, :3]) @ points.unsqueeze(-1)
    points = (cam2img[..., None, :3, :3] @ points).squeeze(-1)
    points = points / points[..., 2:3].clamp(eps)  # NOTE
    if img_aug_mat is not None:
        points = img_aug_mat[..., None, :3, :3] @ points.unsqueeze(-1)
        points = points.squeeze(-1) + img_aug_mat[..., None, :3, 3]
    return points[..., :2]


def rotmat_to_quat(rot_matrices):
    inputs = rot_matrices
    rot_matrices = rot_matrices.detach().cpu().numpy()
    quats = []
    for rot in rot_matrices:
        while not np.allclose(rot @ rot.T, np.eye(3)):
            U, _, V = np.linalg.svd(rot)
            rot = U @ V
        quats.append(Quaternion(matrix=rot).elements)
    return torch.from_numpy(np.stack(quats)).to(inputs)


def get_covariance(s, r):
    L = torch.zeros((*s.shape[:2], 3, 3)).to(s)
    for i in range(s.size(-1)):
        L[..., i, i] = s[..., i]

    L = r @ L
    covariance = L @ L.mT # R x S x S^T x R^T
    return covariance


def unbatched_forward(func):

    def wrapper(*args, **kwargs):
        bs = None
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, torch.Tensor):
                if bs is None:
                    bs = arg.size(0)
                else:
                    assert bs == arg.size(0)

        outputs = []
        for i in range(bs):
            output = func(
                *[
                    arg[i] if isinstance(arg, torch.Tensor) else arg
                    for arg in args
                ], **{
                    k: v[i] if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                })
            outputs.append(output)

        if isinstance(outputs[0], tuple):
            return tuple([
                torch.stack([out[i] for out in outputs])
                for i in range(len(outputs[0]))
            ])
        else:
            return torch.stack(outputs)

    return wrapper


def flatten_bsn_forward(func, *args, **kwargs):
    args = list(args)
    bsn = None
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            if bsn is None:
                bsn = arg.shape[:2]
            args[i] = arg.flatten(0, 1)
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            if bsn is None:
                bsn = v.shape[:2]
            kwargs[k] = v.flatten(0, 1)
    outs = func(*args, **kwargs)
    if isinstance(outs, tuple):
        outs = list(outs)
        for i, out in enumerate(outs):
            outs[i] = out.reshape(bsn + out.shape[1:])
    else:
        outs = outs.reshape(bsn + outs.shape[1:])
    return outs


OCC3D_CATEGORIES = (
    ['barrier'], # 0
    ['bicycle'],
    ['bus'],
    ['car'],
    ['construction vehicle'],
    ['motorcycle'],
    ['person'],
    ['cone'],
    ['trailer'],
    ['truck'],
    ['road'],
    ['sidewalk'],
    ['terrain', 'grass'],
    ['building', 'wall', 'fence', 'pole', 'sign'],
    ['vegetation'],
    ['sky'], # 15, relates to nothing labeled in Occ3D
    )  # `sum(OCC3D_CATEGORIES, [])` if you need to flatten the list.


    
    
