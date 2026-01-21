from functools import reduce

import torch
import torch.nn.functional as F
import numpy as np
from gsplat import rasterization

def generate_grid(grid_shape, value=None, offset=0, normalize=False):
    """
    Args:
        grid_shape: The (scaled) shape of grid.
        value: The (unscaled) value the grid represents.
    Returns:
        Grid coordinates of shape [len(grid_shape), *grid_shape]
    """
    if value is None:
        value = grid_shape
    grid = []
    for i, (s, val) in enumerate(zip(grid_shape, value)):
        g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
        if normalize:
            g /= s - 1
        shape_ = [1 for _ in grid_shape]
        shape_[i] = s
        g = g.reshape(1, *shape_).expand(1, *grid_shape)
        grid.append(g)
    return torch.cat(grid, dim=0)


def cumprod(xs):
    return reduce(lambda x, y: x * y, xs)


def flatten_fov_from_voxels(x3d, fov_mask):
    assert x3d.shape[0] == 1
    if fov_mask.dim() == 2:
        assert fov_mask.shape[0] == 1
        fov_mask = fov_mask.squeeze()
    return x3d.flatten(2)[..., fov_mask].transpose(1, 2)


def index_fov_back_to_voxels(x3d, fov, fov_mask):
    # print(f'x3d.shape: {x3d.shape}')
    # print(f'fov.shape: {fov.shape}')
    # print(f'fov_mask.shape: {fov_mask.shape}')
    assert x3d.shape[0] == fov.shape[0] == 1
    if fov_mask.dim() == 2:
        assert fov_mask.shape[0] == 1
        fov_mask = fov_mask.squeeze()
    fov_concat = torch.zeros_like(x3d).flatten(2)
    # print(f'fov_concat.shape: {fov_concat.shape}')
    # print(f'fov_concat[..., fov_mask].shape: {fov_concat[..., fov_mask].shape}')
    fov_concat[..., fov_mask] = fov.transpose(1, 2)
    return torch.where(fov_mask, fov_concat, x3d.flatten(2)).reshape(*x3d.shape)


def interpolate_flatten(x, src_shape, dst_shape, mode='nearest'):
    """Inputs & returns shape as [bs, n, (c)]
    """
    if len(x.shape) == 3:
        bs, n, c = x.shape
        x = x.transpose(1, 2)
    elif len(x.shape) == 2:
        bs, n, c = *x.shape, 1
    assert cumprod(src_shape) == n
    x = F.interpolate(
        x.reshape(bs, c, *src_shape).float(), dst_shape, mode=mode,
        align_corners=False).flatten(2).transpose(1, 2).to(x.dtype)
    if c == 1:
        x = x.squeeze(2)
    return x


def flatten_multi_scale_feats(feats):
    feat_flatten = torch.cat([nchw_to_nlc(feat) for feat in feats], dim=1)
    shapes = torch.stack([torch.tensor(feat.shape[2:]) for feat in feats]).to(feat_flatten.device)
    return feat_flatten, shapes


def get_level_start_index(shapes):
    return torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))


def nlc_to_nchw(x, shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        shape (Sequence[int]): The height and width of output feature map.
    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    B, L, C = x.shape
    assert L == cumprod(shape), 'The seq_len does not match H, W'
    return x.transpose(1, 2).reshape(B, C, *shape).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.
    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
        tuple: The [H, W] shape.
    """
    return x.flatten(2).transpose(1, 2).contiguous()


def pix2cam(p_pix, depth, K):

    p_pix = torch.cat([p_pix * depth, depth], dim=1)  # bs, 3, h, w
    return K.inverse() @ p_pix.flatten(2)


def cam2vox(p_cam, E, vox_origin, vox_size, offset=0.5, pc_range=[0,0,0,0,0,0]):
    p_wld = E.inverse() @ F.pad(p_cam, (0, 0, 0, 1), value=1)
    pc_left_down = torch.tensor([pc_range[:3]], device=vox_origin.device) # 感知区域的左下角坐标
    p_vox = (p_wld[:, :-1].transpose(1, 2) - vox_origin.unsqueeze(1) - pc_left_down.unsqueeze(1)) / vox_size - offset
    return p_vox


def pix2vox(p_pix, depth, K, E, vox_origin, vox_size, offset=0.5, downsample_z=1, pc_range=[0,0,0,0,0,0]):
    p_cam = pix2cam(p_pix, depth, K)
    p_vox = cam2vox(p_cam, E, vox_origin, vox_size, offset, pc_range)
    if downsample_z != 1:
        p_vox[..., -1] /= downsample_z
    return p_vox


def cam2pix(p_cam, K, image_shape):
    """
    Return:
        p_pix: (bs, H*W, 2)
    """
    p_pix = K @ p_cam / p_cam[:, 2]  # .clamp(min=1e-3)
    p_pix = p_pix[:, :2].transpose(1, 2) / (torch.tensor(image_shape[::-1]).to(p_pix) - 1)
    return p_pix


# def vox2pix(p_vox, K, E, vox_origin, vox_size, image_shape, scene_shape, pc_range=[0,0,0,0,0,0]):
#     p_vox = p_vox.squeeze(2) * torch.tensor(scene_shape).to(p_vox) * vox_size + vox_origin
#     import pdb;
#     pdb.set_trace()
#     p_cam = E @ F.pad(p_vox.transpose(1, 2), (0, 0, 0, 1), value=1)
#     return cam2pix(p_cam[:, :-1], K, image_shape).clamp(0, 1)
def vox2pix(p_vox, K, E, vox_origin, vox_size, image_shape, scene_shape, pc_range=[0,0,0,0,0,0]):
    p_vox = p_vox * torch.tensor(scene_shape).to(p_vox) * vox_size + vox_origin.unsqueeze(1) + torch.tensor(pc_range[:3]).to(p_vox)
    p_cam = E @ F.pad(p_vox.transpose(1, 2), (0, 0, 0, 1), value=1)
    return cam2pix(p_cam[:, :-1], K, image_shape).clamp(0, 1)

def volume_rendering(
        volume,
        image_grid,
        K,
        E,
        vox_origin,
        vox_size,
        image_shape,
        depth_args=(2, 50, 1),
):
    depth = torch.arange(*depth_args).to(image_grid)  # (D,)
    p_pix = F.pad(image_grid, (0, 0, 0, 0, 0, 1), value=1)  # (B, 3, H, W)
    p_pix = p_pix.unsqueeze(-1) * depth.reshape(1, 1, 1, 1, -1)

    p_cam = K.inverse() @ p_pix.flatten(2)
    p_vox = cam2vox(p_cam, E, vox_origin, vox_size)
    p_vox = p_vox.reshape(1, *image_shape, depth.size(0), -1)  # (B, H, W, D, 3)
    p_vox = p_vox / (torch.tensor(volume.shape[-3:]) - 1).to(p_vox)

    return F.grid_sample(volume, torch.flip(p_vox, dims=[-1]) * 2 - 1, padding_mode='zeros'), depth


def render_depth(volume, image_grid, K, E, vox_origin, vox_size, image_shape, depth_args):
    sigmas, z = volume_rendering(volume, image_grid, K, E, vox_origin, vox_size, image_shape,
                                 depth_args)
    beta = z[1] - z[0]
    T = torch.exp(-torch.cumsum(F.pad(sigmas[..., :-1], (1, 0)) * beta, dim=-1))
    alpha = 1 - torch.exp(-sigmas * beta)
    depth_map = torch.sum(T * alpha * z, dim=-1).reshape(1, *image_shape)
    depth_map = depth_map  # + d[..., 0]
    return depth_map


def inverse_warp(img, image_grid, depth, pose, K, padding_mode='zeros'):
    """
    img: (B, 3, H, W)
    image_grid: (B, 2, H, W)
    depth: (B, H, W)
    pose: (B, 3, 4)
    """
    p_cam = pix2cam(image_grid, depth.unsqueeze(1), K)
    p_cam = (pose @ F.pad(p_cam, (0, 0, 0, 1), value=1))[:, :3]
    p_pix = cam2pix(p_cam, K, img.shape[2:])
    p_pix = p_pix.reshape(*depth.shape, 2) * 2 - 1
    projected_img = F.grid_sample(img, p_pix, padding_mode=padding_mode)
    valid_mask = p_pix.abs().max(dim=-1)[0] <= 1
    return projected_img, valid_mask

def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -9.21, 9.21)
    return torch.sigmoid(tensor)



def prepare_gs_attribute(gs_attribute):
    # prepare gaussian
    pc = {}

    # pdb.set_trace()
    pc['get_xyz_grid'] = gs_attribute.means
    pc['get_xyz'] = pc['get_xyz_grid']
    pc['get_opacity'] = gs_attribute.opacities


    # if self.opt.weight_entropy_last > 0:
    #     loss_entropy = -pc['get_opacity'] * torch.log(pc['get_opacity'] + 1e-8)
    #     loss_entropy = self.opt.weight_entropy_last * loss_entropy.mean()
    #     self.outputs[("loss_entropy_last", 0)] = loss_entropy

    pc['flow']  = 0
    pc['get_scaling'] =  gs_attribute.scales
    pc['get_rotation'] = gs_attribute.rotations
    pc['active_sh_degree'] = 0
    pc['confidence'] = torch.ones_like(pc['get_opacity'])

    pc['semantic'] = gs_attribute.semantics

    # pc['get_features'] = torch.ones_like(pc['get_opacity']).repeat(1, 3)
    
    
    #------------------
    # K.shape: [6, 4, 4]
    # pc['get_xyz_grid'].shape: [2160000, 3]
    # pc['get_opacity'].shape: [2160000, 3]
    # pc['flow'] = 0
    # pc['get_scaling'].shape: [2160000, 3]
    # pc['get_rotation'].shape: [2160000, 4]
    # pc['active_sh_degree'] = 0
    # pc['confidence'].shape: [2160000, 1]
    # pc['semantic'].shape: [2160000, 512]
    # pc['get_features'].shape: [2160000, 3]
    
    return pc


def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32), scale=1.0):
    translate = translate.to(R.device)
    
    Rt = torch.zeros((4, 4), dtype=torch.float32, device=R.device)
    # import pdb;
    # pdb.set_trace()
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    
    return Rt


def setup_opengl_proj(w, h, k, w2c, near=0.01, far=100):

    viewpoint_camera = {}
    # import pdb;
    # pdb.set_trace()
    R = w2c[:3, :3].transpose(0, 1)  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
    camera_center = world_view_transform.inverse()[3, :3]

    # projection_matrix
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    
    FoVx = 2 * torch.atan2(torch.tensor(w, dtype=torch.float32), 2 * fx)
    FoVy = 2 * torch.atan2(torch.tensor(h, dtype=torch.float32), 2 * fy)

    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(opengl_proj)).squeeze(0)

    
    viewpoint_camera['FoVx'] = FoVx
    viewpoint_camera['FoVy'] = FoVy
    viewpoint_camera['world_view_transform'] = world_view_transform
    viewpoint_camera['full_proj_transform'] = full_proj_transform
    viewpoint_camera['cam_K'] = k
    viewpoint_camera['camera_center'] = camera_center
    viewpoint_camera['image_height'] = h
    viewpoint_camera['image_width'] = w

    return viewpoint_camera

def cam2world(points, cam2img, cam2ego, img_aug_mat=None):
    if img_aug_mat is not None:
        post_rots = img_aug_mat[..., :3, :3]
        post_trans = img_aug_mat[..., :3, 3]
        points = points - post_trans.unsqueeze(-2)
        points = (torch.inverse(post_rots).unsqueeze(2)
                  @ points.unsqueeze(-1)).squeeze(-1)

    cam2img = cam2img[..., :3, :3]
    # 移除autocast上下文管理器，直接执行计算
    combine = cam2ego[..., :3, :3] @ torch.inverse(cam2img)
    points = points.float()
    points = torch.cat(
        [points[..., :2] * points[..., 2:3], points[..., 2:3]], dim=-1)
    # 修复矩阵乘法的维度不匹配问题：combine.unsqueeze(1) 而不是 unsqueeze(2)
    points = combine.unsqueeze(1) @ points.unsqueeze(-1)
    points = points.squeeze(-1) + cam2ego[..., None, :3, 3]
    return points

def apply_to_items(func, iterable):
    if isinstance(iterable, list):
        return [func(i) for i in iterable]
    elif isinstance(iterable, dict):
        return {k: func(v) for k, v in iterable.items()}

def get_covariance(s, r):
    # 确保s有3个通道
    if s.shape[-1] == 1:
        s = s.expand(*s.shape[:-1], 3)
    elif s.dim() == 1:
        s = s.unsqueeze(-1).expand(-1, 3)
    
    L = torch.zeros(*s.shape[:-1], 3, 3, dtype=s.dtype, device=s.device)
    for i in range(3):
        L[..., i, i] = s[..., i]

    L = r @ L
    covariance = L @ L.mT
    return covariance

def quat_to_rotmat(quats):
    q = quats / torch.sqrt((quats**2).sum(dim=-1, keepdim=True))
    r, x, y, z = [i.squeeze(-1) for i in q.split(1, dim=-1)]

    R = torch.zeros((*r.shape, 3, 3)).to(r)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - r * z)
    R[..., 0, 2] = 2 * (x * z + r * y)
    R[..., 1, 0] = 2 * (x * y + r * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - r * x)
    R[..., 2, 0] = 2 * (x * z - r * y)
    R[..., 2, 1] = 2 * (y * z + r * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def rotmat_to_quat(rot_matrices):
    inputs = rot_matrices
    rot_matrices = rot_matrices.cpu().numpy()
    quats = []
    for rot in rot_matrices:
        while not np.allclose(rot @ rot.T, np.eye(3)):
            U, _, V = np.linalg.svd(rot)
            rot = U @ V
        quats.append(Quaternion(matrix=rot).elements)
    return torch.from_numpy(np.stack(quats)).to(inputs)

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
        for i, out in outs:
            outs[i] = out.reshape(bsn + out.shape[1:])
    else:
        outs = outs.reshape(bsn + outs.shape[1:])
    return outs

OCC3D_CATEGORIES = (
    ['empty'],
    ['ceiling'],
    ['floor'],
    ['wall'],
    ['window'],
    ['chair'],
    ['bed'],
    ['sofa'],
    ['table'],
    ['tvs'],
    ['furn'],
    ['objs'],
)