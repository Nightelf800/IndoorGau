from turtle import width
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.registry import MODELS
from .base_head import BaseTaskHead
from .gaussian_render_head import MLP
from .localagg.local_aggregate import LocalAggregator as LocalAggregator1
from ssc_pl.models.heads.gsplat_rasterization import rasterize_gaussians
from ssc_pl.models.utils import prepare_gs_attribute, setup_opengl_proj
from ..utils.utils import list_2_tensor, get_rotation_matrix



@MODELS.register_module()
class GaussianInternetRenderHead(BaseTaskHead):
    def __init__(self, 
        init_cfg=None,
        semantic_dim=256,
        reduce_dims=32, 
        patch_size=14, 
        segment_head=False,
    ):
        super().__init__(init_cfg)

        self.feat_head = MLP(input_dim=semantic_dim, output_dim=768)
        self.segment_head = MLP(input_dim=semantic_dim, output_dim=12) if segment_head else None
        self.reduce_dims = reduce_dims
        self.patch_size = patch_size

    def forward(self, representation, metas=None, **kwargs):
        render_pc = prepare_gs_attribute(representation[-1]['gaussian'])
        
        # viewpoint_camera = setup_opengl_proj(w = 640, h = 480, k = metas['cam_K'][0], w2c = metas['cam_pose'][0], near=0.01, far=100)
        render_means3d = render_pc['get_xyz'].float()
        render_features = self.feat_head(render_pc['semantic']).float()
        # features = features @ pca_matrix.to(features)
        render_opacities = render_pc['get_opacity'].squeeze(-1).float()
        render_scales = render_pc['get_scaling'].float()
        render_rotations = render_pc['get_rotation'].float()
        cam2img = metas['cam_K'][:, None, :, :].float()
        # viewmat = viewpoint_camera['world_view_transform'].unsqueeze(0).float()
        cam_pose = metas['cam_pose'][:, None, :, :].float()        
        img_h, img_w = metas['img'].shape[-2:]

        render_encoder_feat_ori = metas['encoder_feat_ori']
        
        # print(f'encoder_feat_ori.shape: {encoder_feat_ori.shape}')
        
        # 对encoder_feat_ori进行L2归一化
        # 特征形状为[batch, channels, height, width]，对channels维度进行归一化
        # encoder_feat_ori = F.normalize(encoder_feat_ori, p=2, dim=1, eps=1e-8)
        
        tgt_feats = render_encoder_feat_ori.flatten(-2).mT.flatten(0, 1)

        
        u, s, v = torch.pca_lowrank(
            tgt_feats.double(), q=self.reduce_dims, niter=4)
        tgt_feats = tgt_feats @ v.to(tgt_feats)
        render_features = render_features @ v.to(render_features)
        render_features = render_features.float()

        # print(f'means3d.mean: {means3d.mean()}')
        # print(f'means3d.max: {means3d.max()}')
        # print(f'means3d.min: {means3d.min()}')
        # print(f'features.mean: {features.mean()}')
        # print(f'features.max: {features.max()}')
        # print(f'features.min: {features.min()}')
        # print(f'opacities.mean: {opacities.mean()}')
        # print(f'opacities.max: {opacities.max()}')
        # print(f'opacities.min: {opacities.min()}')
        # print(f'scales.mean: {scales.mean()}')
        # print(f'scales.max: {scales.max()}')
        # print(f'scales.min: {scales.min()}')
        # print(f'rotations.mean: {rotations.mean()}')
        # print(f'rotations.max: {rotations.max()}')
        # print(f'rotations.min: {rotations.min()}')
        
        rendered = rasterize_gaussians(
            render_means3d,
            render_features,
            render_opacities,
            render_scales,
            render_rotations,
            cam2img,
            cam_pose,
            img_aug_mats=None,
            image_size=(img_h, img_w),
            near_plane=0.1,
            far_plane=100,
        render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
        channel_chunk=32).flatten(0, 1)
        
        depth = rendered[:, -1]
        rendered = rendered[:, :-1]
        # depth = depth.clamp(min=0.0, max=80)

        b, c, h, w = rendered.shape
        new_w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size
        new_h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
        
        if new_w != w or new_h != h:
            # print(f'Resizing image from ({w}, {h}) to ({new_w}, {new_h}) to be divisible by patch_size {self.patch_size}')
            # 使用双线性插值调整图像尺寸
            rendered = nn.functional.interpolate(rendered, size=(new_h, new_w), mode='bilinear', align_corners=False)
            h, w = new_h, new_w
        tgt_feats = tgt_feats.reshape(b, h // self.patch_size, w // self.patch_size, c).permute(0, 3, 1, 2)
        tgt_feats = F.interpolate(
            tgt_feats, scale_factor=self.patch_size, mode='bilinear')
        


        if self.segment_head:
            seg_rendered = self.segment_head(rendered).unsqueeze(0)
        else:
            seg_rendered = None
        # print(f'seg_rendered.shape: {seg_rendered.shape}')

        return {
            'rendered_feats': rendered,
            'gt_feats': tgt_feats,
            'rendered_depth': depth,
            'rendered_seg': seg_rendered,
        }