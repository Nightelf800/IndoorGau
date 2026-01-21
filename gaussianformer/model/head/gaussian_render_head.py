import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.registry import MODELS
from .base_head import BaseTaskHead
from .localagg.local_aggregate import LocalAggregator as LocalAggregator1
from ssc_pl.models.heads.gsplat_rasterization import rasterize_gaussians
from ssc_pl.models.utils import prepare_gs_attribute, setup_opengl_proj
from ..utils.utils import list_2_tensor, get_rotation_matrix


class MLP(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=None, output_dim=None):
        super(MLP, self).__init__()
        hidden_dim = hidden_dim or max(input_dim * 4, 128)
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class MLP(nn.Module):
#     def __init__(self,
#                  input_dim,
#                  hidden_dim=None,
#                  output_dim=None,
#                  num_layers=2,
#                  activation='relu',
#                  mode=None,
#                  range=None):
#         super().__init__()
#         hidden_dim = hidden_dim or input_dim * 4
#         output_dim = output_dim or input_dim
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(
#             nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
#         self.activation = activation
#         self.range = range
#         self.mode = mode

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = getattr(F, self.activation)(
#                 layer(x)) if i < self.num_layers - 1 else layer(x)

#         if self.mode is not None:
#             if self.mode == 'sigmoid':
#                 x = F.sigmoid(x)
#             if self.range is not None:
#                 x = self.range[0] + (self.range[1] - self.range[0]) * x
#         return x

@MODELS.register_module()
class GaussianRenderHead(BaseTaskHead):
    def __init__(self, 
        init_cfg=None,
        apply_loss_type=None,
        semantic_dim=256,
        num_classes=12,
        pc_range=None,
        empty_args=None,
        with_empty=False,
        cuda_kwargs=None,
        voxelizer=None,
        text_protos=None,
        dataset_type='nusc',
        empty_label=0, 
        reduce_dims=32, 
        patch_size=14, 
        segment_head=False,
    ):
        super().__init__(init_cfg)
        self.num_classes = num_classes
        self.aggregator = LocalAggregator1(**cuda_kwargs)
        # self.aggregator2 = LocalAggregator2(**cuda_kwargs)
        self.H, self.W, self.D = cuda_kwargs['H'], cuda_kwargs['W'], cuda_kwargs['D']
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float))
            self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :])
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, None, :])
        self.with_emtpy = with_empty
        self.empty_args = empty_args
        self.dataset_type = dataset_type
        self.empty_label = empty_label

        if apply_loss_type == 'all':
            self.apply_loss_type = 'all'
        elif 'random' in apply_loss_type:
            self.apply_loss_type = 'random'
            self.random_apply_loss_layers = int(apply_loss_type.split('_')[1])
        else:
            raise NotImplementedError

        # self.semantic_head = MLP(input_dim=semantic_dim, output_dim=num_classes)

        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))
        self.register_buffer('offsets', torch.tensor((0.5, 0.5, 0.5), dtype=torch.float))
        if text_protos is not None:
            self.register_buffer('text_proto_embeds',
                                 torch.load(text_protos, map_location='cpu'))

        self.feat_head = MLP(input_dim=semantic_dim, output_dim=768)
        self.segment_head = MLP(input_dim=semantic_dim, output_dim=12) if segment_head else None
        self.reduce_dims = reduce_dims
        self.patch_size = patch_size

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _sampling(self, gt_xyz, gt_label, gt_mask=None):
        if gt_mask is None:
            gt_xyz = gt_xyz.flatten(1, 3)
        else:
            assert gt_label.shape[0] == 1, "OccLoss does not support bs > 1"
            gt_xyz = gt_xyz[gt_mask].reshape(1, -1, 3)
        return gt_xyz

    def prepare_gaussian_args(self, gaussians):
        means = gaussians.means # b, g, 3
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)
        if self.with_emtpy:
            assert opacities.shape[-1] == self.num_classes - 1
            if 'kitti' in self.dataset_type:
                opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1)
            else:
                opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)
            means = torch.cat([means, self.empty_mean], dim=1)
            scales = torch.cat([scales, self.empty_scale], dim=1)
            rotations = torch.cat([rotations, self.empty_rot], dim=1)
            empty_sem = self.empty_sem.clone()
            empty_sem[..., self.empty_label] += self.empty_scalar
            opacities = torch.cat([opacities, empty_sem], dim=1)
            origi_opa = torch.cat([origi_opa, self.empty_opa], dim=1)

        bs, g, _ = means.shape
        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        R = get_rotation_matrix(rotations) # b, g, 3, 3
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        CovInv = Cov.cpu().inverse().cuda() # b, g, 3, 3
        return means, origi_opa, opacities, scales, CovInv, Cov


    def forward(self, representation, metas=None, **kwargs):
        bs = metas['voxel_origin'].shape[0]
        num_decoder = len(representation)
        if not self.training:
            apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == "all":
            apply_loss_layers = list(range(num_decoder))
        elif self.apply_loss_type == "random":
            if self.random_apply_loss_layers > 1:
                apply_loss_layers = np.random.choice(num_decoder - 1, self.random_apply_loss_layers - 1, False)
                apply_loss_layers = apply_loss_layers.tolist() + [num_decoder - 1]
            else:
                apply_loss_layers = [num_decoder - 1]
        else:
            raise NotImplementedError

        # 初始化pc_real_range为与self.pc_range相同的形状，并扩展到批处理维度
        pc_real_range = torch.zeros((bs, 6), dtype=torch.float32, device=self.pc_range.device)  # [bs, 6]

        if metas['voxel_origin'] is not None:
            # 获取批处理大小
            # 计算每个维度的范围（批处理版本）
            pc_real_range[:, 0] = self.pc_range[0] + metas['voxel_origin'][:, 0] + self.offsets[0] * metas[
                'voxel_size']  # x_min
            pc_real_range[:, 1] = self.pc_range[1] + metas['voxel_origin'][:, 1] + self.offsets[1] * metas[
                'voxel_size']  # y_min
            pc_real_range[:, 2] = self.pc_range[2] + metas['voxel_origin'][:, 2] + self.offsets[2] * metas[
                'voxel_size']  # z_min
            pc_real_range[:, 3] = self.pc_range[3] + metas['voxel_origin'][:, 0] + self.offsets[0] * metas[
                'voxel_size']  # x_max
            pc_real_range[:, 4] = self.pc_range[4] + metas['voxel_origin'][:, 1] + self.offsets[1] * metas[
                'voxel_size']  # y_max
            pc_real_range[:, 5] = self.pc_range[5] + metas['voxel_origin'][:, 2] + self.offsets[2] * metas[
                'voxel_size']  # z_max

        # 获取每个样本的最小坐标范围 [bs, 3]
        pc_min = pc_real_range[:, :3]

        prediction = []
        prediction_base = []
        # dense = []
        occ_xyz = metas['occ_xyz'].to(self.zero_tensor.device)
        occ_cam_mask = metas['occ_cam_mask'].to(self.zero_tensor.device)
        sampled_xyz = self._sampling(occ_xyz, None)

        for idx in apply_loss_layers:
            gaussians = representation[idx]['gaussian']
            # import pdb;
            # pdb.set_trace()
            means, origi_opa, opacities, scales, CovInv, Cov = self.prepare_gaussian_args(gaussians)
            opacities = self.feat_head(opacities)
            bs, g = means.shape[:2]
            # import pdb;
            # pdb.set_trace()
            # 计算语义和文本相似度 (1, 51200, 12)
            if opacities.shape[-1] != self.num_classes:
                opacities = opacities @ self.text_proto_embeds.T.to(opacities.device)

            # 初始化存储列表
            semantics_list = []
            density_list = []

            # 逐样本计算
            for i in range(bs):
                # 处理第 i 个样本
                current_xyz = sampled_xyz[i].clone().float()  # [g, 3] 或其他形状
                current_opa = origi_opa[i].reshape(-1)  # [g] 或其他形状

                # 调用 aggregator（单样本）
                current_semantics = self.aggregator(
                    current_xyz,
                    means[i],  # 处理 means 的 batch 维度
                    current_opa,
                    opacities[i],
                    scales[i],
                    CovInv[i],
                    pc_min[i]
                ).unsqueeze(0).transpose(1, 2)  # [1, c, n]

                # 保存结果
                semantics_list.append(current_semantics.reshape(self.num_classes, self.H, self.W, self.D))

            # 合并结果（堆叠成 batch）
            semantics = torch.stack(semantics_list, dim=0)  # [bs, num_classes, H, W, D]

            # 添加到 prediction 和 dense
            prediction.append(semantics)


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

        render_encoder_feat_ori = metas['encoder_feat_ori']
        # print(f'encoder_feat_ori.shape: {encoder_feat_ori.shape}')
        
        # 对encoder_feat_ori进行L2归一化
        # 特征形状为[batch, channels, height, width]，对channels维度进行归一化
        # encoder_feat_ori = F.normalize(encoder_feat_ori, p=2, dim=1, eps=1e-8)
        
        tgt_feats = render_encoder_feat_ori.flatten(-2).mT.flatten(0, 1)

        # print(f'tgt_feats.shape: {tgt_feats.shape}')
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
            image_size=(480, 640),
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
        # tgt_feats = tgt_feats.mT.reshape(b, c, h // self.patch_size,
        #                                  w // self.patch_size)
        tgt_feats = F.interpolate(
            tgt_feats, scale_factor=self.patch_size, mode='bilinear')
        


        if self.segment_head:
            seg_rendered = self.segment_head(rendered).unsqueeze(0)
        else:
            seg_rendered = None
        # print(f'seg_rendered.shape: {seg_rendered.shape}')

        return {
            'pred_occ': prediction,
            'rendered_feats': rendered,
            'gt_feats': tgt_feats,
            'rendered_depth': depth,
            'rendered_seg': seg_rendered,
        }