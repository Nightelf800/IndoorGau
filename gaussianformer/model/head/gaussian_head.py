import numpy as np
import torch, torch.nn as nn

from mmengine.registry import MODELS
from .base_head import BaseTaskHead
from .localagg.local_aggregate import LocalAggregator as LocalAggregator1

from .gaussian_render_head import MLP
from ..utils.utils import list_2_tensor, get_rotation_matrix
import clip_surgery

@MODELS.register_module()
class GaussianHead(BaseTaskHead):
    def __init__(
        self, 
        init_cfg=None,
        apply_loss_type=None,
        semantic_dim=128,
        num_classes=12,
        pc_range=None,
        empty_args=None,
        with_empty=False,
        cuda_kwargs=None,
        voxelizer=None,
        text_protos=None,
        dataset_type='nusc',
        empty_label=0,
        use_hvm_hard=False,
        **kwargs,
    ):
        super().__init__(init_cfg)
        
        self.semantic_dim = semantic_dim
        self.num_classes = num_classes
        self.aggregator = LocalAggregator1(**cuda_kwargs)
        # if use_hvm_hard:
        #     from .localagg2.local_aggregate import LocalAggregator as LocalAggregator2
        #     self.aggregator2 = LocalAggregator2(**cuda_kwargs)
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
        self.use_hvm_hard = use_hvm_hard

        if apply_loss_type == 'all':
            self.apply_loss_type = 'all'
        elif 'random' in apply_loss_type:
            self.apply_loss_type = 'random'
            self.random_apply_loss_layers = int(apply_loss_type.split('_')[1])
        else:
            raise NotImplementedError

        self.feat_head = MLP(input_dim=semantic_dim, output_dim=768)
        # self.hard_voxel_head = MLP(input_dim=semantic_dim, output_dim=64)

        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))
        self.register_buffer('offsets', torch.tensor((0.5, 0.5, 0.5), dtype=torch.float))
        if text_protos is not None:
            self.register_buffer('text_proto_embeds',
                                 torch.load(text_protos, map_location='cpu'))

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # all_texts = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa', 'table', 'television', 'furniture', 'objects']
        # base_texts = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair',  'sofa',  'television', 'furniture']
        # # clip_surgery的构造
        # with torch.no_grad():
        #     self.clip_model, _ = clip_surgery.load("ViT-B/14", device=device)
        #     self.clip_model.eval()
        #     self.text_embeddings = clip_surgery.encode_text_with_prompt_ensemble(self.clip_model, all_texts, device)
        #     print(self.text_embeddings.shape)

            # self.text_embeddings_base = clip_surgery.encode_text_with_prompt_ensemble(self.clip_model, base_texts, device)
            # self.clip_basetext_features = torch.load('clip_basetext_features.pt')
            # self.clip_text_features = torch.load('clip_text_features.pt')
        # self.text_prompt_path = '/share/lkl/Symphonies/checkpoints/clip_20250106-073041_prompt_embeddings.pt'
        # self.text_embeddings = torch.load(self.text_prompt_path)
        # self.voxelizer = MODELS.build(voxelizer)

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

    def forward(
        self,
        representation,
        metas=None,
        **kwargs
    ):
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
        prediction_hard_voxel = []
        # dense = []
        occ_xyz = metas['occ_xyz'].to(self.zero_tensor.device)
        occ_cam_mask = metas['occ_cam_mask'].to(self.zero_tensor.device)
        # occ_xyz = list_2_tensor(metas, 'occ_xyz', self.zero_tensor)
        # occ_label = list_2_tensor(metas, 'occ_label', self.zero_tensor)
        # occ_cam_mask = list_2_tensor(metas, 'occ_cam_mask', self.zero_tensor)
        # sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, occ_cam_mask)
        sampled_xyz = self._sampling(occ_xyz, None)
        # print(f'sampled_xyz.min: {sampled_xyz.min()}')
        # print(f'sampled_xyz.max: {sampled_xyz.max()}')
        for idx in apply_loss_layers:
            gaussians = representation[idx]['gaussian']
            # import pdb;
            # pdb.set_trace()
            means, origi_opa, opacities_sem_feat, scales, CovInv, Cov = self.prepare_gaussian_args(gaussians)
            opacities = self.feat_head(opacities_sem_feat)
            bs, g = means.shape[:2]
            # import pdb;
            # pdb.set_trace()
            # 计算语义和文本相似度 (1, 51200, 12)

            if opacities.shape[-1] != self.num_classes:
                opacities = opacities @ self.text_proto_embeds.T.to(opacities.device)

            # opacities_base = opacities @ self.text_embeddings_base.T.to(opacities.device)
            # import pdb;
            # pdb.set_trace()
            # density, grid_feats = self.voxelizer(
            #     means3d=means.float(),
            #     opacities=origi_opa,
            #     features=opacities.softmax(-1),
            #     covariances=Cov)
            # probs = grid_feats.softmax(-1)
            # preds = probs.argmax(-1)
            # preds += 1
            # preds = torch.where(density.squeeze(-1) > 4e-2, preds, 0)


            # 初始化存储列表
            semantics_list = []
            if self.use_hvm_hard:
                hvm_hard_list = []
                # opacities_hard_voxel = self.hard_voxel_head(opacities_sem_feat)     

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
                ).transpose(0, 1)  # [c, n]
                
                # 保存结果
                current_semantics_reshape = current_semantics.reshape(self.num_classes, self.H, self.W, self.D)
                # print(f'current_semantics_reshape.shape: {current_semantics_reshape.shape}')
                semantics_list.append(current_semantics_reshape)

                if self.use_hvm_hard:
                    
                    # print(f'opacities_sem_feat[i].shape: {opacities_sem_feat[i].shape}')
                    
                    # u, s, v = torch.pca_lowrank(
                    #     opacities_sem_feat[i].double(), q=64, niter=4)
                    # opacities_sem_feat_bi = opacities_sem_feat[i] @ v.to(opacities_sem_feat[i])
                    # opacities_sem_feat_bi = opacities_sem_feat_bi.float()
                                   

                    # 调用 aggregator2（单样本）
                    # current_hvm_hard = self.aggregator2(
                    #     current_xyz,
                    #     means[i],
                    #     current_opa,
                    #     opacities_sem_feat_bi,
                    #     scales[i],
                    #     CovInv[i],
                    #     pc_min[i]
                    # ).transpose(0, 1)  # [c, n]
                    # print(f'current_hvm_hard.shape: {current_hvm_hard.shape}')
                    
                    # current_hvm_hard_reshape = current_hvm_hard.reshape(64, self.H, self.W, self.D)
                    hvm_hard_list.append(current_semantics_reshape)

            # 合并结果（堆叠成 batch）
            semantics = torch.stack(semantics_list, dim=0)  # [bs, num_classes, H, W, D]
            # print(f'semantics.shape: {semantics.shape}')
            if self.use_hvm_hard:
                hvm_hard = torch.stack(hvm_hard_list, dim=0)  # [bs, n, c]
                # print(f'hvm_hard.shape: {hvm_hard.shape}')
                prediction_hard_voxel.append(hvm_hard)
            
            # 添加到 prediction 和 dense
            prediction.append(semantics)
            # dense.append(density)

            # print(f'semantics.shape: {semantics.shape}')  # 应为 [bs, num_classes, H, W, D]
            # print(f'density.shape: {density.shape}')  # 应为 [bs, 1, H, W, D]

            # prediction.append(grid_feats)
            # dense.append(density)

            # predicted_classes = torch.argmax(prediction[0], dim=1)
            # predicted_classes_flat = predicted_classes.flatten()
            # class_counts = torch.bincount(predicted_classes_flat.type(torch.long), minlength=12)
            # print(f'--- Voxel each category: {class_counts}')


        return {
            # 'pred_dense': dense,
            'semantics': semantics,
            'pred_occ': prediction,
            'hvm_hard': prediction_hard_voxel[-1] if self.use_hvm_hard else None,
            'sampled_xyz': sampled_xyz,
            'occ_mask': occ_cam_mask,
            'gaussian': representation[-1]['gaussian']
        }

