import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from ... import build_from_configs
from .. import encoders
from ..encoders import dinov2_encoder
from .. import decoders
from ..decoders import SymphoniesDecoder
from ..losses import ce_ssc_loss, frustum_proportion_loss, geo_scal_loss, sem_scal_loss, lovasz_softmax_loss, \
    mse_loss, ce_img_loss
# from depth_eval.depth_anything.dpt import DepthAnything
from depth_eval.zoedepth.utils.config import get_config
from depth_eval.zoedepth.models.builder import build_model
from cfg_module import ConfigManager
# from ...engine import LitModule
import lightning as L
from ... import build_from_configs, evaluation, models
from ..utils import (cumprod, flatten_fov_from_voxels, flatten_multi_scale_feats, generate_grid,
                     get_level_start_index, index_fov_back_to_voxels, interpolate_flatten,
                     nchw_to_nlc, nlc_to_nchw, pix2vox, vox2pix)
from torchvision import transforms



# class litModule(L.LightningModule):

#     def __init__(self, *, model, optimizer, scheduler, criterion=None, evaluator=None, **kwargs):
#         super().__init__()
#         self.model = build_from_configs(models, model, **kwargs)
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.criterion = build_from_configs(nn, criterion) if criterion else self.model.loss
#         self.train_evaluator = build_from_configs(evaluation, evaluator)
#         self.test_evaluator = build_from_configs(evaluation, evaluator)
#         if 'class_names' in kwargs:
#             self.class_names = kwargs['class_names']

#     def forward(self, x):
#         return self.model(x)

#     def _step(self, batch, evaluator=None):
#         x, y = batch

#         torch.cuda.empty_cache()
#         pred = self(x)

#         with autocast(enabled=False):
#             loss = self.criterion(pred, y)
#         if evaluator:
#             evaluator.update(pred, y)
#         return loss

#     def training_step(self, batch, batch_idx):


#         loss = self._step(batch, self.train_evaluator)

#         if isinstance(loss, dict):
#             loss['loss_total'] = sum(loss.values())
#             self.log_dict({f'train/{k}': v for k, v in loss.items()})
#         else:
#             self.log('train/loss', loss)
#         return sum(loss.values()) if isinstance(loss, dict) else loss

#     def validation_step(self, batch, batch_idx):
#         self._shared_eval(batch, 'val')

#     def test_step(self, batch, batch_idx):
#         self._shared_eval(batch, 'test')

#     def inference_step(self, batch):
#         self._shared_eval(batch, 'val')

#     def _shared_eval(self, batch, prefix):
#         # print('-----------batch------------')
#         # print(batch)  # 查看 batch 的内容
#         # print(batch[0].keys())
#         loss = self._step(batch, self.test_evaluator)
#         # Lightning automatically accumulates the metric and averages it
#         # if `self.log` is inside the `validation_step` and `test_step`

#         if isinstance(loss, dict):
#             loss['loss_total'] = sum(loss.values())
#             self.log_dict({f'{prefix}/{k}': v for k, v in loss.items()}, sync_dist=True)
#         else:
#             self.log(f'{prefix}/loss', loss, sync_dist=True)

#     def on_train_epoch_end(self):
#         self._log_metrics(self.train_evaluator, 'train')

#     def on_validation_epoch_end(self):
#         self._log_metrics(self.test_evaluator, 'val')

#     def on_test_epoch_end(self):
#         self._log_metrics(self.test_evaluator, 'test')

#     def on_inference_epoch_end(self):
#         self._log_metrics(self.test_evaluator, 'test')

#     def _log_metrics(self, evaluator, prefix=None):
#         metrics = evaluator.compute()
#         iou_per_class = metrics.pop('iou_per_class')
#         if prefix:
#             metrics = {'/'.join((prefix, k)): v for k, v in metrics.items()}
#         self.log_dict(metrics, sync_dist=True)

#         if hasattr(self, 'class_names'):
#             self.log_dict(
#                 {
#                     f'{prefix}/iou_{c}': s.item()
#                     for c, s in zip(self.class_names, iou_per_class)
#                 },
#                 sync_dist=True)
#         evaluator.reset()

#     def configure_optimizers(self):
#         optimizer_cfg = self.optimizer
#         scheduler_cfg = self.scheduler
#         with open_dict(optimizer_cfg):
#             paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
#         if paramwise_cfg:
#             params = []
#             pgs = [[] for _ in paramwise_cfg]

#             for k, v in self.named_parameters():
#                 in_param_group = False
#                 for i, pg_cfg in enumerate(paramwise_cfg):
#                     if 'name' in pg_cfg and pg_cfg.name in k:
#                         pgs[i].append(v)
#                         in_param_group = True
#                     # USER: Customize more cfgs if needed
#                 if not in_param_group:
#                     params.append(v)
#         else:
#             params = self.parameters()
#         optimizer = build_from_configs(optim, optimizer_cfg, params=params)
#         if paramwise_cfg:
#             for pg, pg_cfg in zip(pgs, paramwise_cfg):
#                 cfg = {}
#                 if 'lr_mult' in pg_cfg:
#                     cfg['lr'] = optimizer_cfg.lr * pg_cfg.lr_mult
#                 # USER: Customize more cfgs if needed
#                 optimizer.add_param_group({'params': pg, **cfg})
#         scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
#         if 'interval' in scheduler_cfg:
#             scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg.interval}
#         return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
#     def state_dict(self, *args, **kwargs):
#         # 只保存 other_model 的参数
#         state_dict = super().state_dict(*args, **kwargs)
#         state_dict = {k: v for k, v in state_dict.items() if 'clip_model' not in k}
#         return state_dict

#     def load_state_dict(self, state_dict, strict=False):
#         # 只加载 other_model 的参数
#         # state_dict = {k: v for k, v in state_dict.items() if 'clip_model' not in k}
#         super().load_state_dict(state_dict, strict=False)


class GaussianFormer(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        embed_dims,
        scene_size,
        view_scales,
        volume_scale,
        num_classes,
        num_layers=3,
        image_shape=(370, 1220),
        pc_range=[0, 0, 0, 4, 4, 2],
        voxel_size=0.2,
        downsample_z=2,
        class_weights=None,
        criterions=None,
        depth=None,
        **kwargs,
    ):
        super().__init__()
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.criterions = criterions
        self.gaussian_weight = 0.5
        self.symphonies_weight = 0.5

        self.encoder = build_from_configs(
            encoders, encoder, embed_dims=embed_dims, scales=view_scales)
        # self.encoder = dinov2_encoder(**encoder)
        # GaussianEncoder
        # self.encoder = build_from_configs(
        #     encoders, encoder, embed_dims=embed_dims)

        # self.symphony_model = ConfigManager.get_global_model()
        image_grid = generate_grid(image_shape)
        image_grid = torch.flip(image_grid, dims=[0]).unsqueeze(0)  # 2(wh), h, w
        self.register_buffer('image_grid', image_grid)
        scene_shape = (100, 100, 50)
        voxel_grid = generate_grid(scene_shape, normalize=True)
        self.register_buffer('voxel_grid', voxel_grid)
        self.voxel_size = voxel_size

        self.projection = nn.Linear(768, 512)
        self.gaussian_decoder = build_from_configs(
            decoders, decoder, embed_dims=embed_dims
        )
        self.pc_range = pc_range
        # self.decoder = SymphoniesDecoder(
        #     embed_dims,
        #     num_classes,
        #     num_layers=num_layers,
        #     num_levels=len(view_scales),
        #     scene_shape=scene_size,
        #     project_scale=volume_scale,
        #     image_shape=image_shape,
        #     voxel_size=voxel_size,
        #     downsample_z=downsample_z,
        #     pc_range=pc_range,
        #     use_tsdf=False,
        # )
        # import pdb;
        # pdb.set_trace()
        # for i in range(1000):
        #     print(pc_range)

        # depth_eval
        self.depth_model = depth['depth_model']
        if depth['depth_model'] == 'depthanything':
            # self.depth_eval_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(depth_encoder)).eval()

            overwrite = {**kwargs, "pretrained_resource": depth['depth_pretrained_resource']} if depth['depth_pretrained_resource'] else kwargs
            config = get_config(depth['depth_model_name'], "eval", depth['depth_dataset'], **overwrite)
            self.depth_eval_model = build_model(config)

    def forward(self, inputs):
        if len(inputs['img'].shape) == 3:
            inputs['img'] = inputs['img'].unsqueeze(0)
        h, w = inputs['img'].shape[-2:]
        # mask_transform = transforms.Compose([
        #     transforms.ToTensor(), 
        #     transforms.Resize((224, 224)),
        #     transforms.Normalize(0.5, 0.26)
        # ])
        # alpha = mask_transform(binary_mask * 255)
        # image_features = self.clip_model.visual(image, alpha)
        # import pdb;
        # pdb.set_trace()
        # image_features = model.encode_image(image)
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # x = self.clip_model(inputs['clip_img']).squeeze(0).permute(1, 2, 0)
        # x = self.clip_model.encode_image(inputs['clip_img'])
        # x = x / x.norm(dim=1, keepdim=True)
        # similarity = clip_surgery.clip_feature_surgery(x, self.text_embeddings)
        # similarity_map = clip_surgery.get_similarity_map(similarity[:, 1:, :], (480, 640))
        # gt = similarity_map.argmax(-1) + 1


        # x = x.squeeze(0)[1:, :].reshape(14, 14, 512).permute(2, 0, 1).unsqueeze(0)
        # import pdb;
        # pdb.set_trace()
        # x = F.interpolate(x, size=(480, 640), mode="bilinear", align_corners=False).squeeze(0).flatten(1).mT
        # x = x.squeeze(0).flatten(1).mT
        # x = self.projection(x).flatten(0, 1)
        # u, s, v = torch.pca_lowrank(x.double(), q=128, niter=4)
        # x_reduced = x @ v.to(x)

        # import pdb;
        # pdb.set_trace()
        # print(f'-------inputs paras----------')
        # print(f'inputs.keys: {inputs.keys()}')
        # for key in inputs.keys():
        #     if isinstance(inputs[key], str):
        #         print(f'key: {key}, name: {inputs[key]}')
        #     elif isinstance(inputs[key], int):
        #         print(f'key: {key}, value: {inputs[key]}')
        #     elif isinstance(inputs[key], list):
        #         print(f'key: {key}, value: {inputs[key]}')
        #     else:
        #         print(f'key: {key}, shape: {inputs[key].shape}')

        # depth_eval
        # print('depth_eval: {}'.format(inputs['depth_eval']))
        # print('depth_model: {}'.format(self.depth_model))
        # print('inputs[img].shape: {}'.format(inputs['img'].shape))
        # if inputs['depth_eval'][-1]:
        #     if self.depth_model == 'depthanything':
        #         # depth_eval_image = self.depth_eval_transform({'image': inputs['img']})['image']
        #         focal = torch.Tensor([715.0873]).cuda()  # This magic number (focal) is only used for evaluating BTS model

        #         with torch.no_grad():

        #             depth = self.depth_infer(self.depth_eval_model, inputs['img'], dataset='nyu', focal=focal)

        #             # depth = self.depth_eval_model(inputs['img'])['metric_depth']
        #             # print(f'depth.shape: {depth.shape}')
        #             depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
        #             pred_min = depth.min()
        #             pred_max = depth.max()
        #             depth = (depth - pred_min) / (pred_max - pred_min) * 5.1980
        #             inputs['depth'] = depth

        #         # print(f'depth_model: {self.depth_model}, depth.shape: {depth.shape}')

        #         from PIL import Image
                # pred_min = depth.min()
                # pred_max = depth.max()
                # print(f'pred.max: {pred_max}')
                # print(f'pred.min: {pred_min}')
                # depth = (depth - pred_min) / (pred_max - pred_min) * 255
                # p = depth.squeeze().cpu().numpy()
                # p_uint8 = np.clip(p, 0, 255).astype(np.uint8)
                # Image.fromarray(p_uint8).save(f"./visual/pred.png")



        # GauusianEncoder
        # ms_img_feats = self.encoder(inputs['img'].unsqueeze(1))

        # maskdino
        pred_insts = self.encoder(inputs['img'])
        pred_masks = pred_insts.pop('pred_masks', None)
        feats = pred_insts.pop('feats')

        ms_img_feats = []
        for i in range(len(feats)):
            ms_img_feats.append(feats[i].unsqueeze(1))

        if 'vox_tsdf' in inputs:
            pred_insts['vox_tsdf'] = inputs['vox_tsdf']

        # depth, K, E, voxel_origin, projected_pix, fov_mask = list(
        #     map(lambda k: inputs[k],
        #         ('depth', 'cam_K', 'cam_pose', 'voxel_origin', f'projected_pix_{self.volume_scale}',
        #          f'fov_mask_{self.volume_scale}')))

        # 先注释
        # outs, symphoines_decoder_outs = self.decoder(pred_insts, feats, pred_masks, depth, K, E, voxel_origin, projected_pix,
        #             fov_mask)


        metas = {
            'img': inputs['img'].unsqueeze(1),
            'depth': inputs['depth'],
            'projection_mat': inputs['projection_mat'].to(torch.float32).unsqueeze(1),
            'image_wh': inputs['image_wh'],
            'voxel_size': self.voxel_size,
            'voxel_origin': inputs['voxel_origin'],
            'occ_xyz': inputs['xyz'],
            'occ_cam_mask': inputs[f'fov_mask_{self.volume_scale}'],
            'cam_K': inputs['cam_K'],
            'cam_pose': inputs['cam_pose'],
            'projected_pix_1': inputs['projected_pix_1'],
            'fov_mask_1': inputs['fov_mask_1'],
        }

        gaussian_deocder_outs = self.gaussian_decoder(metas=metas, ms_img_feats=ms_img_feats)

        # print(f'gaussian_deocder_outs.keys: {gaussian_deocder_outs.keys()}')

        return {'ssc_logits': gaussian_deocder_outs['pred_occ'][-1]}
        # return {'ssc_logits': outs[-1], 'aux_outputs': outs}




    def depth_infer(self, model, images, **kwargs):
        """Inference with flip augmentation"""

        # images.shape = N, C, H, W
        def get_depth_from_prediction(pred):
            if isinstance(pred, torch.Tensor):
                pred = pred  # pass
            elif isinstance(pred, (list, tuple)):
                pred = pred[-1]
            elif isinstance(pred, dict):
                pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
            else:
                raise NotImplementedError(f"Unknown output type {type(pred)}")
            return pred

        pred1 = model(images, **kwargs)
        pred1 = get_depth_from_prediction(pred1)

        pred2 = model(torch.flip(images, [3]), **kwargs)
        pred2 = get_depth_from_prediction(pred2)
        pred2 = torch.flip(pred2, [3])

        mean_pred = 0.5 * (pred1 + pred2)

        return mean_pred

    def silog_loss(self, y_pred, y_true, epsilon=1e-6):
        """
        计算 Scale-Invariant Logarithmic Loss (SILOG Loss)
        
        :param y_true: 真实值 (Tensor)
        :param y_pred: 预测值 (Tensor)
        :param epsilon: 避免对数计算中的零值 (float)
        :return: 计算得到的损失值 (float)
        """
        # 计算对数损失
        loss = (torch.log(y_pred + epsilon) - torch.log(y_true + epsilon)) ** 2
        return torch.mean(loss)

    def depth_loss(self, pred, target, criterion='silog_l1'):
        loss = 0
        if 'silog' in criterion:
            loss += self.silog_loss(pred, target)
        if 'l1' in criterion:
            assert(pred.shape == target.shape)
            target = target.to(pred.device)
            target = target.flatten()
            pred = pred.flatten()[target != 0]
            l1_loss = F.l1_loss(pred, target[target != 0])
            if loss != 0:
                l1_loss *= 0.2
            loss += l1_loss
        return loss

    def loss(self, preds, target):
        loss_map = {
            'ce_ssc': ce_ssc_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
            'frustum': frustum_proportion_loss,
            'mse': mse_loss,
            'ce_img': ce_img_loss,
            # 'lovasz': lovasz_softmax_loss
        }

        # print('target.keys: {}'.format(target.keys()))
        # print('target[target].shape: {}'.format(target['target'].shape))
        # print('target[frustums_masks].shape: {}'.format(target['frustums_masks'].shape))
        # print('target[frustums_class_dists].shape: {}'.format(target['frustums_class_dists'].shape))

        # target['target'] = target['target'].flatten(1)
        # print('target[target].flatten.shape: {}'.format(target['target'].shape))

        # import pdb;
        # pdb.set_trace()
        # target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])[:9]
        # target['class_weights'] = None

        # check nan
        # print('pred[ssc_logits]', torch.isnan(preds['ssc_logits']).any())  # 检查 logits 是否有 NaN
        # print('pred[ssc_logits]', torch.isinf(preds['ssc_logits']).any())   # 检查 logits 是否有 Inf
        # print('target['target']', torch.isnan(target['target']).any())    # 检查目标标签是否有 NaN
        # print('target['target'].unique', target['target'].unique())


        # pred = torch.flatten(preds['ssc_logits'], start_dim=-3).squeeze(0)
        # pred = pred.permute(1, 0)
        # num_voxels = pred.size(0)  # 总体素数量
        # tar = torch.flatten(target['target'], start_dim=-3)
        # tar = tar.permute(1, 0)
        # similarity_list = []  # 用于存储每批次的结果

        # 分批次计算
        # import pdb;
        # pdb.set_trace()
        # idx = 0

        # for start in range(0, num_voxels, 10000):
        #     end = min(start + 10000, num_voxels)  # 确定批次的结束索引
        #     pred_batch = pred[start:end]  # 获取当前批次的体素特征 [10000, 512]
        #     tar_batch = tar[start:end]  # (10000, 1)

        #     # 首先，将tar_batch转换为一维Tensor
        #     tar_batch = tar_batch.squeeze(-1).long()

        #     # 然后，创建mask
        #     mask = tar_batch != 255

        #     # 使用mask索引pred_batch和tar_batch
        #     pred_batch = pred_batch[mask]
        #     tar_batch = tar_batch[mask]
        #     result = torch.index_select(self.text_embeddings, 0, tar_batch) #(1000, 512)

        #     # # 扩展维度以计算余弦相似性
        #     # pred_expanded = pred_batch.unsqueeze(1)  # [batch_size, 1, 512]
        #     # text_embeddings_expanded = self.text_embeddings.unsqueeze(0)  # [1, 18, 512]

        #     # 计算余弦相似性
        #     similarity = F.cosine_similarity(pred_batch, result, dim=1)  # [1000, 12]

        #     # 将结果添加到列表中
        #     similarity_list.append(similarity)

        # # import pdb;
        # # pdb.set_trace()
        # similarity = torch.cat(similarity_list, dim=0)
        # loss_similarity = 1.0 - similarity.mean()
        # losses['similarity'] = loss_similarity


        # tgt_feats = preds['tgt_feats']
        # rendered = preds['rendered_feature'].flatten(0, 1)

        # rendered_depth = preds['rendered_depth']
        # depth = preds['tgt_depth'].permute(1, 2, 0)
        # depth = depth.clamp(min=0.0, max=80)
        # rendered_depth = rendered_depth.clamp(min=0.0, max=80)
        # # import pdb;
        # # pdb.set_trace()
        # # print(rendered.requires_grad)
        # losses['loss_cosine'] = F.cosine_embedding_loss(
        #     rendered.flatten(0, 1), tgt_feats.flatten(0, 1),
        #     torch.ones_like(tgt_feats.flatten(0, 1)[0])) * 5

        # import pdb;
        # pdb.set_trace()
        # losses['loss_depth'] = self.depth_loss(
        #     rendered_depth.flatten(0, 1), depth.flatten(0, 1))   
         
        # losses['mae_depth'] = self.depth_loss(
        #     rendered_depth[:, :, :1].flatten(0, 1),
        #     depth[:, :, :1].flatten(0, 1),
        #     criterion='l1')

        # import pdb;
        # pdb.set_trace()
        # if 'aux_outputs' in preds:
        #     for i, pred in enumerate(preds['aux_outputs']):
        #         scale = 1 if i == len(preds['aux_outputs']) - 1 else 0.5
        #         for loss in self.criterions:
        #             losses['loss_' + loss + '_' + str(i)] = loss_map[loss]({
        #                 'ssc_logits': pred
        #             }, target) * scale
        # else:
        # import pdb;
        # pdb.set_trace()
        # target['target'] -= 1
        # target['target'][(target['target'] == -1) | (target['target'] == 254)] = 255
        # tmp_ssc_logits = preds['ssc_logits']
        # preds['ssc_logits'] = preds['ssc_logits_base']


        # temp_target = target['target']
        # target_clone = target['target'].clone()
        # # 在克隆的Tensor上进行修改
        # target_clone[(target_clone == 6)] = 255
        # target_clone[(target_clone == 8)] = 255
        # target_clone[(target_clone == 11)] = 255

        # target_clone[(target_clone == 7)] = 6
        # target_clone[(target_clone == 9)] = 7
        # target_clone[(target_clone == 10)] = 8
        # target['target'] = target_clone
        # # import pdb;
        # # pdb.set_trace()
        # for loss in self.criterions:
        #     losses['loss_' + loss] = loss_map[loss](preds, target)

        # preds['ssc_logits'] = tmp_ssc_logits
        # target['target'] = temp_target


        # print(f'class_weights: {self.class_weights}')
        target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])
        # print(f'class_weights_update: {self.class_weights}')
        losses = {}
        if 'aux_outputs' in preds:
            for i, pred in enumerate(preds['aux_outputs']):
                scale = 1 if i == len(preds['aux_outputs']) - 1 else 0.5
                for loss in self.criterions:
                    losses['loss_' + loss + '_' + str(i)] = loss_map[loss]({
                        'ssc_logits': pred
                    }, target) * scale
        else:
            for loss in self.criterions:
                losses['loss_' + loss] = loss_map[loss](preds, target)
        return losses
