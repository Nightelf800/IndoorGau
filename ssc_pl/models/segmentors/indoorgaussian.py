import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from ... import build_from_configs
from .. import encoders
from .. import decoders
from .. import heads
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


class IndoorGaussian(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        head,
        embed_dims,
        scene_size,
        view_scales,
        volume_scale,
        num_classes,
        num_layers=3,
        num_queries=300,
        image_shape=(370, 1220),
        scale_factor=1,
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

        self.encoder = build_from_configs(
            encoders, encoder, in_channels=768, embed_dims=embed_dims, scale_factor=scale_factor)

        self.decoder = build_from_configs(
            decoders, decoder, embed_dims=embed_dims)

        self.head = build_from_configs(
            heads, head, embed_dims=embed_dims)

        # self.gaussian_decoder = build_from_configs(
        #     decoders, decoder, embed_dims=embed_dims
        # )
        self.pc_range = pc_range

        # depth_eval
        self.depth_model = depth['depth_model']
        if depth['depth_model'] == 'depthanything':
            # self.depth_eval_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(depth_encoder)).eval()

            overwrite = {**kwargs, "pretrained_resource": depth['depth_pretrained_resource']} if depth['depth_pretrained_resource'] else kwargs
            config = get_config(depth['depth_model_name'], "eval", depth['depth_dataset'], **overwrite)
            self.depth_eval_model = build_model(config)

        self.query_embeds = nn.Embedding(num_queries, embed_dims)

    def forward(self, inputs):
        if len(inputs['img'].shape) == 3:
            inputs['img'] = inputs['img'].unsqueeze(0)
        h, w = inputs['img'].shape[-2:]


        # encoder
        pred_insts = self.encoder(inputs['img'])


        pred_masks = pred_insts.pop('pred_masks', None)
        feats = pred_insts.pop('feats')


        ms_img_feats = []
        for i in range(len(feats)):
            ms_img_feats.append(feats[i].unsqueeze(1))

        if 'vox_tsdf' in inputs:
            pred_insts['vox_tsdf'] = inputs['vox_tsdf']

        # 首先处理特征，得到正确的spatial_shapes和feat_flatten
        decoder_inputs = self.pre_transformer(feats)
        # 使用pre_transformer中已经计算好的feat_flatten，而不是再次调用flatten_multi_scale_feats
        feat_flatten = decoder_inputs['feat_flatten']
        print(f'feat_flatten.shape: {feat_flatten.shape}')
        print(f'spatial_shapes: {decoder_inputs["spatial_shapes"]}')
        print(f'spatial_shapes sum: {decoder_inputs["spatial_shapes"].prod(1).sum()}')
        
        feats = decoder_inputs.pop('feat_flatten')
        decoder_inputs.update(self.pre_decoder(feats))


        print(f'decoder_inputs.keys: {decoder_inputs.keys()}')
        print(f'feats.shape: {feats.shape}')
        print(f'spatial_shapes in decoder_inputs: {decoder_inputs["spatial_shapes"]}')
        print(f'spatial_shapes sum in decoder_inputs: {decoder_inputs["spatial_shapes"].prod(1).sum()}')
        print(f'value shape in decoder_inputs: {decoder_inputs["value"].shape}')

        query, reference_points = self.decoder(**decoder_inputs)

        print(f'query.shape: {query.shape}')
        print(f'reference_points.shape: {reference_points.shape}')

        pred_occ = self.head(query[-1], 
                            reference_points[-1], 
                            depth=inputs['depth'],
                            cam2img=inputs['cam_K'],
                            cam2ego=inputs['cam_pose'],
                            mode='occ')

        print(f'pred_occ.shape: {pred_occ.shape}')
        exit()

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


    def pre_transformer(self, mlvl_feats):
        batch_size = mlvl_feats[0].size(0)

        mlvl_masks = []
        for feat in mlvl_feats:
            mlvl_masks.append(None)

        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask) in enumerate(zip(mlvl_feats, mlvl_masks)):
            batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (bs, num_level, 2)
                [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        else:
            valid_ratios = mlvl_feats[0].new_ones(batch_size, len(mlvl_feats),
                                                  2)

        decoder_inputs_dict = dict(
            key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            feat_flatten=feat_flatten)  # 将计算好的feat_flatten也返回
        
        return decoder_inputs_dict

    def pre_decoder(self, memory):
        bs, _, c = memory.shape
        query = self.query_embeds.weight.unsqueeze(0).expand(bs, -1, -1)
        reference_points = torch.rand((bs, query.size(1), 2)).to(query)

        decoder_inputs_dict = dict(
            query=query, value=memory, reference_points=reference_points)
        return decoder_inputs_dict




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
