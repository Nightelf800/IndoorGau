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
from ..losses import ce_ssc_loss, frustum_proportion_loss, geo_scal_loss, sem_scal_loss, \
    mae_loss, silog_loss, cosine_loss, hvm_ce_ssc_loss
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
from transformers import AutoModelForCausalLM

class Qwen2RotaryEmbedding(nn.Module):
    """
    Qwen2的旋转位置编码实现
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # 计算旋转矩阵系数
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 计算位置编码矩阵
        self._compute_cos_sin_tables(max_position_embeddings)
    
    def _compute_cos_sin_tables(self, max_pos):
        """计算余弦和正弦表"""
        positions = torch.arange(0, max_pos, dtype=torch.float32, device=self.inv_freq.device)
        # 计算频率
        freqs = torch.outer(positions, self.inv_freq)
        # 计算cos和sin值
        cos = freqs.cos().unsqueeze(1)
        sin = freqs.sin().unsqueeze(1)
        # 重复以便应用到所有维度
        # cos形状为[max_pos, 1, dim//2]，需要在最后一个维度重复
        cos = cos.repeat(1, 1, 2)
        sin = sin.repeat(1, 1, 2)
        # 存储为缓冲区
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)
    
    def forward(self, x, positions=None):
        """
        应用旋转位置编码
        
        参数：
            x: 输入张量，形状为 [batch_size, seq_len, dim]
            positions: 可选，位置索引
            
        返回：
            应用旋转编码后的张量
        """
        batch_size, seq_len, dim = x.shape
        
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        
        # 获取对应的cos和sin值
        # cos_table形状为[max_pos, 1, dim]，选择positions后变为[seq_len, 1, dim]
        # 需要将形状调整为[seq_len, dim]以便与输入张量广播
        cos = self.cos_table[positions].squeeze(1)  # [seq_len, dim]
        sin = self.sin_table[positions].squeeze(1)  # [seq_len, dim]
        
        # 对x进行旋转编码
        x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).view_as(x)
        x = x * cos + x_rotated * sin
        
        return x



class GaussianFormerVL(nn.Module):

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
        scale_factor=1,
        text_vocab_size=151936,  # Qwen2.5的词汇表大小
        text_embed_dims=896,     # 文本嵌入维度
        max_text_length=256,      # 最大文本长度
        image_shape=(370, 1220),
        pc_range=[0, 0, 0, 4, 4, 2],
        voxel_size=0.2,
        downsample_z=2,
        class_weights=None,
        criterions=None,
        depth=None,
        render=False,
        use_hvm=False,
        use_danet=False,
        **kwargs,
    ):
        super().__init__()
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.criterions = criterions
        self.gaussian_weight = 0.5
        self.symphonies_weight = 0.5

        self.img_encoder = build_from_configs(
            encoders, encoder, in_channels=768, embed_dims=embed_dims, text_embed_dims=text_embed_dims)

        self.gaussian_decoder = build_from_configs(
            decoders, decoder, embed_dims=embed_dims, use_hvm=use_hvm
        )
        
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.render = render
        self.use_hvm = use_hvm
        self.use_danet = use_danet
        
        # self.text_embedding = nn.Embedding(text_vocab_size, text_embed_dims) 
        # # 定义Qwen2旋转位置编码
        # self.rotary_embedding = Qwen2RotaryEmbedding(
        #     dim=text_embed_dims,
        #     max_position_embeddings=max_text_length
        # )
        # self.text_embed_dims = text_embed_dims
        
        # Qwen2.5 文本编码
        self.text_embed_model = AutoModelForCausalLM.from_pretrained(
            "./checkpoints/Qwen2.5-0.5B-Instruct",
            torch_dtype="auto",
        )
        for param in self.text_embed_model.parameters():
            param.requires_grad = False

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

        # 文本编码
        token_ids = inputs['token_ids']
        attention_mask = inputs['attention_mask']
        
        # 直接将token_ids通过embedding层编码
        # print(f'token_ids.shape: {token_ids.shape}')
        # text_embeds = self.text_embedding(token_ids)  # [batch_size, seq_len, text_embed_dims]
        # print(f'text_embeds.shape: {text_embeds.shape}')
        # 应用Qwen2旋转位置编码
        # text_embeds = self.rotary_embedding(text_embeds)  # [batch_size, seq_len, text_embed_dims]
        # print(f'text_embeds.shape: {text_embeds.shape}')
        # 注意：这里我们不再需要复杂的transformer编码器，直接使用embedding+旋转编码的结果
        # 图像编码（保持不变）
        # print(f'token_ids.shape: {token_ids.shape}')
        # print(f'text_embeds.shape: {text_embeds.shape}')
        
        # Qwen2.5 文本编码
        text_embeds = self.text_embed_model(token_ids, attention_mask).hidden_states[-1].to(dtype=torch.float32)    # 24层 shape=(batch_size, 256, 896)
        # for i in range(len(text_embeds)):
        #     print(f'text_embeds[{i}].shape: {text_embeds[i].shape}')

        pred_insts = self.img_encoder(inputs['img'], use_danet=self.use_danet, text_embeds=text_embeds)

        # maskdino
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
            'encoder_feat_ori': pred_insts['encoder_feat_ori'],
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
            'text_embeds': text_embeds if 'token_ids' in inputs else None,
            'attention_mask': attention_mask if 'attention_mask' in inputs else None,
        }

        gaussian_deocder_outs = self.gaussian_decoder(metas=metas, ms_img_feats=ms_img_feats)

        # print(f'gaussian_deocder_outs.keys: {gaussian_deocder_outs.keys()}')

        if self.render:
            return {
                'rendered_feats': gaussian_deocder_outs['rendered_feats'], 
                'gt_feats': gaussian_deocder_outs['gt_feats'],
                'rendered_depth': gaussian_deocder_outs['rendered_depth'],
                'ssc_logits': gaussian_deocder_outs['pred_occ'][-1]}
        else:
            if self.use_hvm:
                return {'aux_outputs': gaussian_deocder_outs['pred_occ'], 'ssc_logits': gaussian_deocder_outs['pred_occ'][-1], 'hvm_out_dict': gaussian_deocder_outs['hvm_out_dict']}
            else:
                return {'aux_outputs': gaussian_deocder_outs['pred_occ'], 'ssc_logits': gaussian_deocder_outs['pred_occ'][-1]}
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
            'mae': mae_loss,
            'silog': silog_loss,
            'cosine': cosine_loss,
            # 'lovasz': lovasz_softmax_loss
        }
        loss_map_extra = {
            'hvm_ce_ssc': hvm_ce_ssc_loss
        }


        # print(f'class_weights: {self.class_weights}')
        
        # print(f'class_weights_update: {self.class_weights}')
        losses = {}

        # print(f'----------------IMG DEBUG----------------')
        # print(f"preds['rendered_depth'].mean(): {preds['rendered_depth'].mean()}")
        # print(f"preds['rendered_depth'].max(): {preds['rendered_depth'].max()}")
        # print(f"preds['rendered_depth'].min(): {preds['rendered_depth'].min()}")
        # print(f"target['depth'].mean(): {target['depth'].mean()}")
        # print(f"target['depth'].max(): {target['depth'].max()}")
        # print(f"target['depth'].min(): {target['depth'].min()}")
        # print(f"preds['rendered_colors'].mean(): {preds['rendered_feats'].mean()}")
        # print(f"preds['rendered_colors'].max(): {preds['rendered_feats'].max()}")
        # print(f"preds['rendered_colors'].min(): {preds['rendered_feats'].min()}")
        # print(f"target['gt_colors'].mean(): {target['img'].mean()}")
        # print(f"target['gt_colors'].max(): {target['img'].max()}")
        # print(f"target['gt_colors'].min(): {target['img'].min()}")
        # print(f'-----------------------------------------')



        if self.render:
            for loss in self.criterions:
                scale = 1 if loss != 'mae' else 0.2
                if 'rendered_depth' in preds:
                    losses['loss_' + loss + '_depth'] = loss_map[loss](preds['rendered_depth'], target['depth']) * scale
                # if 'rendered_feats' in preds:
                #     losses['loss_' + 'mae' + '_colors'] = loss_map['mae'](preds['rendered_feats'], target['img'])
            if 'rendered_feats' in preds:
                losses['loss_' + 'cosine' + '_feats'] = loss_map['cosine'](preds['rendered_feats'], preds['gt_feats'], weight=5.0)
        else:
            target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])
            if 'aux_outputs' in preds:
                for i, pred in enumerate(preds['aux_outputs']):
                    scale = 1 if i == len(preds['aux_outputs']) - 1 else 0.5
                    for loss in self.criterions:
                        if loss not in loss_map:
                            continue
                        losses['loss_' + loss + '_' + str(i)] = loss_map[loss]({
                            'ssc_logits': pred
                        }, target) * scale
            else:
                for loss in self.criterions:
                    if loss not in loss_map:
                        continue
                    losses['loss_' + loss] = loss_map[loss](preds, target)
            if 'hvm_out_dict' in preds:
                for loss in self.criterions:
                    if loss == 'hvm_ce_ssc':
                        losses['loss_' + loss] = loss_map_extra[loss](preds, target)

        return losses
