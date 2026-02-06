import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from transformers import AutoModel, AutoConfig

from ... import build_from_configs
from .. import encoders
from ..fusion import VisualLanguageFusion, VLFusionAttLayers
from ..decoders import SymphoniesDecoder, SymphoniesDecoderMultiBS
from ..losses import ce_ssc_loss, frustum_proportion_loss, geo_scal_loss, sem_scal_loss, hvm_ce_ssc_loss
import pickle


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



class SymphoniesDinov2VL(nn.Module):

    def __init__(
        self,
        encoder,
        embed_dims,
        scene_size,
        view_scales,
        volume_scale,
        num_classes,
        text_vocab_size=151936,  # Qwen2.5的词汇表大小
        text_embed_dims=1024,     # 文本嵌入维度
        max_text_length=256,      # 最大文本长度
        fusion_type='att',  # 'original' or '3d' or 'att'
        num_layers=3,
        image_shape=(370, 1220),
        scale_factor=1,
        pc_range=[0, 0, 0, 0, 0, 0],
        voxel_size=0.2,
        downsample_z=2,
        use_hvm=False,
        class_weights=None,
        criterions=None,
        depth=None,
        **kwargs,
    ):
        super().__init__()
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.scale_factor = scale_factor
        self.class_weights = class_weights
        self.criterions = criterions
        self.use_hvm = use_hvm
        
        # 定义文本嵌入层
        self.text_embedding = nn.Embedding(text_vocab_size, text_embed_dims)
        
        # 定义Qwen2旋转位置编码
        self.rotary_embedding = Qwen2RotaryEmbedding(
            dim=text_embed_dims,
            max_position_embeddings=max_text_length
        )
        
        # 文本嵌入维度
        self.text_embed_dims = text_embed_dims
        

        # 初始化图像编码器
        self.img_encoder = build_from_configs(
            encoders, encoder, in_channels=768, embed_dims=embed_dims, text_embed_dims=text_embed_dims)

        # 初始化视觉-语言融合模块
        # if fusion_type == '3d':
        #     print("Using VisualLanguageFusion3D module...")
        #     self.vl_fuse = VisualLanguageFusion3D(
        #         img_embed_dims=embed_dims,
        #         text_embed_dims=self.text_embed_dims,
        #         output_dims=embed_dims
        #     )
        # elif fusion_type == 'att':
        #     print("Using VLFusion Attention Layers...")
        #     self.vl_fuse = VLFusionAttLayers(
        #         img_embed_dims=embed_dims,
        #         text_embed_dims=self.text_embed_dims
        #     )

        # else:
        #     print("Using original VisualLanguageFusion module...")
        #     self.vl_fuse = VisualLanguageFusion(
        #         img_embed_dims=embed_dims,
        #         text_embed_dims=self.text_embed_dims,
        #         output_dims=embed_dims
        #     )
        # self.decoder = SymphoniesDecoder(
        #     embed_dims,
        #     num_classes,
        #     num_layers=num_layers,
        #     num_levels=len(view_scales),
        #     scene_shape=scene_size,
        #     project_scale=volume_scale,
        #     image_shape=tuple(x * scale_factor for x in image_shape),
        #     voxel_size=voxel_size,
        #     pc_range = pc_range,
        #     downsample_z=downsample_z)

        self.decoder = SymphoniesDecoderMultiBS(
            embed_dims,
            num_classes,
            num_layers=num_layers,
            num_levels=len(view_scales),
            scene_shape=scene_size,
            project_scale=volume_scale,
            image_shape=image_shape,
            voxel_size=voxel_size,
            pc_range = pc_range,
            downsample_z=downsample_z,
            use_hvm=use_hvm,
        )

    def forward(self, inputs):
        if inputs['img'].dim() == 3:
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

        # depth_eval
        # print('depth_eval: {}'.format(inputs['depth_eval']))
        # print('depth_model: {}'.format(self.depth_model))
        # print('inputs[img].shape: {}'.format(inputs['img'].shape))
        # if inputs['use_depth_eval']:
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
        #         # pred_min = depth.min()
        #         # pred_max = depth.max()
        #         # print(f'pred.max: {pred_max}')
        #         # print(f'pred.min: {pred_min}')
        #         # depth = (depth - pred_min) / (pred_max - pred_min) * 255
        #         # p = depth.squeeze().cpu().numpy()
        #         # p_uint8 = np.clip(p, 0, 255).astype(np.uint8)
        #         # Image.fromarray(p_uint8).save(f"./visual/pred.png")


        # 文本编码
        token_ids = inputs['token_ids']
        attention_mask = inputs['attention_mask']
        
        # 直接将token_ids通过embedding层编码
        # print(f'token_ids.shape: {token_ids.shape}')
        text_embeds = self.text_embedding(token_ids)  # [batch_size, seq_len, text_embed_dims]
        # print(f'text_embeds.shape: {text_embeds.shape}')
        # 应用Qwen2旋转位置编码
        text_embeds = self.rotary_embedding(text_embeds)  # [batch_size, seq_len, text_embed_dims]
        # print(f'text_embeds.shape: {text_embeds.shape}')
        # 注意：这里我们不再需要复杂的transformer编码器，直接使用embedding+旋转编码的结果
        # 图像编码（保持不变）
        pred_insts = self.img_encoder(inputs['img'], scaleup_imgs=inputs['scaleup_img'], text_embeds=text_embeds, attention_mask=attention_mask)
        
        # 提取图像特征
        pred_masks = pred_insts.pop('pred_masks', None)
        feats = pred_insts.pop('feats')

        # for i, feat in enumerate(feats):
        #     print(f'img_feat_{i}.shape: {feat.shape}')
        
        # 融合视觉和语言特征
        # 假设 feats 是一个列表，其中包含不同尺度的图像特征
        # fused_feats = []
        # for i, img_feat in enumerate(feats):
            # 调整特征形状以适应注意力计算
            # img_feat 形状通常为 [batch_size, img_embed_dims, H, W]
            # batch_size, embed_dim, H, W = img_feat.shape
            
            # 转换为 [batch_size, num_patches, img_embed_dims]
            # img_feat_reshaped = img_feat.view(batch_size, embed_dim, -1).permute(0, 2, 1)
            
            # 应用融合模块
            # if i == len(feats) - 1:
            #     fused_feat = self.vl_fuse(
            #         img_features=img_feat,
            #         text_features=text_embeds,
            #         text_attention_mask=attention_mask
            #     )
            #     fused_feats.append(fused_feat)
            # else:
            #     fused_feats.append(img_feat)
            # 转换回原始形状 [batch_size, img_embed_dims, H, W]
            # fused_feat_reshaped = fused_feat.permute(0, 2, 1).view(batch_size, embed_dim, H, W)
            # print(f'fused_feat_reshaped.shape: {fused_feat_reshaped.shape}')
            
        
        # 使用融合后的特征替换原始特征
        # feats = fused_feats

        # for i, fused_feat in enumerate(fused_feats):
        #     print(f'fused_feat_{i}.shape: {fused_feat.shape}')


        depth, K, E, voxel_origin, projected_pix, fov_mask = list(
            map(lambda k: inputs[k],
                ('depth', 'cam_K', 'cam_pose', 'voxel_origin', f'projected_pix_{self.volume_scale}',
                 f'fov_mask_{self.volume_scale}')))
        

        if self.use_hvm:
            outs, hvm_out_dict, hvm_out_dict_pre, hvm_outs_list = self.decoder(
                pred_insts,
                feats,
                pred_masks,
                depth,
                K,
                E,
                voxel_origin,
                projected_pix,
                fov_mask
            )

            return {'ssc_logits': outs[-1], 'aux_outputs': outs, 'hvm_out_dict': hvm_out_dict, 'hvm_out_dict_pre': hvm_out_dict_pre, 'hvm_outs_list': hvm_outs_list}
        else:
            outs = self.decoder(
                pred_insts,
                feats,
                pred_masks,
                depth,
                K,
                E,
                voxel_origin,
                projected_pix,
                fov_mask
            )
        
            return {'ssc_logits': outs[-1], 'aux_outputs': outs}

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

    def loss(self, preds, target):
        loss_map = {
            'ce_ssc': ce_ssc_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
            'frustum': frustum_proportion_loss,
            
        }

        loss_map_extra = {
            'hvm_ce_ssc': hvm_ce_ssc_loss
        }

        # print(f'class_weights: {self.class_weights}')
        target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])
        # print(f'class_weights_update: {self.class_weights}')
        losses = {}
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
                losses['loss_' + loss] = 0
                # losses['loss_' + loss] = loss_map[loss](preds, target)
        if 'hvm_out_dict' in preds:
            for loss in self.criterions:
                if loss == 'hvm_ce_ssc':
                    losses['loss_' + loss] = loss_map_extra[loss](preds, target)
        return losses
