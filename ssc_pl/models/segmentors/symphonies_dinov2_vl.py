import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from transformers import AutoModel, AutoConfig

from ... import build_from_configs
from .. import encoders
from ..decoders import SymphoniesDecoder, SymphoniesDecoderMultiBS
from ..losses import ce_ssc_loss, frustum_proportion_loss, geo_scal_loss, sem_scal_loss
import pickle


class CustomTextEncoder(nn.Module):
    """
    自定义文本编码器，不依赖外部预训练权重
    结构与Qwen2.5兼容，但参数更少
    """
    def __init__(self, vocab_size=151936, embed_dims=4096, max_length=128, num_layers=8, num_heads=16):
        super().__init__()
        self.embed_dims = embed_dims
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dims)
        
        # 位置编码
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, embed_dims))
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=embed_dims * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(embed_dims)
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播
        
        参数：
            input_ids: 输入token IDs，形状为 [batch_size, seq_len]
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]
            
        返回：
            output: 包含last_hidden_state的对象
        """
        # 嵌入层
        embeddings = self.embedding(input_ids) + self.position_embedding[:, :input_ids.size(1), :]
        embeddings = self.layer_norm(embeddings)
        
        # 计算注意力掩码
        src_key_padding_mask = None
        if attention_mask is not None:
            # 使用attention_mask作为src_key_padding_mask（batch_size, seq_len）
            src_key_padding_mask = (attention_mask == 0)  # 转换为True表示需要遮挡的位置
        
        # Transformer编码
        # 注意：PyTorch Transformer的forward方法中，mask参数是src_mask，
        # 而src_key_padding_mask是用于标记填充位置的掩码
        last_hidden_state = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # 返回与预训练模型兼容的输出格式
        class Output:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state
                
        return Output(last_hidden_state)


class VisualLanguageFusion(nn.Module):
    """
    视觉-语言融合模块，使用注意力机制融合两种模态的特征
    """
    def __init__(self, img_embed_dims, text_embed_dims, output_dims):
        super().__init__()
        self.img_embed_dims = img_embed_dims
        self.text_embed_dims = text_embed_dims
        self.output_dims = output_dims
        
        # 文本特征投影到与图像特征相同的维度
        self.text_projection = nn.Linear(text_embed_dims, img_embed_dims)
        
        # 视觉-语言注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=img_embed_dims,
            num_heads=8,
            batch_first=True
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(img_embed_dims, output_dims)
        
        # 残差连接和归一化
        self.norm1 = nn.LayerNorm(img_embed_dims)
        self.norm2 = nn.LayerNorm(output_dims)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, img_features, text_features, text_attention_mask=None):
        """
        前向传播
        
        参数：
            img_features: 图像特征，形状为 [batch_size, num_patches, img_embed_dims]
            text_features: 文本特征，形状为 [batch_size, seq_len, text_embed_dims]
            text_attention_mask: 文本注意力掩码，形状为 [batch_size, seq_len]
            
        返回：
            fused_features: 融合后的特征，形状为 [batch_size, num_patches, output_dims]
        """
        # 投影文本特征
        projected_text = self.text_projection(text_features)
        
        # 计算注意力掩码（如果提供）
        if text_attention_mask is not None:
            # 转换为多头注意力需要的形状 [batch_size, num_heads, num_patches, seq_len]
            # 但 MultiheadAttention 期望的是 [batch_size, seq_len] 形状的 key_padding_mask
            # key_padding_mask 中 1 表示需要被忽略的位置
            key_padding_mask = ~text_attention_mask.bool()
        else:
            key_padding_mask = None
        
        # 应用注意力机制
        # img_features 作为 query，projected_text 作为 key 和 value
        attended_features, _ = self.attention(
            query=img_features,
            key=projected_text,
            value=projected_text,
            key_padding_mask=key_padding_mask
        )
        
        # 残差连接和归一化
        img_features = self.norm1(img_features + self.dropout(attended_features))
        
        # 输出投影
        fused_features = self.output_projection(img_features)
        fused_features = self.norm2(fused_features)
        
        return fused_features

class SymphoniesDinov2VL(nn.Module):

    def __init__(
        self,
        encoder,
        embed_dims,
        scene_size,
        view_scales,
        volume_scale,
        num_classes,
        text_model_name=None,
        num_layers=3,
        image_shape=(370, 1220),
        scale_factor=1,
        pc_range=[0, 0, 0, 0, 0, 0],
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
        self.scale_factor = scale_factor
        self.class_weights = class_weights
        self.criterions = criterions
        
        if text_model_name == "Qwen2.5-0.5B-Instruct":
            # 使用轻量级的Qwen2.5模型替代7B版本
            self.text_encoder = AutoModel.from_pretrained(
                "checkpoints/Qwen2.5-0.5B-Instruct",  # 只有0.5B参数，下载很快
                trust_remote_code=True,
                output_hidden_states=True
            )
            # 冻结文本编码器的参数（可选）
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_embed_dims = self.text_encoder.config.hidden_size
        
        elif text_model_name == "Qwen2.5-7B-Instruct":
            # 使用Qwen2.5-7B-Instruct模型
            self.text_encoder = AutoModel.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct", 
                trust_remote_code=True,
                output_hidden_states=True
            )
            # 冻结文本编码器的参数（可选）
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_embed_dims = self.text_encoder.config.hidden_size

        else:
            print(f"unsupported text_model_name: {text_model_name}, use custom text encoder instead")
            # 使用自定义文本编码器，无需下载任何权重
            self.text_encoder = CustomTextEncoder(
                vocab_size=151936,  # Qwen2.5的词汇表大小
                embed_dims=4096,     # 与Qwen2.5-7B兼容的嵌入维度
                max_length=128,      # 根据需要调整
                num_layers=8,        # 层数，可根据需要调整
                num_heads=16         # 注意力头数，可根据需要调整
            )
            self.text_embed_dims = self.text_encoder.embed_dims
        

        # 初始化图像编码器
        self.img_encoder = build_from_configs(
            encoders, encoder, in_channels=768, embed_dims=embed_dims, scale_factor=scale_factor)

        # 初始化视觉-语言融合模块
        self.vl_fuse = VisualLanguageFusion(
            img_embed_dims=embed_dims,
            text_embed_dims=self.text_embed_dims,
            output_dims=embed_dims
        )
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
        
        # 获取文本特征
        text_outputs = self.text_encoder(
            input_ids=token_ids,
            attention_mask=attention_mask
        )
        # print(f'text_outputs.last_hidden_state.shape: {text_outputs.last_hidden_state.shape}')
        # 使用最后一层隐藏状态作为文本特征
        text_embeds = text_outputs.last_hidden_state  # [batch_size, seq_len, text_embed_dims]
        # 可以考虑使用CLS token特征
        # text_embeds = text_outputs.last_hidden_state[:, 0, :]  # [batch_size, text_embed_dims]
        
        # 图像编码（保持不变）
        pred_insts = self.img_encoder(inputs['img'], inputs['scaleup_img'])
        
        # 提取图像特征
        pred_masks = pred_insts.pop('pred_masks', None)
        feats = pred_insts.pop('feats')
        
        # 融合视觉和语言特征
        # 假设 feats 是一个列表，其中包含不同尺度的图像特征
        fused_feats = []
        for img_feat in feats:
            # 调整特征形状以适应注意力计算
            # img_feat 形状通常为 [batch_size, img_embed_dims, H, W]
            batch_size, embed_dim, H, W = img_feat.shape
            
            # 转换为 [batch_size, num_patches, img_embed_dims]
            img_feat_reshaped = img_feat.view(batch_size, embed_dim, -1).permute(0, 2, 1)
            
            # 应用融合模块
            fused_feat = self.vl_fuse(
                img_features=img_feat_reshaped,
                text_features=text_embeds,
                text_attention_mask=attention_mask
            )
            
            # 转换回原始形状 [batch_size, img_embed_dims, H, W]
            fused_feat_reshaped = fused_feat.permute(0, 2, 1).view(batch_size, embed_dim, H, W)
            # print(f'fused_feat_reshaped.shape: {fused_feat_reshaped.shape}')
            fused_feats.append(fused_feat_reshaped)
        
        # 使用融合后的特征替换原始特征
        feats = fused_feats

        depth, K, E, voxel_origin, projected_pix, fov_mask = list(
            map(lambda k: inputs[k],
                ('depth', 'cam_K', 'cam_pose', 'voxel_origin', f'projected_pix_{self.volume_scale}',
                 f'fov_mask_{self.volume_scale}')))
        

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
            'frustum': frustum_proportion_loss
        }

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
                losses['loss_' + loss] = 0
                # losses['loss_' + loss] = loss_map[loss](preds, target)
        return losses
