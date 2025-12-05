from importlib import import_module

from third_party.dinov3.dinov3.layers.fp8_linear import scale
import torch
import torch.nn as nn
import sys
from mmengine.config import Config
from mmdet.models.layers import inverse_sigmoid
from mmdet.registry import MODELS
import torch
from ..dinov2 import dinov2_vitb14, dinov2_vitg14, dinov2_vitl14, dinov2_vits14
from .en_att import CLGD, DANet

class Dinov2Encoder(nn.Module):

    def __init__(self,
                 in_channels,
                 model_name,
                 checkpoint_path,
                 patch = 14,
                 embed_dims=256,       # 64/128
                 num_queries=100,      # 100
                 scale_factor=1,
                 freeze=True,
                 use_clstoken = True):
        super().__init__()
        self.intermediate_layer_idx = {
            'dinov2_vits14': [5, 8, 11],
            'dinov2_vitb14': [5, 8, 11], 
            'dinov2_vitl14': [11, 17, 23], 
            'dinov2_vitg14': [19, 29, 39]
            }
        self.patch_size = patch
        self.use_clstoken = use_clstoken
        self.hidden_dims = in_channels  # 256
        self.scale_factor = scale_factor
        
        # 初始化压缩线性层，但维度将在forward中动态设置
        # 我们使用ModuleDict来存储每个可能需要的线性层
        self.compress_layers = nn.ModuleList([
            nn.Linear(in_channels * 2, in_channels),
            nn.Linear(in_channels * 2, in_channels)
        ])
        # 初始化标准化层，将在forward中动态设置
        self.norm_layers = nn.ModuleDict()

        self.out_index = self.intermediate_layer_idx[model_name]
        if model_name == "dinov2_vits14":
            self.model = dinov2_vits14(pretrained=False)
            self.scaleup_model = dinov2_vits14(pretrained=False)
        elif model_name == "dinov2_vitb14":
            self.model = dinov2_vitb14(pretrained=False)
            self.scaleup_model = dinov2_vitb14(pretrained=False)
        elif model_name == "dinov2_vitl14":
            self.model = dinov2_vitl14(pretrained=False)
            self.scaleup_model = dinov2_vitl14(pretrained=False)
        elif model_name == "dinov2_vitg14":
            self.model = dinov2_vitg14(pretrained=False)
            self.scaleup_model = dinov2_vitg14(pretrained=False)
        else:
            raise ValueError(f"dinov2 model_name {model_name} not supported")

        if checkpoint_path is not None:
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=torch.device('cpu')),
                strict=True)  # otherwise all the processes will put the loaded weight on rank 0 and may lead to CUDA OOM
            self.scaleup_model.load_state_dict(
                    torch.load(checkpoint_path, map_location=torch.device('cpu')),
                    strict=True)  # otherwise all the processes will put the loaded weight on rank 0 and may lead to CUDA OOM

        self.query_embed = nn.Embedding(num_queries, embed_dims)   # instance query 均匀分布进行随机初始化
        self.pts_embed = nn.Embedding(num_queries, 2)              # instance pts        

        if freeze:  ### 默认不fine_tune dinov2
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True        
        for param in self.scaleup_model.parameters():
                param.requires_grad = False  
        self.projects = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=self.hidden_dims,
                out_channels=embed_dims,
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dims,
                out_channels=embed_dims,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Conv2d(
                in_channels=self.hidden_dims,
                out_channels=embed_dims,
                kernel_size=3,
                stride=1,
                padding=1)
        ])
        self.scaleup_projects = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=self.hidden_dims,
                out_channels=embed_dims,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Conv2d(
                in_channels=self.hidden_dims,
                out_channels=embed_dims,
                kernel_size=3,
                stride=1,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * self.hidden_dims, self.hidden_dims),
                        nn.GELU()))

        self.DANet = DANet(embed_dims, embed_dims, norm_layer=nn.BatchNorm2d)
        self.clgd = CLGD(embed_dims , embed_dims, norm_layer=nn.BatchNorm2d)

    def forward(self, imgs, scaleup_imgs=None):
        B, _, w, h = imgs.shape
        # print(f'imgs.shape: {imgs.shape}')
        
        # 调整输入图像尺寸为能被patch_size整除
        new_w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size
        new_h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
        
        if new_w != w or new_h != h:
            # print(f'Resizing image from ({w}, {h}) to ({new_w}, {new_h}) to be divisible by patch_size {self.patch_size}')
            # 使用双线性插值调整图像尺寸
            imgs = nn.functional.interpolate(imgs, size=(new_w, new_h), mode='bilinear', align_corners=False)
            w, h = new_w, new_h
        
        feature = self.model.get_intermediate_layers(imgs, n=self.out_index, return_class_token=self.use_clstoken)
        
        # 获取原始feature的第二维度大小（1610）
        if self.use_clstoken:
            original_feature_dim = feature[0][0].shape[1]  # 对于带有cls_token的情况
        else:
            original_feature_dim = feature[0].shape[1]  # 对于不带cls_token的情况
        
        if scaleup_imgs is not None:
            scale_w, scale_h = scaleup_imgs.shape[2:]
            new_scaleup_w = ((scale_w + self.patch_size - 1) // self.patch_size) * self.patch_size
            new_scaleup_h = ((scale_h + self.patch_size - 1) // self.patch_size) * self.patch_size
            if new_scaleup_w != scale_w or new_scaleup_h != scale_h:
                scaleup_imgs = nn.functional.interpolate(scaleup_imgs, size=(new_scaleup_w, new_scaleup_h), mode='bilinear', align_corners=False)
            scaleup_feature = self.scaleup_model.get_intermediate_layers(scaleup_imgs, n=self.out_index, return_class_token=False)

        # 目标维度为原始feature第二维度的4倍（1610 * 4 = 6440）
        target_dim = original_feature_dim * self.scale_factor * self.scale_factor

        feats = []
        feats_x = []
        for i, x in enumerate(feature):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0].unsqueeze(0)
            
            if scaleup_feature is not None:
                # 处理scaleup_feature，将其从6348维度扩充到6440维度
                scaleup_feat = scaleup_feature[i]
                current_dim = scaleup_feat.shape[1]
                padding_size = target_dim - current_dim  # 计算需要填充的大小
                
                # 在第二维度上进行零填充
                padding = torch.zeros((scaleup_feat.shape[0], padding_size, scaleup_feat.shape[2]), device=scaleup_feat.device)
                expanded_scaleup_feat = torch.cat([scaleup_feat, padding], dim=1)
                # print(f'expanded_scaleup_feat.shape: {expanded_scaleup_feat.shape}')
                
                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], w // self.patch_size, h // self.patch_size))
                x = self.projects[i](x)
                feats_x.append(x)

                # print(f'feature[{i}].x.shape: {x.shape}')
                # print(f'feature[{i}].expanded_scaleup_feat.shape: {expanded_scaleup_feat.shape}')

                if i < len(self.scaleup_projects):
                    expanded_scaleup_feat = expanded_scaleup_feat.permute(0, 2, 1).reshape((expanded_scaleup_feat.shape[0], expanded_scaleup_feat.shape[-1], w // self.patch_size * self.scale_factor, h // self.patch_size * self.scale_factor))
                    scale_proj = self.scaleup_projects[i](expanded_scaleup_feat)
                    scale_x = self.clgd(x, scale_proj)
                else:
                    scale_x = self.DANet(feats_x[0], x)
                # print(f'feature[{i}].scale_x.shape: {scale_x.shape}')

               
                feats.append(scale_x)
            else:
                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], w // self.patch_size, h // self.patch_size))
                x = self.projects[i](x)
                feats.append(x)

        # for i, x in enumerate(feats):
        #     print(f'feats[{i}].shape: {x.shape}')
        # print(f'dinov2 encoder exit')
        
        bs = feats[0].size(0)
        return dict(                                                   
            queries=self.query_embed.weight.repeat(bs, 1, 1),
            feats=feats,
            pred_pts=self.pts_embed.weight.repeat(bs, 1, 1).sigmoid())

if __name__ == '__main__':
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')