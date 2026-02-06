from importlib import import_module
import torch
import torch.nn as nn
import sys
from mmengine.config import Config
from mmdet.models.layers import inverse_sigmoid
from mmdet.registry import MODELS
import torch
from ..dinov2 import dinov2_vitb14, dinov2_vitg14, dinov2_vitl14, dinov2_vits14
from .en_att import CLGD, DANet
from ..fusion import VLFusionAttLayers, ImgTextSelfCrossFusion

class Dinov2VLFEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 model_name,
                 checkpoint_path,
                 patch = 14,
                 embed_dims=256,       # 64/128
                 num_queries=100,      # 100
                 text_embed_dims=896,
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
        elif model_name == "dinov2_vitb14":
            self.model = dinov2_vitb14(pretrained=False)
        elif model_name == "dinov2_vitl14":
            self.model = dinov2_vitl14(pretrained=False)
        elif model_name == "dinov2_vitg14":
            self.model = dinov2_vitg14(pretrained=False)
        else:
            raise ValueError(f"dinov2 model_name {model_name} not supported")

        if checkpoint_path is not None:
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=torch.device('cpu')),
                strict=True)  # otherwise all the processes will put the loaded weight on rank 0 and may lead to CUDA OOM

        self.query_embed = nn.Embedding(num_queries, embed_dims)   # instance query 均匀分布进行随机初始化
        self.pts_embed = nn.Embedding(num_queries, 2)              # instance pts        

        # self.text_proj = nn.Linear(text_embed_dims, embed_dims)  # 文本特征投影到与3D高斯特征相同的维度
        # self.text_lv_fuse = TextImgAttLayers(
        #     text_embed_dims=embed_dims,
        #     img_embed_dims=embed_dims,
        #     num_layers=4
        # )

        # self.query_fuse = VLFusionAttLayers(
        #     embed_dims=embed_dims,
        #     text_embed_dims=text_embed_dims,
        #     num_layers=8
        # )

        if freeze:  ### 默认不fine_tune dinov2
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True        

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

        
        # self.vl_fuses = nn.ModuleList([
        #     ImgTextSelfCrossFusion(
        #         img_dim=embed_dims,
        #         txt_dim=text_embed_dims,
        #         hidden_dim=embed_dims,
        #         num_heads=4
        #     ) for _ in range(len(self.projects))
        # ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * self.hidden_dims, self.hidden_dims),
                        nn.GELU()))

        self.DANet = DANet(embed_dims, embed_dims, norm_layer=nn.BatchNorm2d)
        self.clgd = CLGD(embed_dims , embed_dims, norm_layer=nn.BatchNorm2d)

        

    def forward(self, imgs, use_danet=False, text_embeds=None):
        B, _, h, w = imgs.shape
        # print(f'imgs.shape: {imgs.shape}')
        
        # 调整输入图像尺寸为能被patch_size整除
        new_h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
        new_w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size
        
        if new_w != w or new_h != h:
            # print(f'Resizing image from ({w}, {h}) to ({new_w}, {new_h}) to be divisible by patch_size {self.patch_size}')
            # 使用双线性插值调整图像尺寸
            imgs = nn.functional.interpolate(imgs, size=(new_h, new_w), mode='bilinear', align_corners=False)
            h, w = new_h, new_w
        
        feature = self.model.get_intermediate_layers(imgs, n=self.out_index, return_class_token=self.use_clstoken)
        
        feats = []
        feats_ori = []
        for i, x in enumerate(feature):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0].unsqueeze(0)
                
            # print(f'feature[{i}].x.shape: {x.shape}')
            
            # if text_embeds is not None:
            #     x = self.vl_fuses[i](x, text_embeds)
            # print(f'feature.vlf[{i}].x.shape: {x.shape}')
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], h // self.patch_size, w // self.patch_size))
            x = self.projects[i](x)
            feats_ori.append(x)
            
            if use_danet and i == len(self.projects) - 1:
                x = self.DANet(x, feats_ori[0])
            
            feats.append(x)
        # for i, x in enumerate(feats):
        #     print(f'feats[{i}].shape: {x.shape}')
        # print(f'dinov2 encoder exit')
        
        bs = feats[0].size(0)
        
        return dict(                                                    
            queries=self.query_embed.weight.repeat(bs, 1, 1),
            feats=feats,
            encoder_feat_ori=feature[-1][0],
            pred_pts=self.pts_embed.weight.repeat(bs, 1, 1).sigmoid())

if __name__ == '__main__':
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')