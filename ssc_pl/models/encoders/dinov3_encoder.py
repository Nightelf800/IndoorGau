from maskdino.models.maskdino_decoder_layers import F
import torch
import torch.nn as nn
from third_party.dinov3.dinov3.hub.backbones import dinov3_vitb16

class Dinov3Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 model_name,
                 checkpoint_path,
                 patch = 16,
                 embed_dims=256,       # 64/128
                 num_queries=100,      # 100
                 freeze=True,
                 use_clstoken = False):
        # 首先调用父类的初始化方法
        super().__init__()
        self.intermediate_layer_idx = {
            'dinov3_vits16': [5, 8, 11],
            'dinov3_vitb16': [5, 8, 11], 
            'dinov3_vitl16': [11, 17, 23], 
            'dinov3_vitg16': [19, 29, 39]
            }
        self.patch_size = patch
        self.hidden_dims = in_channels
        self.use_clstoken = use_clstoken
        self.out_index = self.intermediate_layer_idx[model_name]
        if model_name == "dinov3_vitb16":
            self.dinov3_model = dinov3_vitb16(pretrained=False)
        else:
            raise ValueError(f"dinov3 model_name {model_name} not supported")

        self.query_embed = nn.Embedding(num_queries, embed_dims)
        self.pts_embed = nn.Embedding(num_queries, 2)

        if freeze:  ### 默认不fine_tune dinov3
            for param in self.dinov3_model.parameters():
                param.requires_grad = False
        else:
            for param in self.dinov3_model.parameters():
                param.requires_grad = True        
        self.projects = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=self.hidden_dims,
                out_channels=embed_dims * 4,
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dims,
                out_channels=embed_dims * 4,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Conv2d(
                in_channels=self.hidden_dims,
                out_channels=embed_dims * 4,
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
        
        # 添加输入卷积层，将3通道输入转换为embed_dims通道
        self.ouput_proj = nn.Conv2d(
            in_channels=embed_dims * 4,
            out_channels=embed_dims,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        B, _, w, h = x.shape
        # print(f'self.dinov3_model: {self.dinov3_model}')
        # print(f'x.shape: {x.shape}')
        
        # 将投影后的特征输入到dinov3模型
        intermediate_features = self.dinov3_model.get_intermediate_layers(x, n=self.out_index, return_class_token=self.use_clstoken)

        # for i, f in enumerate(intermediate_features):
        #     if isinstance(f, tuple):
        #         print(f'f[{i}].f.shape: {f[0].shape}')
        #         print(f'f[{i}].cls_token.shape: {f[1].shape}')
        #     else:
        #         print(f'f[{i}].shape: {f.shape}')

        features = []
        for i, f in enumerate(intermediate_features):
            if self.use_clstoken:
                f, cls_token = f[0], f[1]
                readout = cls_token.unsqueeze(1).expand_as(f)
                f = self.readout_projects[i](torch.cat((f, readout), -1))
            
            # print(f'f[{i}].shape: {f.shape}')
            f = f.permute(0, 2, 1).reshape((f.shape[0], f.shape[-1], w // self.patch_size, h // self.patch_size))
            
            f = self.projects[i](f)

            f = self.ouput_proj(f)

            features.append(f)
        # print(f'dinov3.outputs.instance: {type(features)}')
        # for i, output in enumerate(features):
        #     print(f'dinov3.outputs[{i}].shape: {output.shape}')
        # print(f'dinov3.outputs.shape: {outputs.shape}')

        return dict(                                                   
            queries=self.query_embed.weight.repeat(B, 1, 1),
            feats=features,
            pred_pts=self.pts_embed.weight.repeat(B, 1, 1).sigmoid())

if __name__ == "__main__":
    model = Dinov3Encoder(
        in_channels=1024,
        model_name="dinov3_vitb16",
        checkpoint_path="/data2/ylc/codes/IndoorGau/new_module_test/facebook/dinov3-vitb16-pretrain-lvd1689m",
    )   
    print(model.dinov3_model)
