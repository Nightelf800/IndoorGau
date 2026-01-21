import torch
import torch.nn as nn

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


class VLFusionAttLayers(nn.Module):
    """
    图像-文本融合模块，包含以下特性：
    1. 图像特征初始形状为(b, embed_dims, h, w)
    2. 转换为(b, embed_dims, h*w)维度以满足注意力层需要
    3. 图像自注意力
    4. 与文本特征的交叉注意力
    5. FFN处理
    6. 保持原始图像特征形状(b, embed_dims, h, w)
    7. 共4层融合模块
    """
    def __init__(self, embed_dims, text_embed_dims, num_layers=4, num_heads=8, ffn_dim=1024):
        super().__init__()
        self.embed_dims = embed_dims
        self.text_embed_dims = text_embed_dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        
        # 将文本特征投影到与图像特征相同的维度
        self.text_projection = nn.Linear(text_embed_dims, embed_dims)
        
        # 创建4层融合模块
        self.fusion_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                # 图像自注意力层
                # 注意：这里embed_dim是图像特征的通道数，batch_first=False因为我们会转置特征
                'img_self_attn': nn.MultiheadAttention(
                    embed_dim=embed_dims,
                    num_heads=num_heads,
                    batch_first=True  # 我们会使用(batch, seq_len, embed_dim)格式
                ),
                # 视觉-语言交叉注意力层
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=embed_dims,
                    num_heads=num_heads,
                    batch_first=True
                ),
                # FFN层
                'ffn': nn.Sequential(
                    nn.Linear(embed_dims, ffn_dim),
                    nn.GELU(),
                    nn.Linear(ffn_dim, embed_dims)
                ),
                # 归一化层
                'norm1': nn.LayerNorm(embed_dims),
                'norm2': nn.LayerNorm(embed_dims),
                'norm3': nn.LayerNorm(embed_dims),
                # Dropout层
                'dropout': nn.Dropout(0.5)
            })
            self.fusion_layers.append(layer)
        
        # 最终归一化层
        self.final_norm = nn.LayerNorm(embed_dims)
    
    def forward(self, img_features, text_features, text_attention_mask=None):
        """
        前向传播
        
        参数：
            img_features: 图像特征，形状为 [batch_size, embed_dims, h, w]
            text_features: 文本特征，形状为 [batch_size, seq_len, text_embed_dims]
            text_attention_mask: 文本注意力掩码，形状为 [batch_size, seq_len]
            
        返回：
            fused_features: 融合后的特征，形状为 [batch_size, embed_dims, h, w]
        """
        # 保存原始形状信息
        batch_size, num_patches, embed_dims = img_features.shape
        # num_patches = h * w
        
        # 1. 将文本特征投影到与图像特征相同的维度
        projected_text = self.text_projection(text_features)
        
        # 2. 处理文本注意力掩码
        if text_attention_mask is not None:
            # MultiheadAttention期望的是[batch_size, seq_len]形状的key_padding_mask
            # key_padding_mask中True表示需要被忽略的位置
            key_padding_mask = ~text_attention_mask.bool()
        else:
            key_padding_mask = None
        
        # 3. 转换图像特征维度
        # 从 [batch_size, embed_dims, h, w] 转换为 [batch_size, h*w, embed_dims]
        # x = img_features.view(batch_size, embed_dims, num_patches).transpose(1, 2)
        x = img_features
        
        # 4. 经过4层融合模块
        for layer in self.fusion_layers:
            # 图像自注意力
            self_attn_output, _ = layer['img_self_attn'](
                query=x,  # 转换为 [seq_len, batch_size, embed_dim]
                key=x,
                value=x,
                key_padding_mask=None  # 图像特征没有padding
            )
            
            # 残差连接和归一化
            x = layer['norm1'](x + layer['dropout'](self_attn_output))
            
            # 交叉注意力 - 图像特征作为query，文本特征作为key和value

            # print(f'x.shape: {x.shape}')
            # print(f'projected_text.shape: {projected_text.shape}')
            cross_attn_output, _ = layer['cross_attn'](
                query=x,  # 转换为 [seq_len, batch_size, embed_dim]
                key=projected_text,
                value=projected_text,
                key_padding_mask=key_padding_mask
            )
            
            # 残差连接和归一化
            x = layer['norm2'](x + layer['dropout'](cross_attn_output))
            
            # FFN
            ffn_output = layer['ffn'](x)
            
            # 残差连接和归一化
            x = layer['norm3'](x + layer['dropout'](ffn_output))
        
        # 最终归一化
        fused_features = self.final_norm(x)
       
        # 5. 转换回原始形状
        # 从 [batch_size, h*w, embed_dims] 转换为 [batch_size, embed_dims, h, w]
        # fused_features = fused_features.transpose(1, 2).view(batch_size, embed_dims, h, w)
        
        # 确保输出形状与输入保持一致
        assert fused_features.shape == img_features.shape, f"输出形状 {fused_features.shape} 与输入形状 {img_features.shape} 不一致"
        
        return fused_features


class TextImgAttLayers(nn.Module):
    """
    文本-图像注意力层模块，包含以下特性：
    1. 输入是文本特征 text_embed，形状为 [batch_size, seq_len, text_embed_dims]
    2. 经过n层的self_attention
    3. 与图像特征 img_feat 做 cross_attention
    4. 经过FFN和残差连接
    5. 最后返回处理后的文本特征 text_embed
    """
    def __init__(self, text_embed_dims, img_embed_dims, num_layers=4, num_heads=8, ffn_dim=1024):
        super().__init__()
        self.text_embed_dims = text_embed_dims
        self.img_embed_dims = img_embed_dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        
        # 将图像特征投影到与文本特征相同的维度
        self.img_projection = nn.Linear(img_embed_dims, text_embed_dims)
        
        # 创建n层注意力模块
        self.att_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                # 文本自注意力层
                'text_self_attn': nn.MultiheadAttention(
                    embed_dim=text_embed_dims,
                    num_heads=num_heads,
                    batch_first=True
                ),
                # 文本-图像交叉注意力层
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=text_embed_dims,
                    num_heads=num_heads,
                    batch_first=True
                ),
                # FFN层
                'ffn': nn.Sequential(
                    nn.Linear(text_embed_dims, ffn_dim),
                    nn.GELU(),
                    nn.Linear(ffn_dim, text_embed_dims)
                ),
                # 归一化层
                'norm1': nn.LayerNorm(text_embed_dims),
                'norm2': nn.LayerNorm(text_embed_dims),
                'norm3': nn.LayerNorm(text_embed_dims),
                # Dropout层
                'dropout': nn.Dropout(0.5)
            })
            self.att_layers.append(layer)
        
        # 最终归一化层
        self.final_norm = nn.LayerNorm(text_embed_dims)
    
    def forward(self, text_features, img_features, text_attention_mask=None):
        """
        前向传播
        
        参数：
            text_features: 文本特征，形状为 [batch_size, seq_len, text_embed_dims]
            img_features: 图像特征，形状为 [batch_size, num_patches, img_embed_dims]
            text_attention_mask: 文本注意力掩码，形状为 [batch_size, seq_len]
            img_attention_mask: 图像注意力掩码，形状为 [batch_size, num_patches]
            
        返回：
            processed_text: 处理后的文本特征，形状为 [batch_size, seq_len, text_embed_dims]
        """
        # 1. 将图像特征投影到与文本特征相同的维度
        projected_img = self.img_projection(img_features)
        
        # 2. 处理文本注意力掩码
        if text_attention_mask is not None:
            # MultiheadAttention期望的是[batch_size, seq_len]形状的key_padding_mask
            # key_padding_mask中True表示需要被忽略的位置
            text_key_padding_mask = ~text_attention_mask.bool()
        else:
            text_key_padding_mask = None
        # 4. 初始化文本特征
        x = text_features
        
        # 5. 经过n层注意力模块
        for layer in self.att_layers:
            # 文本自注意力
            self_attn_output, _ = layer['text_self_attn'](
                query=x,
                key=x,
                value=x,
                key_padding_mask=text_key_padding_mask
            )
            
            # 残差连接和归一化
            x = layer['norm1'](x + layer['dropout'](self_attn_output))
            
            # 交叉注意力 - 文本特征作为query，图像特征作为key和value
            cross_attn_output, _ = layer['cross_attn'](
                query=x,
                key=projected_img,
                value=projected_img,
                key_padding_mask=None
            )
            
            # 残差连接和归一化
            x = layer['norm2'](x + layer['dropout'](cross_attn_output))
            
            # FFN
            ffn_output = layer['ffn'](x)
            
            # 残差连接和归一化
            x = layer['norm3'](x + layer['dropout'](ffn_output))
        
        # 最终归一化
        processed_text = self.final_norm(x)
        
        # 确保输出形状与输入保持一致
        assert processed_text.shape == text_features.shape, f"输出形状 {processed_text.shape} 与输入形状 {text_features.shape} 不一致"
        
        return processed_text


class VisualLanguageFusion3D(nn.Module):
    """
    更适合3D视觉领域的视觉-语言融合模块
    结合文本特征池化、多尺度特征融合、特征调制和空间注意力引导
    """
    def __init__(self, img_embed_dims, text_embed_dims, output_dims):
        super().__init__()
        self.img_embed_dims = img_embed_dims
        self.text_embed_dims = text_embed_dims
        self.output_dims = output_dims
        
        # 文本特征处理
        # 1. 全局文本特征（池化）
        self.text_global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        # 2. 文本特征投影
        self.text_projection = nn.Linear(text_embed_dims, img_embed_dims)
        
        # 特征调制模块
        # 使用文本特征调制图像特征的通道权重
        self.channel_modulation = nn.Sequential(
            nn.Linear(img_embed_dims, img_embed_dims),
            nn.Sigmoid()
        )
        
        # 空间注意力引导
        # 使用文本特征生成空间注意力图
        self.spatial_attention = nn.Sequential(
            nn.Linear(img_embed_dims, img_embed_dims),
            nn.ReLU(),
            nn.Linear(img_embed_dims, 1)
        )
        
        # 多尺度融合控制
        self.scale_weight = nn.Parameter(torch.ones(1))
        
        # 输出处理
        self.output_projection = nn.Linear(img_embed_dims, output_dims)
        self.norm = nn.LayerNorm(output_dims)
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
        # 1. 文本特征处理
        # 计算全局文本特征
        if text_attention_mask is not None:
            # 应用注意力掩码进行加权池化
            text_features_masked = text_features * text_attention_mask.unsqueeze(-1)
            text_global = self.text_global_pool(text_features_masked.transpose(1, 2))
        else:
            text_global = self.text_global_pool(text_features.transpose(1, 2))
        
        # 投影文本特征
        projected_text = self.text_projection(text_global)  # [batch_size, img_embed_dims]
        
        # 2. 特征调制
        # 使用文本特征调制图像特征的通道权重
        channel_weights = self.channel_modulation(projected_text)  # [batch_size, img_embed_dims]
        img_features_modulated = img_features * channel_weights.unsqueeze(1)  # [batch_size, num_patches, img_embed_dims]
        
        # 3. 空间注意力引导
        # 计算空间注意力权重
        spatial_attn = self.spatial_attention(img_features_modulated)  # [batch_size, num_patches, 1]
        spatial_attn = torch.softmax(spatial_attn, dim=1)  # 在空间维度上归一化
        
        # 应用空间注意力
        img_features_attended = img_features_modulated * spatial_attn  # [batch_size, num_patches, img_embed_dims]
        
        # 4. 残差融合
        # 将调制和注意力引导后的特征与原始特征融合
        fused_features = img_features + self.dropout(img_features_attended)
        
        # 5. 输出投影
        fused_features = self.output_projection(fused_features)
        fused_features = self.norm(fused_features)
        
        return fused_features
