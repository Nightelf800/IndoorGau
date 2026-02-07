import torch
import torch.nn as nn
from torchvision.models.resnet import resnet34, resnet50, resnet101, resnet152


class ResNet(nn.Module):
    def __init__(self, in_channels=3, embed_dims=256, scales=None, backbone="resnet50", dropout_rate=0.2,
                 pretrained=True):
        super(ResNet, self).__init__()

        # ----------------------------------------------------------------------------- #
        # Encoder
        # ----------------------------------------------------------------------------- #
        if backbone == "resnet34":
            net = resnet34(pretrained)
            self.expansion = 1
        elif backbone == "resnet50":
            net = resnet50(pretrained)
            self.expansion = 4
        elif backbone == "resnet101":
            net = resnet101(pretrained)
            self.expansion = 4
        elif backbone == "resnet152":
            net = resnet152(pretrained)
            self.expansion = 4
        else:
            raise NotImplementedError("invalid backbone: {}".format(backbone))
        
        self.feature_channels = [64 * self.expansion, 128 * self.expansion, 256 * self.expansion, 512 * self.expansion]
        self.backbone_name = backbone

        # Note that we do not downsample for conv1
        # self.conv1 = net.conv1
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        if in_channels == 3:
            self.conv1.weight.data = net.conv1.weight.data
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        # dropout
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=512,
                out_channels=embed_dims,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.Conv2d(
                in_channels=1024,
                out_channels=embed_dims,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.Conv2d(
                in_channels=2048,
                out_channels=embed_dims,
                kernel_size=3,
                stride=1,
                padding=1)
        ])

    def forward(self, x):
        # pad input to be divisible by 16 = 2 ** 4
        h, w = x.shape[2], x.shape[3]
        # check input size
        if h % 16 != 0 or w % 16 != 0:
            assert False, "invalid input size: {}".format(x.shape)

        # ----------------------------------------------------------------------------- #
        # Encoder
        # ----------------------------------------------------------------------------- #
        # inter_features = []
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        layer1_out = self.layer1(self.maxpool(conv1_out))
        layer2_out = self.layer2(layer1_out)  # downsample
        layer3_out = self.dropout(self.layer3(layer2_out))  # downsample
        layer4_out = self.dropout(self.layer4(layer3_out))  # downsample
        
        layer2_out = self.projects[0](layer2_out)
        layer3_out = self.projects[1](layer3_out)
        layer4_out = self.projects[2](layer4_out)
        
        feats = [layer2_out, layer3_out, layer4_out]
        
        return dict(                                                   
            feats=feats,
            encoder_feat_ori=feats[-1])