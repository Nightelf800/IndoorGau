import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):

    def __init__(self, channels, norm_layer):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            norm_layer(channels),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            norm_layer(channels),
        )

    def forward(self, x):
        out = x + self.convs(x)
        return F.relu_(out)


class BottleneckDDR3D(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size=3,
                 stride=1,
                 dilation=(1, 1, 1),
                 expansion=4,
                 downsample=None,
                 norm_layer=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = norm_layer(channels)

        self.conv2 = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 1, kernel_size),
            stride=(1, 1, stride),
            padding=(0, 0, dilation[0]),
            dilation=(1, 1, dilation[0]),
            bias=False,
        )
        self.bn2 = norm_layer(channels)
        self.conv3 = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, kernel_size, 1),
            stride=(1, stride, 1),
            padding=(0, dilation[1], 0),
            dilation=(1, dilation[1], 1),
            bias=False,
        )
        self.bn3 = norm_layer(channels)
        self.conv4 = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            stride=(stride, 1, 1),
            padding=(dilation[2], 0, 0),
            dilation=(dilation[2], 1, 1),
            bias=False,
        )
        self.bn4 = norm_layer(channels)

        self.conv5 = nn.Conv3d(channels, channels * expansion, kernel_size=1, bias=False)
        self.bn5 = norm_layer(channels * expansion)

        self.stride = stride
        self.downsample = downsample
        if stride != 1:
            self.downsample2 = nn.Sequential(
                nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
                nn.Conv3d(channels, channels, kernel_size=1, stride=1, bias=False),
                norm_layer(channels),
            )
            self.downsample3 = nn.Sequential(
                nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
                nn.Conv3d(channels, channels, kernel_size=1, stride=1, bias=False),
                norm_layer(channels),
            )
            self.downsample4 = nn.Sequential(
                nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
                nn.Conv3d(channels, channels, kernel_size=1, stride=1, bias=False),
                norm_layer(channels),
            )

    def forward(self, x):
        out1 = F.relu_(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = F.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out2 + out3
        out3_relu = F.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out2 + out3 + out4
        out4_relu = F.relu(out4)

        out5 = self.bn5(self.conv5(out4_relu))
        if self.downsample is not None:
            x = self.downsample(x)
        out = F.relu_(x + out5)
        return out

class DDRUnit3D(nn.Module):
    def __init__(self, c_in, c, c_out, kernel=3, stride=1, dilation=1, residual=True,
                 batch_norm=False, inst_norm=False):
        super(DDRUnit3D, self).__init__()
        s = stride
        k = kernel
        d = dilation
        p = k // 2 * d
        self.batch_norm = batch_norm
        self.conv_in = nn.Conv3d(c_in, c, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(c) if batch_norm else None
        self.conv1x1x3 = nn.Conv3d(c, c, (1, 1, k), stride=s, padding=(0, 0, p), bias=True, dilation=(1, 1, d))
        self.conv1x3x1 = nn.Conv3d(c, c, (1, k, 1), stride=s, padding=(0, p, 0), bias=True, dilation=(1, d, 1))
        self.conv3x1x1 = nn.Conv3d(c, c, (k, 1, 1), stride=s, padding=(p, 0, 0), bias=True, dilation=(d, 1, 1))
        self.bn4 = nn.BatchNorm3d(c) if batch_norm else None
        self.conv_out = nn.Conv3d(c, c_out, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm3d(c_out) if batch_norm else None
        self.residual = residual
        self.conv_resid = nn.Conv3d(c_in, c_out, kernel_size=1, bias=False) if residual and c_in != c_out else None
        self.inst_norm = nn.InstanceNorm3d(c_out) if inst_norm else None

    def forward(self, x):
        y0 = self.conv_in(x)
        if self.batch_norm:
            y0 = self.bn1(y0)
        y0 = F.relu(y0, inplace=True)

        y1 = self.conv1x1x3(y0)
        y1 = F.relu(y1, inplace=True)

        y2 = self.conv1x3x1(y1) + y1
        y2 = F.relu(y2, inplace=True)

        y3 = self.conv3x1x1(y2) + y2 + y1
        if self.batch_norm:
            y3 = self.bn4(y3)
        y3 = F.relu(y3, inplace=True)

        y = self.conv_out(y3)
        if self.batch_norm:
            y = self.bn5(y)

        x_squip = x if self.conv_resid is None else self.conv_resid(x)

        y = y + x_squip if self.residual else y

        y = self.inst_norm(y) if self.inst_norm else y

        y = F.relu(y, inplace=True)

        return y

class DDRBlock3D(nn.Module):
    def __init__(self, c_in, c, c_out, units=2, kernel=3, stride=1, dilation=1,
                 pool=True, residual=True, batch_norm=False, inst_norm=False):
        super(DDRBlock3D, self).__init__()
        self.pool = nn.MaxPool3d(2, stride=2) if pool else None
        self.units = nn.ModuleList()
        for i in range(units):
            if i == 0:
                self.units.append(DDRUnit3D(c_in, c, c_out, kernel, stride, dilation, residual, batch_norm, inst_norm))
            else:
                self.units.append(DDRUnit3D(c_out, c, c_out, kernel, stride, dilation, residual, batch_norm, inst_norm))

    def forward(self, x):
        y = self.pool(x) if self.pool is not None else x
        for ddr_unit in self.units:
            y = ddr_unit(y)
        return y