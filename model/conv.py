import torch.nn as nn
import torch


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def define_norm(norm, num_channels, learnable_norm):
    norm = norm.lower()
    # Batch Normalization
    if norm == 'bn':
        norm_obj = nn.BatchNorm2d(num_channels, affine=learnable_norm)
    # Instance Normalization
    elif norm == 'in':
        norm_obj = nn.InstanceNorm2d(num_channels, affine=learnable_norm)

    # No Normalization
    else:
        norm_obj = Identity()

    return norm_obj


class ResBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm='BN',
                 learnable_norm=True,
                 need_out_relu=True
                 ):
        super().__init__()

        self.learnable_norm = learnable_norm
        self.need_out_relu = need_out_relu

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3, stride=stride, padding=1
        )
        self.norm1 = define_norm(norm, out_channels, learnable_norm)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.norm2 = define_norm(norm, out_channels, learnable_norm)

        self.match_dimensions = (in_channels != out_channels) or (stride != 1)
        if self.match_dimensions:
            self.match_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1, stride=stride
            )
            self.match_norm = define_norm(norm, out_channels, learnable_norm)

    def forward(self, x):
        residual = x
        if self.match_dimensions:
            residual = self.match_conv(residual)
            residual = self.match_norm(residual)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + residual
        if self.need_out_relu:
            out = self.relu(out)

        return out


class Up(nn.Module):

    def __init__(self,
                 n_feats,
                 scale,
                 norm='BN',
                 learnable_norm=True,
                 need_out_relu=True
                 ):
        super().__init__()

        self.learnable_norm = learnable_norm
        self.need_out_relu = need_out_relu

        self.conv = nn.Conv2d(
            n_feats,
            n_feats * scale * scale,
            kernel_size=3, stride=1, padding=1
        )
        self.up = nn.PixelShuffle(scale)
        self.norm = define_norm(norm, n_feats, learnable_norm)
        if need_out_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        x = self.norm(x)
        if self.need_out_relu:
            x = self.relu(x)
        return x


class UpX2Res(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm='BN',
                 learnable_norm=True,
                 need_out_relu=True
                 ):
        super().__init__()

        self.up = Up(
            n_feats=in_channels,
            scale=2,
            norm=norm,
            learnable_norm=learnable_norm,
            need_out_relu=True
        )

        self.res = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            norm=norm,
            learnable_norm=learnable_norm,
            need_out_relu=need_out_relu
        )

    def forward(self, x):
        x = self.up(x)
        x = self.res(x)
        return x
