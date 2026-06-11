import torch
import torch.nn as nn
from .PFE import ERA5_PhysicsConv, TimeDerivative6h


class ST_LKA_Attention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ST_LKA_Attention, self).__init__()

        self.temp_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.temp_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )

        self.spatial_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=in_channels, bias=False),
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x_temp = self.temp_pool(x)
        att_temp = self.temp_conv(x_temp)

        x_spatial = self.spatial_pool(x)
        att_spatial = self.spatial_conv(x_spatial)

        attention_map = torch.sigmoid(att_temp + att_spatial)

        return x * attention_map


class ERA5Encoder(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super(ERA5Encoder, self).__init__()

        self.phy_conv = ERA5_PhysicsConv()
        self.time_d = TimeDerivative6h()

        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, (3, 3, 3), (1, 2, 2), (1, 1, 1)),  # 修改了输入通道
            nn.BatchNorm3d(32),
            nn.ReLU(),
            ST_LKA_Attention(32)
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            ST_LKA_Attention(64)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            ST_LKA_Attention(128)
        )

        self.pool1 = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool2 = nn.AdaptiveAvgPool3d((None, 2, 2))
        self.pool4 = nn.AdaptiveAvgPool3d((None, 4, 4))

        self.spp_feat_dim = 128 * 21
        self.linear = nn.Sequential(
            nn.Linear(self.spp_feat_dim, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        time_d = self.time_d(x)

        phy_list = []
        for i in range(T):
            phy_input_step = self.phy_conv(x[:, i, :, :, :])
            phy_list.append(phy_input_step.unsqueeze(1))

        phy_input = torch.cat(phy_list, dim=1)
        phy_input = torch.cat([x, time_d, phy_input], dim=2)
        input = phy_input.permute(0, 2, 1, 3, 4).contiguous()
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)

        x1 = self.pool1(out).view(B, -1, T, 1)
        x2 = self.pool2(out).view(B, -1, T, 4)
        x4 = self.pool4(out).view(B, -1, T, 16)

        x_cat = torch.cat([x1, x2, x4], dim=-1).permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        feat = self.linear(x_cat)
        return feat


class HimawariEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super(HimawariEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            nn.ReLU(),
            ST_LKA_Attention(32)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            nn.ReLU(),
            ST_LKA_Attention(64)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            nn.ReLU(),
            ST_LKA_Attention(128)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(128, 256, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            nn.ReLU(),
            ST_LKA_Attention(256)
        )
        self.layer5 = nn.Sequential(
            nn.Conv3d(256, 128, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            nn.ReLU(),
            ST_LKA_Attention(128)
        )
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(128, out_channels, 1, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fusion(x)
        return x.squeeze(-1).squeeze(-1).permute(0, 2, 1).contiguous()
