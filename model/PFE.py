import torch
import torch.nn as nn
import math


class ERA5_PhysicsConv(nn.Module):
    def __init__(self, dx=0.25, dy=0.25):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.in_channels = 16

        kernel_dx = torch.tensor([[[[0, 0, 0],
                                    [-1, 0, 1],
                                    [0, 0, 0]]]], dtype=torch.float32) / (2 * self.dx)
        self.conv_dx = nn.Conv2d(
            self.in_channels, self.in_channels,
            kernel_size=3, padding=1, groups=self.in_channels, bias=False
        )
        self.conv_dx.weight = nn.Parameter(kernel_dx.repeat(self.in_channels, 1, 1, 1), requires_grad=False)

        kernel_dy = torch.tensor([[[[0, -1, 0],
                                    [0, 0, 0],
                                    [0, 1, 0]]]], dtype=torch.float32) / (2 * self.dy)
        self.conv_dy = nn.Conv2d(
            self.in_channels, self.in_channels,
            kernel_size=3, padding=1, groups=self.in_channels, bias=False
        )
        self.conv_dy.weight = nn.Parameter(kernel_dy.repeat(self.in_channels, 1, 1, 1), requires_grad=False)

    def forward(self, x):
        phy_feat_list = []
        dx = self.conv_dx(x)
        dy = self.conv_dy(x)

        du500_dx = dx[:, 9:10]
        du500_dy = dy[:, 9:10]
        dv500_dx = dx[:, 13:14]
        dv500_dy = dy[:, 13:14]
        du850_dx = dx[:, 10:11]
        du850_dy = dy[:, 10:11]
        dv850_dx = dx[:, 14:15]
        dv850_dy = dy[:, 14:15]

        vort500 = dv500_dx - du500_dy
        phy_feat_list.append(vort500)
        vort850 = dv850_dx - du850_dy
        phy_feat_list.append(vort850)
        div500 = du500_dx + dv500_dy
        phy_feat_list.append(div500)
        div850 = du850_dx + dv850_dy
        phy_feat_list.append(div850)

        phys_feat = torch.cat(phy_feat_list, dim=1)
        return phys_feat


class TimeDerivative6h(nn.Module):
    def __init__(self):
        super().__init__()
        self.dt = 6.0

    def forward(self, x_seq):

        B, T, C, H, W = x_seq.shape
        d_dt = (x_seq[:, 1:] - x_seq[:, :-1]) / self.dt
        pad = torch.zeros(B, 1, C, H, W, device=d_dt.device, dtype=d_dt.dtype)
        d_dt_padded = torch.cat([d_dt, pad], dim=1)
        return d_dt_padded


def calculate_physics_features(track):
    delta = torch.zeros_like(track)
    delta[:, 1:, :] = track[:, 1:, :] - track[:, :-1, :]
    d_lat, d_lon = delta[:, :, 0:1], delta[:, :, 1:2]
    speed = torch.sqrt(d_lat ** 2 + d_lon ** 2 + 1e-6)
    heading = torch.atan2(d_lon, d_lat + 1e-6) / math.pi
    lat_rad = (track[:, :, 0:1] * 5 + 30.0) * (math.pi / 180.0)
    coriolis = torch.sin(lat_rad)
    return torch.cat([d_lat, d_lon, speed, heading, coriolis], dim=-1)
