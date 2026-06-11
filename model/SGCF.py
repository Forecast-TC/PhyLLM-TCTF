import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SGCF_Module(nn.Module):

    def __init__(self, d_model=768):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, S, H):
        Q = self.q_proj(S)
        K = self.k_proj(H)
        V = self.v_proj(H)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        weights = F.softmax(scores, dim=-1)
        C = torch.matmul(weights, V)

        semantic_info = C.mean(dim=1, keepdim=True)
        combined = torch.cat([H, semantic_info.expand(-1, H.size(1), -1)], dim=-1)
        g = self.gate(combined)

        return g * H + (1 - g) * semantic_info
