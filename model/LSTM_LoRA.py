import torch
import torch.nn as nn
import math


class LSTM_LoRA(nn.Module):
    def __init__(self, original_layer, d_model=768, rank=16):
        super().__init__()
        self.original_layer = original_layer
        self.d_model = d_model
        self.rank = rank

        self.lora_A = nn.Linear(d_model, rank)
        self.lstm_long = nn.LSTM(rank, rank, batch_first=True)
        self.lstm_short = nn.LSTM(rank, rank, batch_first=True)
        self.lora_B = nn.Linear(rank, d_model)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        res = self.original_layer(x)

        a = self.lora_A(x)
        l, _ = self.lstm_long(a)
        s, _ = self.lstm_short(l)
        b = self.lora_B(s)

        b_expanded = torch.cat([b, b, b], dim=-1)

        return res + b_expanded

def applay_LSTM_LoRA(model, rank=16):
    for block in model.h:
        block.attn.c_attn = LSTM_LoRA(block.attn.c_attn, d_model=768, rank=rank)
    return model
