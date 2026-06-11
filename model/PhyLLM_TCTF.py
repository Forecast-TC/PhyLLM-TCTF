import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from .encoder import HimawariEncoder, ERA5Encoder
from .LSTM_LoRA import applay_LSTM_LoRA
from .SGCF import SGCF_Module
from .PFE import calculate_physics_features


class TyphoonTrackPredictor(nn.Module):
    def __init__(self,
                 pred_len=4,
                 track_dim=2,
                 feat_dim=128,
                 llm_ckp_dir=None,
                 word_size=1000,
                 dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.track_dim = track_dim
        self.d_model = 768

        self.himawari_encoder = HimawariEncoder(in_channels=4, out_channels=feat_dim)
        self.era5_encoder = ERA5Encoder(in_channels=36, out_channels=feat_dim)

        self.physics_proj = nn.Linear(5, self.d_model)
        self.env_proj = nn.Linear(128 + 128, self.d_model)

        self.gpt2 = GPT2Model.from_pretrained(llm_ckp_dir) if llm_ckp_dir else GPT2Model(GPT2Config(hidden_size=768))
        for p in self.gpt2.parameters():
            p.requires_grad = False

        self.gpt2 = applay_LSTM_LoRA(self.gpt2, rank=16)

        self.sgcf = SGCF_Module(d_model=self.d_model)

        self.word_size = word_size
        self.word_projector = nn.Linear(self.d_model, self.d_model)

        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, pred_len * track_dim)
        )

    def forward(self, track, era5, himawari):
        B, T, _ = track.shape

        e_feat = self.era5_encoder(era5)
        h_feat = self.himawari_encoder(himawari)
        env_h = self.env_proj(torch.cat([e_feat, h_feat], dim=-1))

        track_phys = calculate_physics_features(track)
        track_h = self.physics_proj(track_phys)

        H = env_h + track_h

        S_raw = self.gpt2.wte.weight[:self.word_size, :].unsqueeze(0).repeat(B, 1, 1)
        S = self.word_projector(S_raw)

        enhanced_H = self.sgcf(S, H)

        outputs = self.gpt2(inputs_embeds=enhanced_H)

        last_token = outputs.last_hidden_state[:, -1, :]
        pred_deltas = self.output_head(last_token).reshape(B, self.pred_len, self.track_dim)

        last_pos = track[:, -1, :2].unsqueeze(1)
        pred_track = []
        curr = last_pos
        for i in range(self.pred_len):
            pred = curr + pred_deltas[:, i, :].unsqueeze(1)
            pred_track.append(pred)

        return torch.cat(pred_track, dim=1)


if __name__ == '__main__':
    B, T = 8, 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    track = torch.randn(B, T, 2).to(device)
    era5 = torch.randn(B, T, 16, 81, 81).to(device)
    himawari = torch.randn(B, T, 4, 401, 401).to(device)

    model = TyphoonTrackPredictor(llm_ckp_dir='./LLM/gpt2-local').to(device)
    out = model(track, era5, himawari)

    print("Input sequence length:", T)
    print("Output prediction length:", out.shape[1])
    print("Output shape:", out.shape)