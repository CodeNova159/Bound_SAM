import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class IGAF(nn.Module):
    def __init__(self, channels: int = 768, num_layers: int = 3, embed_dim: int = 192):
        super().__init__()
        self.num_layers = num_layers
        self.channels = channels
        self.embed_dim = embed_dim

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU()
            )
            for _ in range(num_layers)
        ])

        self.score_mlp = nn.Sequential(
            nn.Linear(num_layers, num_layers * 2),
            nn.GELU(),
            nn.Linear(num_layers * 2, 1)
        )

        self.align_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        assert len(layer_features) == self.num_layers

        B, C, H, W = layer_features[0].shape
        L = self.num_layers

        proj_feats = [self.proj[i](layer_features[i]) for i in range(L)]

        proj_feats = [f.flatten(2) for f in proj_feats]
        proj_feats = [F.normalize(f, dim=1) for f in proj_feats]

        sim_rows = []
        for i in range(L):
            row = []
            for j in range(L):
                sim_ij = (proj_feats[i] * proj_feats[j]).sum(dim=1).mean(dim=1, keepdim=True)
                row.append(sim_ij)
            row = torch.cat(row, dim=1)
            sim_rows.append(row)

        relation_matrix = torch.stack(sim_rows, dim=1)

        raw_w = torch.cat(
            [self.score_mlp(relation_matrix[:, i, :]) for i in range(L)],
            dim=1
        )

        weights = torch.softmax(raw_w, dim=1)

        fused = sum(
            weights[:, i].view(B, 1, 1, 1) * layer_features[i]
            for i in range(L)
        )

        fused = fused + layer_features[-1]
        fused = self.align_conv(fused)

        return fused