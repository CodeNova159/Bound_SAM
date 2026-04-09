import torch
import torch.nn as nn
from typing import Tuple


class CGPA(nn.Module):
    def __init__(self, in_channels: int = 768, out_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = 4
        total_channels = in_channels * self.num_groups
        reduction = 16
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(total_channels, total_channels // reduction),
            nn.GELU(),
            nn.Linear(total_channels // reduction, total_channels),
            nn.Sigmoid()
        )

        self.compress = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
            for _ in range(self.num_groups)
        ])

        self.residual = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
            for _ in range(self.num_groups)
        ])

    def forward(
        self,
        g1: torch.Tensor,  # (B, 768, H, W)
        g2: torch.Tensor,  # (B, 768, H, W)
        g3: torch.Tensor,  # (B, 768, H, W)
        g4: torch.Tensor,  # (B, 768, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        groups = [g1, g2, g3, g4]
        B, C, H, W = g1.shape

        concat = torch.cat(groups, dim=1)   # (B, 3072, H, W)

        attn_weights = self.channel_attn(concat)            # (B, 3072)
        attn_weights = attn_weights.view(B, -1, 1, 1)       # (B, 3072, 1, 1)
        concat_attended = concat * attn_weights              # (B, 3072, H, W)

        outputs = []
        for i in range(self.num_groups):
            group_feat = concat_attended[
                :, i * C: (i + 1) * C, :, :
            ]                                               # (B, 768, H, W)

            compressed = self.compress[i](group_feat)       # (B, 256, H, W)

            residual = self.residual[i](groups[i])        # (B, 256, H, W)
            outputs.append(compressed + residual)

        return tuple(outputs)
