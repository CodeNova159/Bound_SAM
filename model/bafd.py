import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union

def _make_direction_kernel() -> torch.Tensor:

    kernels = torch.zeros(4, 1, 3, 3)

    kernels[0, 0] = torch.tensor([[-1., -1., -1.],
                                   [ 0.,  0.,  0.],
                                   [ 1.,  1.,  1.]])

    kernels[1, 0] = torch.tensor([[-1.,  0.,  1.],
                                   [-1.,  0.,  1.],
                                   [-1.,  0.,  1.]])

    kernels[2, 0] = torch.tensor([[-1., -1.,  0.],
                                   [-1.,  0.,  1.],
                                   [ 0.,  1.,  1.]])

    kernels[3, 0] = torch.tensor([[ 0., -1., -1.],
                                   [ 1.,  0., -1.],
                                   [ 1.,  1.,  0.]])
    return kernels


class DirectionAwareBoundaryDetector(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        self.dir_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
            for _ in range(4)
        ])

        ref_kernels = _make_direction_kernel()
        for d, conv in enumerate(self.dir_convs):
            with torch.no_grad():
                conv.weight.copy_(ref_kernels[d].expand(channels, -1, -1, -1))

        self.dir_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 4, channels * 4 // 8, 1),
            nn.GELU(),
            nn.Conv2d(channels * 4 // 8, channels * 4, 1),
            nn.Sigmoid()
        )

        self.boundary_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 0.02))

    def forward(self, x):
        dir_responses = [conv(x) for conv in self.dir_convs]
        dir_cat = torch.cat(dir_responses, dim=1)  # (B, 4C, H, W)

        dir_w = self.dir_attn(dir_cat)  # (B, 4C,1,1)
        dir_w = dir_w.view(x.size(0), 4, self.channels, 1, 1)

        weighted = sum(
            dir_w[:, d] * dir_responses[d]
            for d in range(4)
        )

        boundary_map = self.boundary_gate(weighted) + weighted

        out = x + self.gamma * boundary_map * x

        return out, boundary_map


class SpatialAttentionGate(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_feat = x.max(dim=1, keepdim=True)[0]
        avg_feat = x.mean(dim=1, keepdim=True)
        spatial_w = self.attn(torch.cat([max_feat, avg_feat], dim=1))
        return x * spatial_w

class BAFD(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.boundary_detector = DirectionAwareBoundaryDetector(channels)
        self.spatial_attn      = SpatialAttentionGate(channels)

    def forward(
        self,
        x   : torch.Tensor,
        skip: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if skip is not None:
            x = x + F.interpolate(
                skip, size=x.shape[-2:],
                mode='bilinear', align_corners=False
            )
        x = self.fuse_conv(x)
        x, boundary_map = self.boundary_detector(x)
        x = self.spatial_attn(x)
        return x, boundary_map

class AuxEdgeHead(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class Decoder(nn.Module):
    def __init__(self, fpn_channels=256, num_classes=1):
        super().__init__()
        C = fpn_channels

        self.down2 = nn.Sequential(
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C), nn.GELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C), nn.GELU()
        )

        self.level4 = BAFD(C)
        self.level3 = BAFD(C)
        self.level2 = BAFD(C)
        self.level1 = BAFD(C)

        self.seg_head = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.GELU(),
            nn.Conv2d(C // 2, C // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(C // 4),
            nn.GELU(),
            nn.Conv2d(C // 4, num_classes, 1)
        )

        self.edge_head = nn.Sequential(
            nn.Conv2d(C, C // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(C // 4),
            nn.GELU(),
            nn.Conv2d(C // 4, 1, 1)
        )

    def forward(self, f1, f2, f3, f4, original_size):
        f4_down2 = self.down2(f4)
        f4_down3 = self.down3(f4_down2)

        d4, _ = self.level4(f4_down3)
        d3, _ = self.level3(f4_down2, d4)

        f23 = (f2 + f3) / 2
        d2, _ = self.level2(f23, d3)
        d1, bm1 = self.level1(f1, d2)

        d1_up = F.interpolate(d1, size=original_size, mode='bilinear', align_corners=False)
        bm1_up = F.interpolate(bm1, size=original_size, mode='bilinear', align_corners=False)

        mask = self.seg_head(d1_up)
        edge_map = self.edge_head(bm1_up)

        return mask, edge_map

    def generate_edge_gt(
        mask_gt: torch.Tensor,
        kernel_size: int = 5
    ) -> torch.Tensor:

        dilated = F.max_pool2d(
            mask_gt, kernel_size=kernel_size,
            stride=1, padding=kernel_size // 2
        )
        eroded  = -F.max_pool2d(
            -mask_gt, kernel_size=kernel_size,
            stride=1, padding=kernel_size // 2
        )
        return (dilated - eroded).clamp(0, 1)

    def compute_edge_loss(
        edge_map: torch.Tensor,
        aux_edges: List[torch.Tensor],
        edge_gt: torch.Tensor,
        main_w: float = 0.4,
        aux_w: float = 0.1
    ) -> torch.Tensor:

        pos_weight = torch.tensor([10.0], device=edge_map.device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        loss = main_w * bce(edge_map, edge_gt)
        for aux_e in aux_edges:
            loss = loss + aux_w * bce(aux_e, edge_gt)

        return loss
