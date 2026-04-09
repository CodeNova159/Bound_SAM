import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TotalLoss(nn.Module):
    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_edge: float = 0.5
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_edge = lambda_edge
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        smooth = 1e-5
        inter = (pred * target).sum(dim=(2, 3))
        dice = (2 * inter + smooth) / (
            pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth
        )
        return 1 - dice.mean()

    def forward(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        pred_edge: Optional[torch.Tensor] = None
    ) -> dict:

        l_bce = self.bce(pred_mask, gt_mask)
        l_dice = self.dice_loss(pred_mask, gt_mask)
        l_seg = 0.6 * l_bce + 0.4 * l_dice

        losses = {
            'loss_total': self.lambda_seg * l_seg,
            'loss_seg': l_seg,
            'loss_bce': l_bce,
            'loss_dice': l_dice,
        }

        if pred_edge is not None:
            edge_gt = self._generate_edge_gt(gt_mask)
            l_edge = self.bce(pred_edge, edge_gt)
            losses['loss_edge'] = l_edge
            losses['loss_total'] = losses['loss_total'] + self.lambda_edge * l_edge
        else:
            losses['loss_edge'] = torch.tensor(0.0, device=pred_mask.device)

        return losses

    @staticmethod
    def _generate_edge_gt(mask_gt: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(mask_gt, kernel_size=5, stride=1, padding=2)
        eroded = -F.max_pool2d(-mask_gt, kernel_size=5, stride=1, padding=2)
        return (dilated - eroded).clamp(0, 1)
