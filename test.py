import torch
import torch.nn.functional as F
import sys, os
import warnings
import matplotlib.pyplot as plt
import logging

from model.initial_sapmedsam import init_network
from trainer import train_net, get_data

from model.initial_sapmedsam import Bound_SAM

from monai.metrics import compute_hausdorff_distance, DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.losses import DiceCELoss
from monai.losses import DiceCELoss, DiceLoss
from tqdm import tqdm
from model.utils import AverageMeter

warnings.filterwarnings("ignore", category=UserWarning)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


def test_net(model, test_loader, device, pretrained_weights, save_dir=None):
    model.eval()

    if os.path.exists(pretrained_weights):
        print(f"Loading checkpoint from {pretrained_weights}...")
        checkpoint = torch.load(pretrained_weights, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    criterion = DiceCELoss(include_background=False, to_onehot_y=True, sigmoid=True, softmax=False,
                           lambda_dice=1, lambda_ce=1)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean",get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean",get_not_nans=False)

    test_loss = AverageMeter("Val Loss", ":.4f")

    with torch.no_grad():


        ce_loss = torch.nn.BCEWithLogitsLoss()
        dice_loss = DiceLoss(sigmoid=True)

        test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {1} Validation", leave=True, position=0)

        for idx, batch_data in enumerate(test_loader_tqdm):
            inputs, labels, name = batch_data["image"].to(device), batch_data["label"].to(device), batch_data["name"]

            with torch.cuda.amp.autocast():
                pred, edge_map = model(inputs)

                pred = F.interpolate(
                    pred,
                    size=(labels.shape[2], labels.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )

                lc1, lc2 = 0.6, 0.4

                loss_ce = ce_loss(pred, labels)
                loss_dice = dice_loss(pred, labels.float())

                loss = (lc1 * loss_ce + lc2 * loss_dice)

            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()

            dice_metric(y_pred=pred, y=labels.long())
            iou_metric(y_pred=pred, y=labels.long())
            hd95_metric(y_pred=pred, y=labels.long())

            save_path = os.path.join(save_dir, name[0] + f".png")
            plt.imsave(save_path, pred.squeeze().cpu().numpy(), cmap="gray")

            test_loader_tqdm.set_postfix(loss = loss.item())

            test_loss.update(loss.item(), inputs.size(0))

        mean_dice = dice_metric.aggregate().item()
        mean_iou = iou_metric.aggregate().item()
        mean_hausdorff = hd95_metric.aggregate().item()

    return mean_dice, mean_iou, mean_hausdorff



if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pretrained_weights = ""

    save_dir = ""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_loader, val_loader = get_data(base_folder="./data/", batch_size=1, return_test=True, debug=False)

    image_encoder = init_network(device=device)

    model = Bound_SAM(
        image_encoder=image_encoder,
    ).to(device)


    dice, iou, hd95 = test_net(model, val_loader, device, pretrained_weights, save_dir)

    print(f"average dice: {dice:.4f} average iou: {iou:.4f} average hd95: {hd95:.4f}")

