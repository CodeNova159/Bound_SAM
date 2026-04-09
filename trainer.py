import logging
import os
import random
import numpy as np
import time
import pandas as pd
import glob

from matplotlib.textpath import text_to_path
from monai.transforms import (RandFlipd,
                              RandShiftIntensityd,
                              RandRotate90d)
from monai.transforms import (ThresholdIntensityd, NormalizeIntensityd, ScaleIntensityd, ToTensord, NormalizeIntensity, CastToTyped)
from model.utils import AverageMeter, exp_lr_scheduler_with_warmup, log_evaluation_result, get_optimizer
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, PersistentDataset

from monai.metrics import compute_hausdorff_distance, DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.losses import DiceCELoss, DiceLoss
import torchvision.transforms as transforms
import torch
import warnings
from PIL import Image
# from training.loss import DiceLoss
from torch.nn.modules.loss import CrossEntropyLoss

warnings.filterwarnings("ignore", category=UserWarning)
import torch.nn.functional as F
# from .script_utils import save_checkpoint
from torch.optim import AdamW
import yaml
import logging
import gc
from tqdm import tqdm
import shutil
from scipy import ndimage
from pathlib import Path
from preprocess.visualize import visualize_pred_and_image
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torchvision
from model.loss import TotalLoss

def random_rot_flip_pil(image, label):
    k = random.randint(0, 3)
    # 旋转 k*90 度
    image = image.rotate(90 * k)
    label = label.rotate(90 * k)

    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)

    return image, label

def random_rotate_pil(image, label):
    angle = random.randint(-20, 20)
    image = image.rotate(angle, resample=Image.BILINEAR)
    label = label.rotate(angle, resample=Image.NEAREST)
    return image, label

class LungDataset(Dataset):
    def __init__(self, data, image_transform=None, label_transform=None, augment=False):
        self.data = data
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]["image"]
        label_path = self.data[idx]["label"]

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        if self.augment:
            if random.random() < 0.5:
                image, label = random_rot_flip_pil(image, label)
            if random.random() < 0.5:
                image, label = random_rotate_pil(image, label)

        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)


        return {"image": image, "label": label, "name": Path(image_path).stem}

def get_data(base_folder="./data/Lung/",
             batch_size=1, return_test=False, debug=False):
    # Load image names for training, validation, and testing
    with open(os.path.join(base_folder, 'list', 'train.yaml'), 'r') as f:
        img_name_list_train = yaml.load(f, Loader=yaml.BaseLoader)

    with open(os.path.join(base_folder, 'list', 'val.yaml'), 'r') as f:
        img_name_list_val = yaml.load(f, Loader=yaml.BaseLoader)

    with open(os.path.join(base_folder, 'list', 'test.yaml'), 'r') as f:
        img_name_list_test = yaml.load(f, Loader=yaml.BaseLoader)

    # Prepare file paths for training, validation, and testing
    train_files = [{"image": f'{base_folder}/train/input/{name}.jpg', "label": f'{base_folder}/train/label/{name}.png'} for name in img_name_list_train]
    val_files = [{"image": f'{base_folder}/val/input/{name}.jpg', "label": f'{base_folder}/val/label/{name}.png'} for name in img_name_list_val]
    test_files = [{"image": f'{base_folder}/test/input/{name}.jpg', "label": f'{base_folder}/test/label/{name}.png'} for name in img_name_list_test]

    if debug:
        train_files = train_files[:int(len(train_files) * 0.1)]
        val_files = val_files[:int(len(val_files) * 0.1)]
        test_files = test_files[:int(len(test_files) * 0.1)]
        print(f'10% of data is used for training, validation, and testing. (debugging)')
    else:
        train_files = train_files[:int(len(train_files) * 1)]
        val_files = val_files[:int(len(val_files) * 1)]
        test_files = test_files[:int(len(test_files) * 1)]

    print('Training files:', len(train_files), '\nValidation files:', len(val_files), '\nTest files:', len(test_files))

    image_transforms = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    label_transforms = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # If test data is needed, return the test loader
    if return_test:
        test_ds = LungDataset(data=test_files, image_transform=image_transforms, label_transform=label_transforms)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        val_ds = LungDataset(data=val_files, image_transform=image_transforms, label_transform=label_transforms)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        return test_loader, val_loader
    else:
        # Return train and validation loaders
        train_ds = LungDataset(data=train_files, image_transform=image_transforms, label_transform=label_transforms, augment=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_ds = LungDataset(data=test_files, image_transform=image_transforms, label_transform=label_transforms)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader


def validate_net(model, val_loader, device, logger=None, epoch=0):
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean", get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean", get_not_nans=False)


    with torch.no_grad():

        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=True, position=0)

        for idx, batch_data in enumerate(val_loader_tqdm):
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            with torch.cuda.amp.autocast():

                pred, _ = model(inputs)

            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()

            dice_metric(y_pred=pred, y=labels.long())
            iou_metric(y_pred=pred, y=labels.long())
            hd95_metric(y_pred=pred, y=labels.long())

        mean_dice = dice_metric.aggregate().item()
        mean_iou = iou_metric.aggregate().item()
        mean_hausdorff = hd95_metric.aggregate().item()

        if logger:
            logger.info(f"Epoch {epoch + 1} , Dice Score: {mean_dice:.6f} , IoU: {mean_iou:.6f} , Hausdorff: {mean_hausdorff:.6f}")

    return mean_dice, mean_iou, mean_hausdorff


def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def train_net(model, train_loader, val_loader, device, args, logger=None, writer=None):

    max_epochs = args.max_epochs
    val_interval = args.val_interval

    optimizer = AdamW([i for i in model.parameters() if i.requires_grad], lr=args.base_lr, weight_decay=1e-7)

    best_dice = 0.0

    if os.path.exists(args.pretrained_weights):
        logger.info(f"Loading checkpoint from {args.pretrained_weights}...")
        checkpoint = torch.load(args.pretrained_weights, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    for epoch in range(max_epochs):

        model.train()

        scaler = torch.cuda.amp.GradScaler()

        epoch_mask_loss = 0
        epoch_slice = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}", total=len(train_loader))

        for idx, batch_data in enumerate(train_loader_tqdm):
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)


            with torch.cuda.amp.autocast():

                pred, edge_map = model(inputs)

                criterion = TotalLoss(lambda_seg=1.0, lambda_edge=0.5)
                losses = criterion(pred, labels, edge_map)

            scaler.scale(losses['loss_total']).backward()
            scaler.step(optimizer)

            scaler.update()
            optimizer.zero_grad()

            epoch_mask_loss += losses['loss_total'].item()
            epoch_slice += 1
            train_loader_tqdm.set_postfix(mask_loss=losses['loss_total'].item())

        avg_mask_loss = epoch_mask_loss / epoch_slice

        logger.info(f"epoch {epoch + 1} average loss: {avg_mask_loss:.7f}")
        writer.add_scalar('train_mask_loss', avg_mask_loss, epoch + 1)

        ################################################################################################################

        if (epoch + 1) % val_interval == 0:
            mean_dice, mean_iou, mean_hd95 = validate_net(model, val_loader, device, logger, epoch)

            csv_path = os.path.join(args.log_path, "train.csv")

            epoch_data = {
                "Epoch": epoch + 1,
                "Train Loss": f"{avg_mask_loss:.6f}",
                "Dice Score": f"{mean_dice:.6f}",
                "IoU Score": f"{mean_iou:.6f}",
                "HD95 Score": f"{mean_hd95:.6f}"
            }

            df = pd.DataFrame([epoch_data])

            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, index=False)

            if mean_dice > best_dice:
                best_dice = mean_dice

                best_checkpoint_path = os.path.join(args.log_path, "best.pth.tar")

                if os.path.exists(best_checkpoint_path):
                    os.remove(best_checkpoint_path)

                save_checkpoint(model, optimizer, best_checkpoint_path)

                if logger:
                    logger.info(f"New best model saved with Dice Score: {best_dice:.6f}")
