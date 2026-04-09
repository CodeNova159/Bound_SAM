import os

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from .SAM.image_encoder import ImageEncoderViT
from .SAM.mask_decoder import MaskDecoder
from .SAM.prompt_encoder import PromptEncoder
from .SAM import TwoWayTransformer
from .SAM.common import LayerNorm2d

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from segment_anything.build_sam import load_from
import math
import matplotlib.pyplot as plt
import numpy as np

from model.igaf import IGAF
from model.cgpa import CGPA
from model.bafd import Decoder

class LoRA_qkv(nn.Module):

    def __init__(self, qkv: nn.Linear, r: int):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.r = r

        self.lora_a_q = nn.Linear(self.dim, r, bias=False)
        self.lora_b_q = nn.Linear(r, self.dim, bias=False)
        self.lora_a_k = nn.Linear(self.dim, r, bias=False)
        self.lora_b_k = nn.Linear(r, self.dim, bias=False)
        self.lora_a_v = nn.Linear(self.dim, r, bias=False)
        self.lora_b_v = nn.Linear(r, self.dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_q.weight)
        nn.init.kaiming_uniform_(self.lora_a_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_k.weight)
        nn.init.kaiming_uniform_(self.lora_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_v.weight)

    def forward(self, x):
        qkv_out = self.qkv(x)  # (B, N, 3*dim)

        delta_q = self.lora_b_q(self.lora_a_q(x))
        delta_k = self.lora_b_k(self.lora_a_k(x))
        delta_v = self.lora_b_v(self.lora_a_v(x))

        qkv_out[..., :self.dim] += delta_q
        qkv_out[..., self.dim:2 * self.dim] += delta_k
        qkv_out[..., -self.dim:] += delta_v
        return qkv_out

def init_network(device=None, use_lora=True, lora_rank=4, lora_layers=None):
    sam = sam_model_registry["vit_b"](checkpoint="./model/ckpt/sam_vit_b_01ec64.pth")
    image_encoder = sam.image_encoder
    image_encoder.to(device)

# ------------------------------------------------------------------------------------------------
    if use_lora:
        if lora_layers is None:
            lora_layers = list(range(len(sam.image_encoder.blocks)))

        for p in sam.image_encoder.parameters():
            p.requires_grad = False

        for i, blk in enumerate(sam.image_encoder.blocks):
            if i in lora_layers:
                blk.attn.qkv = LoRA_qkv(blk.attn.qkv, r=lora_rank)

    image_encoder.pos_embed.requires_grad = True

    for name, param in image_encoder.named_parameters():
        if "rel_pos" in name:
            param.requires_grad = True

    del sam

    return image_encoder

class Bound_SAM(nn.Module):
    def __init__(
            self,
            image_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder

        self.igaf1 = IGAF(channels=768, num_layers=3)
        self.igaf2 = IGAF(channels=768, num_layers=3)
        self.igaf3 = IGAF(channels=768, num_layers=3)
        self.igaf4 = IGAF(channels=768, num_layers=3)

        self.cgpa = CGPA(in_channels=768, out_channels=256)

        self.decoder = Decoder(fpn_channels=256, num_classes=1)

    def forward(self, image):

        B, _, H_img, W_img = image.shape


        image_embedding, feature_list = self.image_encoder(image)

        features1 = feature_list[:3]
        features2 = feature_list[3:6]
        features3 = feature_list[6:9]
        features4 = feature_list[9:12]

        g1 = self.igaf1(features1)
        g2 = self.igaf2(features2)
        g3 = self.igaf3(features3)
        g4 = self.igaf4(features4)


        p1, p2, p3, p4 = self.cgpa(g1, g2, g3, g4)

        mask, edge_map = self.decoder(p1, p2, p3, p4, original_size=(224, 224))

        return mask, edge_map
