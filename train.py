import contextlib
import io
import json
import math
import random
import tempfile
import time
import warnings
from collections import defaultdict
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import box_convert, generalized_box_iou
from torchvision.ops import nms as torchvision_nms
from tqdm import tqdm

CFG = dict(
    # 路徑
    data_dir="/content/nycu-hw2-data",
    save_dir="/content/drive/MyDrive/HW2_checkpoints",
    # 模型架構
    image_size=256,
    num_digit_classes=10,
    num_object_queries=100,
    transformer_dim=256,
    num_attn_heads=8,
    encoder_depth=6,
    decoder_depth=6,
    ffn_dim=512,
    attn_dropout=0.1,
    num_sample_points=2,
    num_feature_levels=4,
    # 訓練參數
    batch_size=4,
    total_epochs=50,
    peak_lr=1e-4,
    backbone_lr=1e-5,
    weight_decay=1e-4,
    warmup_epochs=5,
    gradient_clip=0.1,
    mixed_precision="bf16",
    eval_every=3,
    num_workers=2,
    random_seed=42,
    # Loss 係數
    w_cls=1.0,
    w_l1=5.0,
    w_giou=2.0,
    bg_coef=0.1,
    aux_loss_weight=0.5,
    confidence_thr=0.05,
    nms_threshold=0.5,
    max_predictions=30,
)

Path(CFG["save_dir"]).mkdir(parents=True, exist_ok=True)
random.seed(CFG["random_seed"])
np.random.seed(CFG["random_seed"])
torch.manual_seed(CFG["random_seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | Mixed Precision: {CFG['mixed_precision']}")

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# 資料集


def load_annotation_file(ann_path: str):
    """讀取 COCO 格式的 annotation JSON"""
    with open(ann_path) as fp:
        raw = json.load(fp)
    img_meta = {x["id"]: x for x in raw["images"]}
    ann_by_img = defaultdict(list)
    for ann in raw["annotations"]:
        if not ann.get("iscrowd", 0):
            ann_by_img[ann["image_id"]].append(ann)
    return img_meta, ann_by_img


def resize_with_padding(pil_img: Image.Image, target: int):
    orig_w, orig_h = pil_img.size
    ratio = min(target / orig_w, target / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target, target), color=(114, 114, 114))
    offset_x = (target - new_w) // 2
    offset_y = (target - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas, ratio, offset_x, offset_y


def build_augmentation_pipeline():
    return A.Compose([
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.4),
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.2, hue=0.05, p=0.5),
        A.Affine(scale=(0.85, 1.15), rotate=0, shear=0, p=0.5),
    ], bbox_params=A.BboxParams(
        format="coco",
        label_fields=["digit_labels"],
        min_visibility=0.3,
    ))


class DigitDetectionDataset(Dataset):

    def __init__(self, img_dir, ann_path, image_size=256, is_training=True):
        self.img_dir = Path(img_dir)
        self.image_size = image_size
        self.is_training = is_training
        self.aug_pipeline = (
            build_augmentation_pipeline() if is_training else None)
        self.img_meta, self.annotations = load_annotation_file(ann_path)
        self.all_img_ids = list(self.img_meta.keys())

    def __len__(self):
        return len(self.all_img_ids)

    def __getitem__(self, index):
        img_id = self.all_img_ids[index]
        meta = self.img_meta[img_id]
        img = Image.open(self.img_dir / meta["file_name"]).convert("RGB")

        # 收集 bounding boxes（xyxy 格式）
        raw_boxes, raw_labels = [], []
        for ann in self.annotations[img_id]:
            x, y, w, h = ann["bbox"]
            if w > 1 and h > 1:
                raw_boxes.append([x, y, x + w, y + h])
                raw_labels.append(ann["category_id"])

        # Letterbox resize
        img_padded, scale, pad_x, pad_y = resize_with_padding(
            img, self.image_size)

        # 將 box 座標對應到 padded image
        scaled_boxes = [
            [x1 * scale + pad_x, y1 * scale + pad_y,
             x2 * scale + pad_x, y2 * scale + pad_y]
            for x1, y1, x2, y2 in raw_boxes
        ]

        # 訓練時做資料增強
        if self.is_training and self.aug_pipeline and scaled_boxes:
            img_padded, scaled_boxes, raw_labels = self._augment(
                img_padded, scaled_boxes, raw_labels)

        # Clip boxes 到圖片範圍內
        S = float(self.image_size)
        final_boxes, final_labels = [], []
        for (x1, y1, x2, y2), lbl in zip(scaled_boxes, raw_labels):
            x1, y1 = max(0., x1), max(0., y1)
            x2, y2 = min(S, x2), min(S, y2)
            if x2 - x1 > 2 and y2 - y1 > 2:
                final_boxes.append([x1, y1, x2, y2])
                final_labels.append(lbl)

        # 轉成 tensor
        img_tensor = TF.normalize(TF.to_tensor(img_padded), IMG_MEAN, IMG_STD)

        if final_boxes:
            box_tensor = box_convert(
                torch.tensor(
                    final_boxes,
                    dtype=torch.float32),
                "xyxy",
                "cxcywh")
            box_tensor = (box_tensor / self.image_size).clamp(0., 1.)
            valid_mask = (box_tensor[:, 2] > 1e-4) & (box_tensor[:, 3] > 1e-4)
            box_tensor = box_tensor[valid_mask]
            # category_id 是 1-indexed，轉成 0-indexed
            lbl_tensor = (
                torch.tensor(
                    final_labels,
                    dtype=torch.long) -
                1)[valid_mask]
        else:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            lbl_tensor = torch.zeros((0,), dtype=torch.long)

        return img_tensor, {
            "image_id": torch.tensor(img_id),
            "orig_size": torch.tensor([meta["height"], meta["width"]]),
            "boxes": box_tensor,
            "labels": lbl_tensor,
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
        }

    def _augment(self, img, boxes_xyxy, labels):
        """套用 albumentations 增強，轉換前後都用 coco (xywh) 格式"""
        W = H = self.image_size
        coco_boxes, valid_labels = [], []
        for (x1, y1, x2, y2), lbl in zip(boxes_xyxy, labels):
            x1, y1 = max(0., x1), max(0., y1)
            x2, y2 = min(float(W), x2), min(float(H), y2)
            if x2 - x1 > 1 and y2 - y1 > 1:
                coco_boxes.append([x1, y1, x2 - x1, y2 - y1])
                valid_labels.append(lbl)
        if not coco_boxes:
            return img, boxes_xyxy, labels
        result = self.aug_pipeline(
            image=np.array(img),
            bboxes=coco_boxes,
            digit_labels=valid_labels,
        )
        new_img = Image.fromarray(result["image"])
        new_boxes = [[x, y, x + w, y + h] for x, y, w, h in result["bboxes"]]
        new_labels = list(result["digit_labels"])
        return new_img, new_boxes, new_labels


class TestImageDataset(Dataset):
    """Test set（無 annotation），僅回傳圖片與 metadata"""

    def __init__(self, img_dir, image_size=256):
        self.image_size = image_size
        self.files = sorted(
            f for ext in ("*.png", "*.jpg", "*.jpeg")
            for f in Path(img_dir).glob(ext))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        padded, ratio, px, py = resize_with_padding(img, self.image_size)
        img_t = TF.normalize(TF.to_tensor(padded), IMG_MEAN, IMG_STD)
        return img_t, {
            "image_id": int(path.stem),
            "orig_size": (orig_h, orig_w),
            "scale": ratio, "pad_x": px, "pad_y": py,
        }


def detection_collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


# ============================================================
# 模型：Deformable DETR
# 參考論文: Zhu et al., ICLR 2021
# ============================================================

def make_sine_position_encoding(height: int, width: int,
                                embed_dim: int, device) -> torch.Tensor:
    """
    2D sine-cosine positional encoding。
    論文 Section 3.2，每個 spatial location 對應一個 embed_dim 維向量。
    回傳 shape: (H*W, embed_dim)
    """
    assert embed_dim % 4 == 0, "embed_dim 必須是 4 的倍數"
    half = embed_dim // 4

    # 每個 location 的正規化座標 (0~1)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    grid_x = (grid_x + 0.5) / max(width, 1)
    grid_y = (grid_y + 0.5) / max(height, 1)

    # 頻率向量
    freq = 1.0 / (10000 ** (torch.arange(half,
                  dtype=torch.float32, device=device) / max(half, 1)))

    # 角度 = coord * freq * 2π
    angle_x = grid_x.flatten().unsqueeze(1) * freq.unsqueeze(0) * 2.0 * math.pi
    angle_y = grid_y.flatten().unsqueeze(1) * freq.unsqueeze(0) * 2.0 * math.pi

    # sin / cos 各取一半，拼接成完整 embedding
    encoding = torch.cat([angle_x.sin(), angle_x.cos(),
                          angle_y.sin(), angle_y.cos()], dim=1)
    return encoding  # (H*W, embed_dim)


class DeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention (論文 Eq.3)。

    核心思想：每個 query 預測 K 個取樣偏移量，
    在 feature map 上做雙線性內插取值，再加權求和。
    這比 global attention 更高效，且對小物件更友善。
    """

    def __init__(self, embed_dim=256, num_heads=8,
                 num_levels=4, num_points=4):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        # 三個投影層：offset、attention weight、value
        self.offset_fc = nn.Linear(
            embed_dim, num_heads * num_levels * num_points * 2)
        self.weight_fc = nn.Linear(
            embed_dim, num_heads * num_levels * num_points)
        self.value_fc = nn.Linear(embed_dim, embed_dim)
        self.output_fc = nn.Linear(embed_dim, embed_dim)

        self._init_parameters()

    def _init_parameters(self):
        # offset 初始化：均勻分布在圓上，避免所有點堆在一起
        nn.init.constant_(self.offset_fc.weight, 0.0)
        angles = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads)
        unit = torch.stack([angles.cos(), angles.sin()], dim=-1)  # (H, 2)
        unit = unit.view(self.num_heads, 1, 1, 2).expand(
            -1, self.num_levels, self.num_points, -1).clone()
        for k in range(self.num_points):
            unit[:, :, k, :] *= (k + 1)
        with torch.no_grad():
            self.offset_fc.bias.copy_(unit.reshape(-1))

        nn.init.constant_(self.weight_fc.weight, 0.0)
        nn.init.constant_(self.weight_fc.bias, 0.0)
        nn.init.xavier_uniform_(self.value_fc.weight)
        nn.init.constant_(self.value_fc.bias, 0.0)
        nn.init.xavier_uniform_(self.output_fc.weight)
        nn.init.constant_(self.output_fc.bias, 0.0)

    def forward(self, query, ref_points, feature_flat, spatial_shapes):
        """
        Args:
            query:         (B, Nq, C)
            ref_points:    (B, Nq, num_levels, 2) — 正規化參考點 [0,1]
            feature_flat:  (B, sum(H*W), C)       — 所有 level 的 feature
            spatial_shapes: list of (H, W)
        Returns:
            output: (B, Nq, C)
        """
        B, Nq, _ = query.shape
        _, Nv, _ = feature_flat.shape

        # Value projection
        vals = self.value_fc(feature_flat).view(
            B, Nv, self.num_heads, self.head_dim)  # (B, Nv, H, D)

        # 預測取樣偏移量
        offsets = self.offset_fc(query).view(
            B, Nq, self.num_heads, self.num_levels, self.num_points, 2)

        # 注意力權重（對所有 level × point 做 softmax）
        attn_w = F.softmax(
            self.weight_fc(query).view(
                B, Nq, self.num_heads, self.num_levels * self.num_points),
            dim=-1,
        ).view(B, Nq, self.num_heads, self.num_levels, self.num_points)

        # 各 level 的 spatial 尺寸 (W, H) 用於正規化 offset
        level_sizes = torch.tensor(
            [[w, h] for h, w in spatial_shapes],
            dtype=torch.float32, device=query.device)  # (L, 2)

        # 取樣點座標 = 參考點 + 偏移量 / 尺寸，轉到 grid_sample 的 [-1,1] 空間
        sample_pts = 2.0 * (
            ref_points[:, :, None, :, None, :]          # (B, Nq, 1, L, 1, 2)
            + offsets / level_sizes[None, None, None,
                                    :, None, :]  # (B, Nq, H, L, P, 2)
        ) - 1.0  # (B, Nq, H, L, P, 2)

        # 對每個 level 做雙線性取樣
        level_sizes_flat = [h * w for h, w in spatial_shapes]
        outputs_per_level = []
        for lid, (lh, lw) in enumerate(spatial_shapes):
            # 取出這個 level 的 feature: (B*H, head_dim, lh, lw)
            feat_level = (
                vals.split(level_sizes_flat, dim=1)[lid]
                .permute(0, 2, 3, 1)
                .reshape(B * self.num_heads, self.head_dim, lh, lw)
            )
            # 取樣座標: (B*H, Nq, P, 2)
            coords = (
                sample_pts[:, :, :, lid, :, :]
                .permute(0, 2, 1, 3, 4)
                .reshape(B * self.num_heads, Nq, self.num_points, 2)
            )
            sampled = F.grid_sample(
                feat_level, coords,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )  # (B*H, head_dim, Nq, P)
            outputs_per_level.append(sampled)

        # 合併所有 level 的取樣結果：(B*H, head_dim, Nq, L*P)
        all_sampled = torch.cat(outputs_per_level, dim=-1)

        # 加權求和
        flat_w = (
            attn_w.view(
                B,
                Nq,
                self.num_heads,
                self.num_levels *
                self.num_points) .permute(
                0,
                2,
                1,
                3) .reshape(
                B *
                self.num_heads,
                1,
                Nq,
                self.num_levels *
                self.num_points))
        aggregated = (all_sampled * flat_w).sum(dim=-1)  # (B*H, head_dim, Nq)

        # 重排成 (B, Nq, embed_dim)
        out = (
            aggregated
            .view(B, self.num_heads, self.head_dim, Nq)
            .permute(0, 3, 1, 2)
            .reshape(B, Nq, self.embed_dim)
        )
        return self.output_fc(out)


class FeedForwardBlock(nn.Module):
    """標準 Transformer FFN: Linear → ReLU → Dropout → Linear"""

    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x + self.drop(self.net(x)))


class EncoderLayer(nn.Module):
    """
    Deformable DETR Encoder Layer。
    Self-attention 用 deformable attention（query = key = value = feature）。
    """

    def __init__(self, embed_dim=256, num_heads=8,
                 num_levels=4, num_points=4, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.self_attn = DeformableAttention(
            embed_dim, num_heads, num_levels, num_points)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardBlock(embed_dim, ffn_dim, dropout)

    def forward(self, features, pos_enc, ref_pts, shapes):
        # Self-attention: query = features + pos
        attended = self.self_attn(
            features + pos_enc, ref_pts, features, shapes)
        features = self.norm1(features + self.drop1(attended))
        return self.ffn(features)


class DecoderLayer(nn.Module):
    """
    Deformable DETR Decoder Layer。
    包含: self-attention (standard) → cross-attention (deformable) → FFN
    """

    def __init__(self, embed_dim=256, num_heads=8,
                 num_levels=4, num_points=4, ffn_dim=1024, dropout=0.1):
        super().__init__()
        # Self-attention（query 之間互相 attend）
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Cross-attention（query attend to encoder memory）
        self.cross_attn = DeformableAttention(
            embed_dim, num_heads, num_levels, num_points)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # FFN
        self.ffn = FeedForwardBlock(embed_dim, ffn_dim, dropout)

    def forward(self, queries, query_pos, memory, ref_pts, shapes):
        # Self-attention
        q = k = queries + query_pos
        sa_out, _ = self.self_attn(q, k, queries)
        queries = self.norm1(queries + self.drop1(sa_out))

        # Cross-attention
        ca_out = self.cross_attn(
            queries + query_pos, ref_pts, memory, shapes)
        queries = self.norm2(queries + self.drop2(ca_out))

        return self.ffn(queries)


class DeformableDETR(nn.Module):
    """
    Deformable DETR 完整模型。

    架構：
        ResNet-50 backbone (C2~C5) →
        Multi-scale feature projection →
        Deformable Encoder →
        Deformable Decoder with iterative refinement →
        Class head + Box head (per decoder layer)
    """

    def __init__(self, cfg):
        super().__init__()
        C = cfg["transformer_dim"]
        self.C = C
        self.num_levels = cfg["num_feature_levels"]
        self.num_decoder = cfg["decoder_depth"]
        self.num_queries = cfg["num_object_queries"]
        self.num_classes = cfg["num_digit_classes"]

        # ── Backbone: ResNet-50（凍結 BatchNorm）──
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # 拆解成 4 個階段，對應 C2~C5
        self.stage1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool, backbone.layer1)   # stride 4,  ch=256
        self.stage2 = backbone.layer2            # stride 8,  ch=512
        self.stage3 = backbone.layer3            # stride 16, ch=1024
        self.stage4 = backbone.layer4            # stride 32, ch=2048

        # 凍結所有 BatchNorm
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

        # ── Feature Projection：各 level 投影到相同維度 C ──
        backbone_channels = [256, 512, 1024, 2048]
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, C, kernel_size=1, bias=False),
                nn.GroupNorm(32, C),
            )
            for ch in backbone_channels
        ])

        # Level embedding：讓模型區分不同 scale 的 feature
        self.level_embeddings = nn.Parameter(
            torch.randn(self.num_levels, C) * 0.01)

        # ── Encoder ──
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(C, cfg["num_attn_heads"], self.num_levels,
                         cfg["num_sample_points"],
                         cfg["ffn_dim"], cfg["attn_dropout"])
        ])

        # ── Decoder ──
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(C, cfg["num_attn_heads"], self.num_levels,
                         cfg["num_sample_points"],
                         cfg["ffn_dim"], cfg["attn_dropout"])
            for _ in range(cfg["decoder_depth"])
        ])
        self.decoder_norm = nn.LayerNorm(C)

        # ── Object Queries：每個 query 分成 content + positional ──
        self.query_embeddings = nn.Embedding(self.num_queries, C * 2)

        # 參考點投影：從 positional query 預測初始參考點
        self.ref_point_proj = nn.Linear(C, 4)

        # ── 每個 decoder layer 各有一個 head（iterative refinement）──
        self.class_predictors = nn.ModuleList([
            nn.Linear(C, self.num_classes + 1)
            for _ in range(self.num_decoder)
        ])
        self.box_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(C, C), nn.ReLU(inplace=True),
                nn.Linear(C, C), nn.ReLU(inplace=True),
                nn.Linear(C, 4),
            )
            for _ in range(self.num_decoder)
        ])

    def train(self, mode=True):
        """覆寫 train()，確保 BatchNorm 永遠是 eval 模式"""
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def _compute_encoder_ref_points(self, spatial_shapes, device):
        """
        計算 encoder 的參考點：每個 spatial location 的中心點座標（正規化）。
        Shape: (1, sum(H*W), num_levels, 2)
        """
        ref_list = []
        for feat_h, feat_w in spatial_shapes:
            # 生成 grid，正規化到 [0,1]
            cy = torch.linspace(
                0.5,
                feat_h - 0.5,
                feat_h,
                device=device) / feat_h
            cx = torch.linspace(
                0.5,
                feat_w - 0.5,
                feat_w,
                device=device) / feat_w
            grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")
            pts = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
            ref_list.append(pts)
        ref = torch.cat(ref_list, dim=0)  # (sum(H*W), 2)
        # 擴展到所有 level（每個 location 在所有 level 都有相同的參考點）
        return ref[:, None, :].expand(-1, len(spatial_shapes), -1).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> dict:
        B = x.size(0)
        device = x.device

        # ── Backbone forward pass ──
        f1 = self.stage1(x)  # C2
        f2 = self.stage2(f1)  # C3
        f3 = self.stage3(f2)  # C4
        f4 = self.stage4(f3)  # C5

        # ── Project & flatten all levels ──
        all_features, all_positions, spatial_shapes = [], [], []
        for level_idx, feat in enumerate([f1, f2, f3, f4]):
            proj = self.projections[level_idx](feat)  # (B, C, H, W)
            _, _, H, W = proj.shape
            spatial_shapes.append((H, W))

            # Positional encoding + level embedding
            pos = make_sine_position_encoding(H, W, self.C, device)  # (H*W, C)
            pos = pos.unsqueeze(0).expand(
                B, -1, -1)                  # (B, H*W, C)
            pos = pos + self.level_embeddings[level_idx].view(1, 1, -1)

            all_features.append(
                proj.flatten(2).permute(
                    0, 2, 1))  # (B, H*W, C)
            all_positions.append(pos)

        feat_flat = torch.cat(all_features, dim=1)   # (B, sum(H*W), C)
        pos_flat = torch.cat(all_positions, dim=1)  # (B, sum(H*W), C)

        # ── Encoder ──
        enc_refs = self._compute_encoder_ref_points(
            spatial_shapes, device).expand(B, -1, -1, -1)
        memory = feat_flat
        for enc_layer in self.encoder_layers:
            memory = enc_layer(memory, pos_flat, enc_refs, spatial_shapes)

        # ── Decoder 初始化 ──
        # query_embeddings 前半是 positional（用來預測 ref points）
        # 後半是 content（初始 query 內容）
        q_pos, q_content = self.query_embeddings.weight.split(self.C, dim=-1)
        q_pos = q_pos.unsqueeze(0).expand(B, -1, -1)      # (B, Nq, C)
        queries = q_content.unsqueeze(0).expand(B, -1, -1)  # (B, Nq, C)

        # 初始參考點由 positional query 預測
        ref_points = self.ref_point_proj(q_pos).sigmoid()  # (B, Nq, 4)

        # ── Decoder（iterative refinement）──
        all_logits, all_boxes = [], []
        for dec_idx, dec_layer in enumerate(self.decoder_layers):
            # 取 ref_points 的 xy 作為 cross-attention 參考
            ref_xy = ref_points[:, :, None, :2].expand(
                -1, -1, self.num_levels, -1)  # (B, Nq, L, 2)

            queries = dec_layer(queries, q_pos, memory, ref_xy, spatial_shapes)
            normed = self.decoder_norm(queries)

            # 預測 class logits
            logits = self.class_predictors[dec_idx](normed)

            # 預測 box（residual 相對於 ref_points，套用 inverse sigmoid）
            def _inv_sigmoid(t, eps=1e-5):
                t = t.clamp(eps, 1 - eps)
                return torch.log(t / (1 - t))

            box_delta = self.box_predictors[dec_idx](normed)
            boxes = (box_delta + _inv_sigmoid(ref_points)).sigmoid()

            all_logits.append(logits)
            all_boxes.append(boxes)

            # 更新參考點（讓下一層從更準確的位置出發）
            ref_points = boxes.detach()

        return {
            "pred_logits": all_logits[-1],
            "pred_boxes": all_boxes[-1],
            "aux_outputs": [
                {"pred_logits": lg, "pred_boxes": b}
                for lg, b in zip(all_logits[:-1], all_boxes[:-1])
            ],
        }


# ============================================================
# Loss：Hungarian Matching + Set Prediction Loss
# ============================================================

class BipartiteMatcher(nn.Module):
    """
    用 Hungarian algorithm 做預測與 GT 的最佳一對一匹配。
    Cost = w_cls * focal_cost + w_l1 * L1_cost + w_giou * GIoU_cost
    """

    def __init__(self, w_cls=1., w_l1=5., w_giou=2.):
        super().__init__()
        self.w_cls, self.w_l1, self.w_giou = w_cls, w_l1, w_giou

    @torch.no_grad()
    def forward(self, predictions: dict, targets: list) -> list:
        matched_indices = []
        B = predictions["pred_logits"].shape[0]

        for b in range(B):
            gt_labels = targets[b]["labels"]
            gt_boxes = targets[b]["boxes"]

            if gt_labels.numel() == 0:
                matched_indices.append((
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long),
                ))
                continue

            pred_probs = predictions["pred_logits"][b].float().softmax(-1)
            pred_boxes = predictions["pred_boxes"][b].float()

            # Classification cost（負 log-prob）
            cls_cost = -pred_probs[:, gt_labels]

            # L1 cost
            l1_cost = torch.cdist(pred_boxes, gt_boxes.float(), p=1)

            # GIoU cost
            giou_cost = -generalized_box_iou(
                box_convert(pred_boxes, "cxcywh", "xyxy"),
                box_convert(gt_boxes.float(), "cxcywh", "xyxy"),
            )

            total_cost = (self.w_cls * cls_cost
                          + self.w_l1 * l1_cost
                          + self.w_giou * giou_cost)
            total_cost = torch.nan_to_num(
                total_cost, nan=1e4, posinf=1e4, neginf=-1e4)

            row_idx, col_idx = linear_sum_assignment(total_cost.cpu().numpy())
            matched_indices.append((
                torch.as_tensor(row_idx, dtype=torch.long),
                torch.as_tensor(col_idx, dtype=torch.long),
            ))

        return matched_indices


class DetectionLoss(nn.Module):
    """
    DETR Set Prediction Loss。
    包含：CE Loss + L1 Loss + GIoU Loss + auxiliary decoder losses
    """

    def __init__(self, cfg, matcher):
        super().__init__()
        self.nc = cfg["num_digit_classes"]
        self.matcher = matcher
        self.w_cls = cfg["w_cls"]
        self.w_l1 = cfg["w_l1"]
        self.w_giou = cfg["w_giou"]
        self.aux_w = cfg["aux_loss_weight"]

        # no-object class 的 weight 調低
        cls_weights = torch.ones(self.nc + 1)
        cls_weights[-1] = cfg["bg_coef"]
        self.register_buffer("cls_weights", cls_weights)

    def forward(self, predictions: dict, targets: list):
        indices = self.matcher(predictions, targets)

        loss_cls = self._classification_loss(predictions, targets, indices)
        loss_l1, loss_iou = self._box_loss(predictions, targets, indices)

        total = (self.w_cls * loss_cls
                 + self.w_l1 * loss_l1
                 + self.w_giou * loss_iou)

        # Auxiliary losses（每個 decoder layer 都計算一次）
        for aux in predictions.get("aux_outputs", []):
            aux_idx = self.matcher(aux, targets)
            a_cls = self._classification_loss(aux, targets, aux_idx)
            a_l1, a_iou = self._box_loss(aux, targets, aux_idx)
            total = total + self.aux_w * (
                self.w_cls * a_cls + self.w_l1 * a_l1 + self.w_giou * a_iou)

        return total, {
            "loss_cls": loss_cls,
            "loss_l1": loss_l1,
            "loss_iou": loss_iou,
        }

    def _classification_loss(self, predictions, targets, indices):
        logits = predictions["pred_logits"]
        B, Q = logits.shape[:2]
        device = logits.device

        # 預設全部是 background（no-object class = self.nc）
        target_cls = torch.full(
            (B, Q), self.nc, dtype=torch.long, device=device)
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx):
                target_cls[b, pred_idx] = targets[b]["labels"][gt_idx]

        weights = self.cls_weights.to(device)
        return F.cross_entropy(
            logits.float().flatten(0, 1),
            target_cls.flatten(),
            weight=weights,
        )

    def _box_loss(self, predictions, targets, indices):
        device = predictions["pred_boxes"].device
        matched_pred, matched_gt = [], []

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            matched_pred.append(predictions["pred_boxes"][b][pred_idx])
            matched_gt.append(targets[b]["boxes"][gt_idx])

        if not matched_pred:
            zero = torch.tensor(0., device=device)
            return zero, zero

        src = torch.cat(matched_pred).float().clamp(1e-6, 1 - 1e-6)
        tgt = torch.cat(matched_gt).float().clamp(1e-6, 1 - 1e-6)
        N = src.shape[0]

        l1_loss = F.l1_loss(src, tgt, reduction="sum") / N

        # GIoU（需要 xyxy 格式，且確保 x2>x1, y2>y1）
        def to_valid_xyxy(boxes_cxcywh):
            bx = box_convert(boxes_cxcywh, "cxcywh", "xyxy")
            x1, y1, x2, y2 = bx.unbind(-1)
            return torch.stack([x1, y1,
                                torch.maximum(x2, x1 + 1e-4),
                                torch.maximum(y2, y1 + 1e-4)], dim=-1)

        giou = torch.diag(generalized_box_iou(
            to_valid_xyxy(src), to_valid_xyxy(tgt)))
        giou_loss = (1 - torch.nan_to_num(giou, nan=0.)).sum() / N

        return l1_loss, giou_loss


# ============================================================
# 訓練輔助函式
# ============================================================

def setup_amp(device, mode: str):
    """設定 AMP（自動混合精度）"""
    if device.type != "cuda" or mode == "none":
        return False, None, False
    if mode == "bf16":
        return True, torch.bfloat16, False
    return True, torch.float16, True   # fp16 需要 GradScaler


def build_lr_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup + Cosine decay"""
    def lr_fn(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = min((epoch - warmup_epochs) /
                       max(1, total_epochs - warmup_epochs), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def train_one_epoch(model, loss_fn, optimizer, loader,
                    device, epoch, scaler, amp_en, amp_dt, clip_norm):
    model.train()
    running_loss, n_valid, n_skipped = 0., 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]",
                leave=False, dynamic_ncols=True)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=amp_dt, enabled=amp_en):
            preds = model(images)

        # Cast to fp32 for loss calculation
        preds_fp32 = {
            k: v.float() if isinstance(v, torch.Tensor)
            and v.is_floating_point() else v
            for k, v in preds.items()
        }

        if not (preds_fp32["pred_logits"].isfinite().all() and
                preds_fp32["pred_boxes"].isfinite().all()):
            n_skipped += 1
            continue

        loss, loss_dict = loss_fn(preds_fp32, targets)

        if not loss.isfinite():
            n_skipped += 1
            continue

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        n_valid += 1
        running_loss += loss.item()
        pbar.set_postfix(
            loss=f"{running_loss / n_valid:.4f}",
            cls=f"{loss_dict['loss_cls'].item():.3f}",
            l1=f"{loss_dict['loss_l1'].item():.3f}",
            iou=f"{loss_dict['loss_iou'].item():.3f}",
        )

    pbar.close()
    if n_skipped:
        print(f"  [warn] skipped {n_skipped} batches (NaN/Inf)")
    return running_loss / max(n_valid, 1)


def per_class_nms(boxes, scores, labels, iou_threshold):
    """對每個類別分別做 NMS，避免不同數字互相抑制"""
    if len(scores) == 0:
        return boxes, scores, labels
    keep_list = []
    for cls_id in labels.unique():
        mask = labels == cls_id
        orig_idx = mask.nonzero(as_tuple=True)[0]
        kept = torchvision_nms(
            boxes[mask].float(), scores[mask].float(), iou_threshold)
        keep_list.append(orig_idx[kept])
    keep = torch.cat(keep_list)
    keep = keep[scores[keep].argsort(descending=True)]
    return boxes[keep], scores[keep], labels[keep]


@torch.no_grad()
def compute_val_map(model, loader, device, val_json, img_size,
                    amp_en, amp_dt, conf_thr=0.05, nms_iou=0.5, max_det=30):
    """在 validation set 上計算 COCO mAP"""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    all_predictions = []

    for images, targets in tqdm(loader, desc="            [mAP]",
                                leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=amp_dt, enabled=amp_en):
            preds = model(images)

        for i, t in enumerate(targets):
            probs = preds["pred_logits"][i].float().softmax(-1)
            scores, cids = probs[:, :-1].max(-1)
            keep = scores > conf_thr
            if not keep.any():
                continue

            sc, cid = scores[keep], cids[keep]
            boxes_xyxy = box_convert(
                preds["pred_boxes"][i][keep].float() * img_size,
                "cxcywh", "xyxy")

            # 反正規化回原始圖片座標
            orig_h, orig_w = t["orig_size"].tolist()
            px, py, ratio = t["pad_x"], t["pad_y"], t["scale"]
            boxes_xyxy[:, [0, 2]] = (
                (boxes_xyxy[:, [0, 2]] - px) / ratio).clamp(0, orig_w)
            boxes_xyxy[:, [1, 3]] = (
                (boxes_xyxy[:, [1, 3]] - py) / ratio).clamp(0, orig_h)

            # Per-class NMS + max predictions cap
            boxes_xyxy, sc, cid = per_class_nms(boxes_xyxy, sc, cid, nms_iou)
            if len(sc) > max_det:
                topk = sc.topk(max_det)
                boxes_xyxy, cid, sc = (boxes_xyxy[topk.indices],
                                       cid[topk.indices], topk.values)

            boxes_xywh = box_convert(boxes_xyxy, "xyxy", "xywh")
            img_id = t["image_id"].item()

            for s, c, b in zip(sc.tolist(), cid.tolist(), boxes_xywh.tolist()):
                all_predictions.append({
                    "image_id": img_id,
                    "category_id": int(c) + 1,  # 轉回 1-indexed
                    "bbox": b,
                    "score": float(s),
                })

    if not all_predictions:
        print("  [mAP] 沒有預測結果")
        return 0.

    print(f"  [mAP] 預測數量={len(all_predictions)}")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(val_json)
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False) as f:
            json.dump(all_predictions, f)
        coco_eval = COCOeval(coco_gt, coco_gt.loadRes(f.name), "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0])


# ============================================================
# 推理：生成 pred.json
# ============================================================

@torch.no_grad()
def run_inference(model, cfg, device, conf_thr=0.05):
    """對 test set 推理並生成 pred.json"""
    model.eval()
    amp_en, amp_dt, _ = setup_amp(device, cfg["mixed_precision"])

    test_ds = TestImageDataset(
        Path(cfg["data_dir"]) / "test",
        image_size=cfg["image_size"])
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=lambda batch: (
            torch.stack([x[0] for x in batch]),
            [x[1] for x in batch]),
    )

    all_results = []
    for images, metas in tqdm(
            test_loader, desc="Inference", dynamic_ncols=True):
        images = images.to(device)
        with autocast(device_type="cuda", dtype=amp_dt, enabled=amp_en):
            preds = model(images)

        for i, m in enumerate(metas):
            probs = preds["pred_logits"][i].float().softmax(-1)
            scores, cids = probs[:, :-1].max(-1)
            keep = scores > conf_thr
            if not keep.any():
                continue

            sc, cid = scores[keep], cids[keep]
            boxes_xyxy = box_convert(
                preds["pred_boxes"][i][keep].float() * cfg["image_size"],
                "cxcywh", "xyxy")

            orig_h, orig_w = m["orig_size"]
            px, py, ratio = m["pad_x"], m["pad_y"], m["scale"]
            boxes_xyxy[:, [0, 2]] = (
                (boxes_xyxy[:, [0, 2]] - px) / ratio).clamp(0, orig_w)
            boxes_xyxy[:, [1, 3]] = (
                (boxes_xyxy[:, [1, 3]] - py) / ratio).clamp(0, orig_h)

            boxes_xyxy, sc, cid = per_class_nms(
                boxes_xyxy, sc, cid, cfg["nms_threshold"])
            if len(sc) > cfg["max_predictions"]:
                topk = sc.topk(cfg["max_predictions"])
                boxes_xyxy, cid, sc = (boxes_xyxy[topk.indices],
                                       cid[topk.indices], topk.values)

            boxes_xywh = box_convert(boxes_xyxy, "xyxy", "xywh")
            img_id = m["image_id"]

            for s, c, b in zip(sc.tolist(), cid.tolist(), boxes_xywh.tolist()):
                all_results.append({
                    "image_id": img_id,
                    "category_id": int(c) + 1,
                    "bbox": [round(v, 2) for v in b],
                    "score": round(float(s), 4),
                })

    output_path = "/content/pred.json"
    with open(output_path, "w") as fp:
        json.dump(all_results, fp)
    print(f"儲存 {len(all_results)} 筆預測結果 → {output_path}")
    print("請將 pred.json 壓成 zip 後上傳至 CodaBench！")


# ============================================================
# 主程式
# ============================================================

cfg = CFG
data_dir = Path(cfg["data_dir"])
save_dir = Path(cfg["save_dir"])
val_json = str(data_dir / "valid.json")

amp_en, amp_dt, use_scaler = setup_amp(DEVICE, cfg["mixed_precision"])
scaler = GradScaler("cuda", enabled=use_scaler) if use_scaler else None

# ── Datasets & DataLoaders ──
train_set = DigitDetectionDataset(
    data_dir / "train", data_dir / "train.json",
    image_size=cfg["image_size"], is_training=True)
val_set = DigitDetectionDataset(
    data_dir / "valid", data_dir / "valid.json",
    image_size=cfg["image_size"], is_training=False)

train_loader = DataLoader(
    train_set,
    batch_size=cfg["batch_size"],
    shuffle=True,
    num_workers=cfg["num_workers"],
    collate_fn=detection_collate,
    pin_memory=True)
val_loader = DataLoader(
    val_set,
    batch_size=cfg["batch_size"] * 2,
    shuffle=False,
    num_workers=cfg["num_workers"],
    collate_fn=detection_collate,
    pin_memory=True)

n_train, n_val = len(train_set), len(val_set)
print(f"Training samples: {n_train} | Validation samples: {n_val}")

# ── Model ──
model = DeformableDETR(cfg).to(DEVICE)

# ── Optimizer（backbone 用較小的 lr）──
backbone_param_ids = set(
    id(p) for name,
    p in model.named_parameters() if any(
        name.startswith(s) for s in (
            "stage1",
            "stage2",
            "stage3",
            "stage4")))

backbone_params = [
    p for p in model.parameters() if id(p) in backbone_param_ids]
other_params = [
    p for p in model.parameters() if id(p) not in backbone_param_ids]
optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": cfg["backbone_lr"]},
    {"params": other_params, "lr": cfg["peak_lr"]},
], weight_decay=cfg["weight_decay"])

scheduler = build_lr_scheduler(
    optimizer,
    cfg["warmup_epochs"],
    cfg["total_epochs"])

# ── Loss ──
matcher = BipartiteMatcher(cfg["w_cls"], cfg["w_l1"], cfg["w_giou"])
loss_fn = DetectionLoss(cfg, matcher)

# ── Resume from checkpoint ──
start_epoch, best_map = 0, -1.
last_ckpt = save_dir / "last.pth"
best_ckpt = save_dir / "best.pth"

if last_ckpt.exists():
    ck = torch.load(last_ckpt, map_location=DEVICE)
    model.load_state_dict(ck["model_state"])
    optimizer.load_state_dict(ck["optimizer_state"])
    start_epoch = ck["epoch"] + 1
    best_map = ck.get("best_map", -1.)
    # 重建 scheduler 並 fast-forward 到正確 epoch
    scheduler = build_lr_scheduler(
        optimizer,
        cfg["warmup_epochs"],
        cfg["total_epochs"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(start_epoch):
            scheduler.step()
    if scaler and "scaler_state" in ck:
        scaler.load_state_dict(ck["scaler_state"])
    current_lr = optimizer.param_groups[1]["lr"]
    msg = f"Resumed from epoch {start_epoch}"
    msg += f" | best mAP={best_map:.4f} | lr={current_lr:.2e}"
    print(msg)
else:
    print("從頭開始訓練。")

# ── 訓練迴圈 ──
for epoch in range(start_epoch, cfg["total_epochs"]):
    t_start = time.time()

    train_loss = train_one_epoch(
        model, loss_fn, optimizer, train_loader,
        DEVICE, epoch, scaler, amp_en, amp_dt, cfg["gradient_clip"])
    scheduler.step()

    do_eval = (
        epoch %
        cfg["eval_every"] == 0) or (
        epoch == cfg["total_epochs"] -
        1)
    val_map = 0.
    if do_eval:
        val_map = compute_val_map(
            model,
            val_loader,
            DEVICE,
            val_json,
            cfg["image_size"],
            amp_en,
            amp_dt,
            cfg["confidence_thr"],
            cfg["nms_threshold"],
            cfg["max_predictions"])

    elapsed = time.time() - t_start
    lr_now = optimizer.param_groups[1]["lr"]
    map_str = f" | mAP={val_map:.4f}" if do_eval else ""
    print(f"Epoch {epoch:3d}/{cfg['total_epochs']} | "
          f"loss={train_loss:.4f}{map_str} | lr={lr_now:.2e} | {elapsed:.1f}s")

    # 每個 epoch 都存 last.pth（不怕斷線）
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_map": best_map,
    }
    if scaler:
        checkpoint["scaler_state"] = scaler.state_dict()
    torch.save(checkpoint, last_ckpt)

    # 有改善才存 best.pth
    if do_eval and val_map > best_map:
        best_map = val_map
        checkpoint["best_map"] = best_map
        torch.save(checkpoint, best_ckpt)
        print(f"  ✓ 最佳模型已儲存！(mAP={val_map:.4f})")

print(f"\n訓練完成。最佳 mAP = {best_map:.4f}")

# ── 生成 pred.json ──
print("\n載入最佳模型進行推理...")
best_ck = torch.load(best_ckpt, map_location=DEVICE)
model.load_state_dict(best_ck["model_state"])
run_inference(model, cfg, DEVICE, conf_thr=cfg["confidence_thr"])
