import warnings

warnings.filterwarnings("ignore")
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score
from functools import partial
import typing as t
from einops import rearrange

# ====================== 配置区域 ======================
CONFIG = {
    # 数据路径
    'data_root': r"D:\guangfu\shujuji\fenlei\shujuji_reorganized_final",

    # 类别定义
    'classes': ['crack', 'contact', 'interconnect', 'corrosion'],

    # 超参数
    'img_size': 224,
    'batch_size': 32,
    'epochs': 100,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'label_smoothing': 0.0,
    'grad_clip': 5.0,

    # 设备
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,

    # 输出目录
    'output_dir': './results_faster_vit_with_scsa_multiple'
}

# 创建输出目录
os.makedirs(CONFIG['output_dir'], exist_ok=True)
#print(f"📂 结果将保存至：{os.path.abspath(CONFIG['output_dir'])}")
#print(f"⚙️ 当前配置：Label Smoothing={CONFIG['label_smoothing']}, Grad Clip={CONFIG['grad_clip']}")


# ====================== SCSA 注意力模块 ======================
class SCSA(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(SCSA, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)
            else:
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The dim of x is (B, C, H, W)
        """
        # Spatial attention priority calculation
        b, c, h_, w_ = x.size()

        # Handle cases where dimensions don't match perfectly
        if c != self.dim:
            return x

        # (B, C, H)
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        # (B, C, W)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn

        # Channel attention based on self attention
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()

        # normalization first, then reshape -> (B, H, W, C) -> (B, C, H * W) and generate q, k and v
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # (B, head_num, head_dim, head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (B, head_num, head_dim, N)
        attn = attn @ v
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x


# ====================== Multiple 多尺度特征融合模块（修复版）======================
class Multiple(nn.Module):
    def __init__(self,
                 init_value=1e-6,
                 embed_dim=512,
                 predict_channels=1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(Multiple, self).__init__()

        # 为每个尺度的特征创建独立的gamma参数，维度匹配各自的通道数
        self.gamma1 = nn.Parameter(init_value * torch.ones((64)), requires_grad=True)
        self.gamma2 = nn.Parameter(init_value * torch.ones((128)), requires_grad=True)
        self.gamma3 = nn.Parameter(init_value * torch.ones((256)), requires_grad=True)
        self.gamma4 = nn.Parameter(init_value * torch.ones((512)), requires_grad=True)
        self.gamma5 = nn.Parameter(init_value * torch.ones((512)), requires_grad=True)
        self.gamma6 = nn.Parameter(init_value * torch.ones((512)), requires_grad=True)

        self.norm = norm_layer(embed_dim)

        # 将所有特征统一到embed_dim维度
        self.conv_layer1 = nn.Conv2d(in_channels=64, out_channels=embed_dim, kernel_size=1, stride=1, padding=0)
        self.conv_layer2 = nn.Conv2d(in_channels=128, out_channels=embed_dim, kernel_size=1, stride=1, padding=0)
        self.conv_layer3 = nn.Conv2d(in_channels=256, out_channels=embed_dim, kernel_size=1, stride=1, padding=0)
        self.conv_layer4 = nn.Conv2d(in_channels=512, out_channels=embed_dim, kernel_size=1, stride=1, padding=0)
        self.conv_layer5 = nn.Conv2d(in_channels=512, out_channels=embed_dim, kernel_size=1, stride=1, padding=0)
        self.conv_layer6 = nn.Conv2d(in_channels=512, out_channels=embed_dim, kernel_size=1, stride=1, padding=0)

        self.conv_last = nn.Conv2d(embed_dim, predict_channels, kernel_size=1)

    def forward(self, features):
        c1, c2, c3, c4, c5, c6 = features

        # 获取目标空间尺寸（使用c1的尺寸）
        b, _, h, w = c1.shape

        # 将所有特征图调整到相同的空间尺寸和通道数
        c1_proj = self.conv_layer1(c1)  # (B, 64, H, W) -> (B, embed_dim, H, W)

        c2_resized = F.interpolate(c2, size=(h, w), mode='bilinear', align_corners=False)
        c2_proj = self.conv_layer2(c2_resized)  # (B, 128, H, W) -> (B, embed_dim, H, W)

        c3_resized = F.interpolate(c3, size=(h, w), mode='bilinear', align_corners=False)
        c3_proj = self.conv_layer3(c3_resized)  # (B, 256, H, W) -> (B, embed_dim, H, W)

        c4_resized = F.interpolate(c4, size=(h, w), mode='bilinear', align_corners=False)
        c4_proj = self.conv_layer4(c4_resized)  # (B, 512, H, W) -> (B, embed_dim, H, W)

        c5_resized = F.interpolate(c5, size=(h, w), mode='bilinear', align_corners=False)
        c5_proj = self.conv_layer5(c5_resized)  # (B, 512, H, W) -> (B, embed_dim, H, W)

        c6_resized = F.interpolate(c6, size=(h, w), mode='bilinear', align_corners=False)
        c6_proj = self.conv_layer6(c6_resized)  # (B, 512, H, W) -> (B, embed_dim, H, W)

        # 展平操作：将特征图转换为序列格式
        c1_flat = c1_proj.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        c2_flat = c2_proj.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        c3_flat = c3_proj.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        c4_flat = c4_proj.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        c5_flat = c5_proj.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        c6_flat = c6_proj.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)

        # 为每个尺度计算权重（基于原始特征的全局信息）
        w1 = torch.sigmoid(self.gamma1).mean()  # 标量权重
        w2 = torch.sigmoid(self.gamma2).mean()
        w3 = torch.sigmoid(self.gamma3).mean()
        w4 = torch.sigmoid(self.gamma4).mean()
        w5 = torch.sigmoid(self.gamma5).mean()
        w6 = torch.sigmoid(self.gamma6).mean()

        # 加权融合
        x = w1 * c1_flat + w2 * c2_flat + w3 * c3_flat + w4 * c4_flat + w5 * c5_flat + w6 * c6_flat

        # 重塑回2D特征图
        x = x.transpose(1, 2).reshape(b, -1, h, w)  # (B, embed_dim, H, W)

        # LayerNorm和最终卷积
        x = (self.norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2).contiguous()
        x = self.conv_last(x)

        return x


# ====================== 1. 多标签数据集定义 ======================
class MultiLabelDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels_path = os.path.join(root_dir, 'labels.csv')

        if not os.path.exists(self.labels_path):
            parent_dir = os.path.dirname(root_dir)
            self.labels_path = os.path.join(parent_dir, 'labels.csv')

        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"❌ 找不到标签文件：{self.labels_path}")

        self.data_frame = pd.read_csv(self.labels_path)
        self.classes = CONFIG['classes']

        missing_cols = [c for c in self.classes if c not in self.data_frame.columns]
        if missing_cols:
            raise ValueError(f"❌ labels.csv 中缺少列：{missing_cols}")

        self.img_dir = os.path.join(root_dir, 'images')
        if not os.path.exists(self.img_dir):
            self.img_dir = root_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        img_name = row['filename']

        possible_paths = [
            os.path.join(self.img_dir, img_name),
            os.path.join(self.root_dir, img_name),
            img_name
        ]

        img_path = None
        for p in possible_paths:
            if os.path.exists(p):
                img_path = p
                break

        if not img_path or not os.path.exists(img_path):
            return self.__getitem__((idx + 1) % len(self.data_frame))

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            return self.__getitem__((idx + 1) % len(self.data_frame))

        label_vector = row[self.classes].values.astype(float)

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor(label_vector)


# ====================== 2. 改进的 FasterViT Block ======================
class FasterViTBlock(nn.Module):
    def __init__(self, dim, out_dim, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, stride=stride, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, out_dim, 1)
        self.norm2 = nn.BatchNorm2d(out_dim)

        self.downsample = nn.Sequential(
            nn.Conv2d(dim, out_dim, 1, stride=stride),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


# ====================== 3. 增强版 FasterViT (集成SCSA和Multiple) ======================
class EnhancedFasterViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Patch Embedding
        self.patch_embed = nn.Conv2d(3, 64, 4, stride=4, padding=0)

        # Stage 1
        self.stage1_block1 = FasterViTBlock(64, 64)
        self.stage1_block2 = FasterViTBlock(64, 64)
        self.scsa1 = SCSA(dim=64, head_num=4, window_size=7)

        # Stage 2
        self.stage2_block1 = FasterViTBlock(64, 128, stride=2)
        self.stage2_block2 = FasterViTBlock(128, 128)
        self.scsa2 = SCSA(dim=128, head_num=8, window_size=7)

        # Stage 3
        self.stage3_block1 = FasterViTBlock(128, 256, stride=2)
        self.stage3_block2 = FasterViTBlock(256, 256)
        self.scsa3 = SCSA(dim=256, head_num=16, window_size=7)

        # Stage 4
        self.stage4_block1 = FasterViTBlock(256, 512, stride=2)
        self.stage4_block2 = FasterViTBlock(512, 512)
        self.scsa4 = SCSA(dim=512, head_num=32, window_size=7)

        # Multiple多尺度特征融合模块
        self.multiple_fusion = Multiple(
            init_value=1e-6,
            embed_dim=512,
            predict_channels=num_classes,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        # 分类头 (保留作为备用)
        self.norm = nn.LayerNorm(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(512, num_classes)

        # 是否使用Multiple融合
        self.use_multiple_fusion = True

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 收集多尺度特征
        features = []

        # Patch Embedding
        x = self.patch_embed(x)

        # Stage 1
        x = self.stage1_block1(x)
        x = self.stage1_block2(x)
        x = self.scsa1(x)
        features.append(x)  # c1: 64 channels, size: 56x56 (for 224 input)

        # Stage 2
        x = self.stage2_block1(x)
        x = self.stage2_block2(x)
        x = self.scsa2(x)
        features.append(x)  # c2: 128 channels, size: 28x28

        # Stage 3
        x = self.stage3_block1(x)
        x = self.stage3_block2(x)
        x = self.scsa3(x)
        features.append(x)  # c3: 256 channels, size: 14x14

        # Stage 4
        x = self.stage4_block1(x)
        x = self.stage4_block2(x)
        x = self.scsa4(x)
        features.append(x)  # c4: 512 channels, size: 7x7

        # 生成c5和c6 (通过额外的下采样)
        c5 = F.avg_pool2d(x, kernel_size=2, stride=2)  # size: 4x4
        c6 = F.avg_pool2d(c5, kernel_size=2, stride=2)  # size: 2x2
        features.append(c5)
        features.append(c6)

        if self.use_multiple_fusion:
            # 使用Multiple模块进行多尺度特征融合
            fusion_output = self.multiple_fusion(features)
            # Global average pooling
            output = F.adaptive_avg_pool2d(fusion_output, (1, 1)).flatten(1)
            return output
        else:
            # 原始分类头
            x = self.avgpool(x).flatten(1)
            x = self.norm(x)
            return self.head(x)


# ====================== 4. 核心指标计算 ======================
def calculate_metrics(outputs, targets, class_names, threshold=0.5):
    probs = torch.sigmoid(outputs).detach().cpu().numpy()
    targets_np = targets.cpu().numpy()
    preds = (probs > threshold).astype(int)
    epsilon = 1e-7

    try:
        ap_per_class = average_precision_score(targets_np, probs, average=None)
        mAP = np.mean(ap_per_class)
    except:
        mAP = 0.0
        ap_per_class = [0.0] * len(class_names)

    matches = np.all(preds == targets_np, axis=1)
    overall_accuracy = np.mean(matches)

    precision_list, recall_list, f1_list = [], [], []
    true_positive_counts = np.sum(targets_np, axis=0)

    for i in range(len(class_names)):
        tp = np.sum(preds[:, i] * targets_np[:, i])
        fp = np.sum(preds[:, i] * (1 - targets_np[:, i]))
        fn = np.sum((1 - preds[:, i]) * targets_np[:, i])

        prec = tp / (tp + fp + epsilon)
        rec = tp / (tp + fn + epsilon)
        f1 = 2 * (prec * rec) / (prec + rec + epsilon)

        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

    return {
        'mAP': mAP,
        'overall_accuracy': overall_accuracy,
        'precision_avg': np.mean(precision_list),
        'recall_avg': np.mean(recall_list),
        'f1_avg': np.mean(f1_list),
        'preds': preds,
        'targets': targets_np,
        'probs': probs,
        'ap_per_class': ap_per_class,
        'true_counts': true_positive_counts
    }


# ====================== 5. 生成WPS表格与绘图 ======================
def save_results_to_wps(final_results, class_names, history, output_dir):
    print("\n💾 正在生成 WPS/Excel 表格...")

    flat_history = []
    for epoch, metrics in enumerate(history, 1):
        row = {'Epoch': epoch}
        for phase in ['train', 'val']:
            if phase in metrics:
                row[f'{phase}_loss'] = round(metrics[phase]['loss'], 4)
                row[f'{phase}_acc'] = round(metrics[phase]['acc'], 4)
                if phase == 'val':
                    row['val_mAP'] = round(metrics[phase]['mAP'], 4)
        flat_history.append(row)
    pd.DataFrame(flat_history).to_csv(os.path.join(output_dir, 'training_log.csv'), index=False, encoding='utf-8-sig')
    print(f"✅ 已生成：training_log.csv")

    report_data = []
    for phase in ['train', 'val', 'test']:
        if phase not in final_results: continue
        res = final_results[phase]
        row = {
            'Dataset': phase.upper(),
            'Overall_Accuracy': round(res['overall_accuracy'], 4),
            'mAP': round(res['mAP'], 4),
            'Precision_Avg': round(res['precision_avg'], 4),
            'Recall_Avg': round(res['recall_avg'], 4),
            'F1_Score_Avg': round(res['f1_avg'], 4)
        }
        for i, cls in enumerate(class_names):
            row[f'AP_{cls}'] = round(res['ap_per_class'][i], 4)
        report_data.append(row)
    pd.DataFrame(report_data).to_csv(os.path.join(output_dir, 'final_report.csv'), index=False, encoding='utf-8-sig')
    print(f"✅ 已生成：final_report.csv")

    if 'test' in final_results:
        targets = final_results['test']['targets']
        preds = final_results['test']['preds']
        true_counts = final_results['test']['true_counts']

        n_classes = len(class_names)
        matrix = np.zeros((n_classes, n_classes))

        for i in range(len(targets)):
            true_idx = np.where(targets[i] == 1)[0]
            pred_idx = np.where(preds[i] == 1)[0]
            for t in true_idx:
                for p in pred_idx:
                    matrix[t, p] += 1

        for i in range(n_classes):
            if true_counts[i] > 0:
                matrix[i, :] /= true_counts[i]

        df_cm = pd.DataFrame(matrix, index=[f'True_{c}' for c in class_names],
                             columns=[f'Pred_{c}' for c in class_names])
        descriptions = [f'当真实为 {cls} 时，预测为各列缺陷的概率 (分母={int(true_counts[i])})'
                        for i, cls in enumerate(class_names)]
        df_cm.insert(0, 'Description', descriptions)
        df_cm.to_csv(os.path.join(output_dir, 'confusion_probability.csv'), encoding='utf-8-sig')
        print(f"✅ 已生成：confusion_probability.csv")

        total_positive_labels = int(np.sum(true_counts))
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=class_names, yticklabels=class_names)

        plt.title(f'Enhanced FasterViT with SCSA & Multiple - 条件概率\n(基于测试集 {total_positive_labels:,} 个正例)',
                  pad=20, fontsize=12, fontweight='bold')
        plt.ylabel('真实标签 (True Label)', fontsize=11)
        plt.xlabel('预测标签 (Predicted Label)', fontsize=11)

        plt.figtext(0.5, 0.01,
                    f'注：集成了SCSA注意力和Multiple多尺度融合模块。',
                    ha='center', fontsize=9, style='italic',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(os.path.join(output_dir, 'confusion_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已生成：confusion_heatmap.png")


# ====================== 6. 数据加载 ======================
def get_transforms(is_training, img_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


def load_data(root, batch_size, num_workers):
    phases = ['train', 'val', 'test']
    datasets = {}
    loaders = {}
    sizes = {}

    for phase in phases:
        dir_path = os.path.join(root, phase)
        if not os.path.exists(dir_path):
            if phase == 'test':
                print(f"⚠️ 未找到测试集，将使用验证集作为测试集。")
                continue
            else:
                raise FileNotFoundError(f"缺少 {phase} 数据集路径：{dir_path}")

        dataset = MultiLabelDefectDataset(dir_path, transform=get_transforms(is_training=(phase == 'train')))
        datasets[phase] = dataset
        sizes[phase] = len(dataset)
        loaders[phase] = DataLoader(dataset, batch_size=batch_size, shuffle=(phase == 'train'),
                                    num_workers=num_workers, pin_memory=True, drop_last=(phase == 'train'))
        print(f"✅ {phase.capitalize()} 集加载完成：{sizes[phase]} 张")

    if 'test' not in loaders and 'val' in loaders:
        loaders['test'] = loaders['val']
        sizes['test'] = sizes['val']

    return loaders, sizes


# ====================== 7. 训练主循环 ======================
def train():
    device = torch.device(CONFIG['device'])
    print(f"🚀 开始训练增强版 FasterViT (SCSA + Multiple) (设备：{device})")
    print(f"   - 标签平滑：{CONFIG['label_smoothing']} (已关闭以修复 Loss 虚高)")
    print(f"   - 梯度裁剪：{CONFIG['grad_clip']} (已放宽)")

    loaders, sizes = load_data(CONFIG['data_root'], CONFIG['batch_size'], CONFIG['num_workers'])

    model = EnhancedFasterViT(num_classes=len(CONFIG['classes'])).to(device)
    print(f"🤖 模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)

    best_map = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    history = []

    print("\n" + "=" * 90)
    print(f"{'Epoch':^5} | {'Loss(T/V)':^12} | {'Acc(T/V)':^12} | {'Val_mAP':^8} | {'Status'}")
    print("=" * 90)

    for epoch in range(CONFIG['epochs']):
        metrics = {}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_outputs, all_targets = [], []

            loader = loaders[phase]
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels_used = labels

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_used)

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['grad_clip'])
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_outputs.append(outputs.cpu())
                all_targets.append(labels.cpu())

            epoch_loss = running_loss / sizes[phase]
            outs_cat = torch.cat(all_outputs, dim=0)
            tgts_cat = torch.cat(all_targets, dim=0)

            stats = calculate_metrics(outs_cat, tgts_cat, CONFIG['classes'])

            metrics[phase] = {
                'loss': epoch_loss,
                'acc': stats['overall_accuracy'],
                'mAP': stats['mAP']
            }

        scheduler.step()

        status = "-"
        if metrics['val']['mAP'] > best_map:
            best_map = metrics['val']['mAP']
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, os.path.join(CONFIG['output_dir'], 'best_model.pth'))
            status = "🏆 Best"

        log_line = (
            f"{epoch + 1:^5} | "
            f"{metrics['train']['loss']:.4f}/{metrics['val']['loss']:.4f} | "
            f"{metrics['train']['acc']:.4f}/{metrics['val']['acc']:.4f} | "
            f"{metrics['val']['mAP']:^8.4f} | "
            f"{status}"
        )
        print(log_line)
        history.append(metrics)

    print("=" * 90)
    print(f"✅ 训练完成！最佳 Val mAP: {best_map:.4f}")

    # ====================== 8. 最终测试与表格生成 ======================
    print("\n🔄 加载最佳模型进行最终测试...")
    model.load_state_dict(best_wts)
    model.eval()

    final_results = {}
    for phase in ['train', 'val', 'test']:
        if phase not in loaders: continue

        all_outputs, all_targets = [], []
        with torch.no_grad():
            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                all_outputs.append(outputs.cpu())
                all_targets.append(labels.cpu())

        outs_cat = torch.cat(all_outputs, dim=0)
        tgts_cat = torch.cat(all_targets, dim=0)
        final_results[phase] = calculate_metrics(outs_cat, tgts_cat, CONFIG['classes'])

    # 打印控制台报告
    print("\n📊 【最终性能报告 - 增强版 FasterViT with SCSA & Multiple】")
    print("-" * 90)
    print(f"{'Dataset':<10} | {'Overall Acc':^12} | {'mAP':^10} | {'Precision':^10} | {'Recall':^10}")
    print("-" * 90)

    for phase in ['train', 'val', 'test']:
        if phase not in final_results: continue
        res = final_results[phase]
        print(
            f"{phase.upper():<10} | {res['overall_accuracy']:^12.4f} | {res['mAP']:^10.4f} | {res['precision_avg']:^10.4f} | {res['recall_avg']:^10.4f}")
    print("-" * 90)

    save_results_to_wps(final_results, CONFIG['classes'], history, CONFIG['output_dir'])

    print(f"\n💾 所有结果已保存至：{CONFIG['output_dir']}")


if __name__ == "__main__":
    train()