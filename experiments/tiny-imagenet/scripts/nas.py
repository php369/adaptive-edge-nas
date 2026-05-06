# NAS Fine-Tuning — Standalone Model from Best Architecture

import os
import json
import time
import random
import pickle
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR      = Path.cwd()
PROCESSED_DIR = BASE_DIR / 'processed'
MODELS_DIR    = BASE_DIR / 'models'
RESULTS_DIR   = BASE_DIR / 'results'
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print(f'Device : {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
# Load manifest and best architecture configuration
manifest_path = PROCESSED_DIR / 'data_manifest.pkl'
assert manifest_path.exists(), 'Run data-preprocessing.ipynb first!'

with open(manifest_path, 'rb') as f:
    manifest = pickle.load(f)

train_samples = [(Path(p), lbl) for p, lbl in manifest['train']]
val_samples   = [(Path(p), lbl) for p, lbl in manifest['val']]
MEAN          = tuple(manifest['mean'])
STD           = tuple(manifest['std'])
NUM_CLASSES   = manifest['num_classes']

arch_path = RESULTS_DIR / 'best_arch.json'
assert arch_path.exists(), 'Run hardware-aware.ipynb first to generate best_arch.json!'

with open(arch_path) as f:
    best = json.load(f)
ARCH = best['arch']   # list of int op indices, length = NUM_CELLS

print(f'Train : {len(train_samples):,} | Val : {len(val_samples):,} | Classes : {NUM_CLASSES}')
print(f'Arch  : {ARCH}')
# Define transforms and datasets
TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(64, padding=8, padding_mode='reflect'),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    T.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0),
    T.Normalize(mean=MEAN, std=STD),
])

VAL_TRANSFORM = T.Compose([
    T.CenterCrop(56),
    T.Resize(64, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


class TinyImageNetTrain(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return TRAIN_TRANSFORM(Image.open(path).convert('RGB')), label


class TinyImageNetVal(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return VAL_TRANSFORM(Image.open(path).convert('RGB')), label


NUM_WORKERS = 8

train_loader = DataLoader(
    TinyImageNetTrain(train_samples),
    batch_size=256, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True, prefetch_factor=2, drop_last=True,
)
val_loader = DataLoader(
    TinyImageNetVal(val_samples),
    batch_size=128, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True, prefetch_factor=2,
)

print(f'Train batches : {len(train_loader)} | Val batches : {len(val_loader)}')
# Define search-space primitives required for StandaloneNASModel
OP_NAMES = [
    'identity', 'dwconv3x3', 'dwconv5x5',
    'mbconv3x3', 'mbconv5x5', 'shuffle_block', 'se_block',
]

CELL_CONFIG = [
    (32,  1), (32,  1), (64,  2),
    (64,  1), (64,  1), (128, 2),
    (128, 1), (128, 1), (128, 1),
    (128, 1), (192, 2), (192, 1),
    (192, 1), (192, 1), (256, 2),
    (256, 1), (256, 1), (256, 1),
    (256, 1), (256, 1),
]
STEM_CH   = 32
NUM_CELLS = len(CELL_CONFIG)


def _make_div8(v):
    return max(8, int(v + 4) // 8 * 8)


class DepthwiseSepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride,
                      padding=pad, groups=C_in, bias=False),
            nn.BatchNorm2d(C_in), nn.ReLU6(inplace=True),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out), nn.ReLU6(inplace=True),
        )

    def forward(self, x): return self.op(x)


class MBConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, expand_ratio=3):
        super().__init__()
        mid = _make_div8(C_in * expand_ratio)
        pad = kernel_size // 2
        self.use_res = (stride == 1) and (C_in == C_out)
        self.op = nn.Sequential(
            nn.Conv2d(C_in, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
            nn.Conv2d(mid, mid, kernel_size, stride=stride,
                      padding=pad, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
            nn.Conv2d(mid, C_out, 1, bias=False), nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        out = self.op(x)
        return x + out if self.use_res else out


class ShuffleBlock(nn.Module):
    def __init__(self, C_in, C_out, stride=1):
        super().__init__()
        branch = C_out // 2
        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(C_in, C_in, 3, stride=stride, padding=1, groups=C_in, bias=False),
                nn.BatchNorm2d(C_in),
                nn.Conv2d(C_in, branch, 1, bias=False), nn.BatchNorm2d(branch), nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(C_in, branch, 1, bias=False), nn.BatchNorm2d(branch), nn.ReLU(inplace=True),
                nn.Conv2d(branch, branch, 3, stride=stride, padding=1, groups=branch, bias=False),
                nn.BatchNorm2d(branch),
                nn.Conv2d(branch, branch, 1, bias=False), nn.BatchNorm2d(branch), nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Identity()
            half = C_in // 2
            self.branch2 = nn.Sequential(
                nn.Conv2d(half, branch, 1, bias=False), nn.BatchNorm2d(branch), nn.ReLU(inplace=True),
                nn.Conv2d(branch, branch, 3, padding=1, groups=branch, bias=False),
                nn.BatchNorm2d(branch),
                nn.Conv2d(branch, branch, 1, bias=False), nn.BatchNorm2d(branch), nn.ReLU(inplace=True),
            )
        self.stride = stride

    def _shuffle(self, x, groups=2):
        B, C, H, W = x.shape
        return x.view(B, groups, C // groups, H, W).transpose(1, 2).contiguous().view(B, C, H, W)

    def forward(self, x):
        if self.stride > 1:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([self.branch1(x1), self.branch2(x2)], dim=1)
        return self._shuffle(out)


class SEBlock(nn.Module):
    def __init__(self, C_in, C_out, stride=1, reduction=4):
        super().__init__()
        mid = max(8, C_in // reduction)
        self.use_res = (stride == 1) and (C_in == C_out)
        self.bn      = nn.BatchNorm2d(C_out)
        self.conv_dw = nn.Conv2d(C_in, C_in, 3, stride=stride, padding=1, groups=C_in, bias=False)
        self.conv_pw = nn.Conv2d(C_in, C_out, 1, bias=False)
        self.se      = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C_out, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, C_out), nn.Sigmoid(),
        )

    def forward(self, x):
        out = F.relu6(self.conv_dw(x))
        out = self.conv_pw(out)
        se  = self.se(out).view(out.size(0), out.size(1), 1, 1)
        out = out * se
        return (out + x) if self.use_res else self.bn(out)


class Identity(nn.Module):
    def __init__(self, C_in, C_out, stride=1):
        super().__init__()
        self.adapt = nn.Identity() if (C_in == C_out and stride == 1) else \
                     nn.Sequential(nn.Conv2d(C_in, C_out, 1, stride=stride, bias=False),
                                   nn.BatchNorm2d(C_out))

    def forward(self, x): return self.adapt(x)


def build_op(op_name: str, C_in: int, C_out: int, stride: int = 1) -> nn.Module:
    if op_name == 'identity':        return Identity(C_in, C_out, stride)
    elif op_name == 'dwconv3x3':     return DepthwiseSepConv(C_in, C_out, 3, stride)
    elif op_name == 'dwconv5x5':     return DepthwiseSepConv(C_in, C_out, 5, stride)
    elif op_name == 'mbconv3x3':     return MBConv(C_in, C_out, 3, stride)
    elif op_name == 'mbconv5x5':     return MBConv(C_in, C_out, 5, stride)
    elif op_name == 'shuffle_block': return ShuffleBlock(C_in, C_out, stride)
    elif op_name == 'se_block':      return SEBlock(C_in, C_out, stride)
    else: raise ValueError(f'Unknown op: {op_name}')


print('All primitive ops defined.')
# Build Standalone NAS model
class StandaloneNASModel(nn.Module):
    """
    Instantiate the best-found architecture as a single static net.
    Much leaner than SuperNet (no parallel ops, no op-switching overhead).
    """
    def __init__(self, arch: list, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, STEM_CH, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(STEM_CH),
            nn.ReLU6(inplace=True),
        )
        self.cells = nn.Sequential()
        C_in = STEM_CH
        for cell_idx, (op_idx, (C_out, stride)) in enumerate(zip(arch, CELL_CONFIG)):
            op = build_op(OP_NAMES[op_idx], C_in, C_out, stride)
            self.cells.add_module(f'cell_{cell_idx:02d}', op)
            C_in = C_out

        C_final = CELL_CONFIG[-1][0]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(C_final, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.cells(self.stem(x)))


model = StandaloneNASModel(ARCH).to(DEVICE)
model = model.to(memory_format=torch.channels_last)

# torch.compile for faster training
if hasattr(torch, 'compile'):
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print('[compile] torch.compile() applied to NAS model')
    except Exception as e:
        print(f'[compile] torch.compile() skipped: {e}')

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'StandaloneNASModel built — {n_params / 1e6:.2f} M parameters')
print(f'Ops: {[OP_NAMES[i] for i in ARCH]}')
# Set fine-tuning configuration parameters
FT_CFG = dict(
    epochs        = 100,
    lr            = 3e-3,
    weight_decay  = 5e-4,
    label_smooth  = 0.10,
    mixup_alpha   = 0.2,
    cutmix_alpha  = 1.0,
    cutmix_prob   = 0.5,
    patience      = 25,
    warmup_epochs = 5,
    grad_clip     = 5.0,
)

# Set DRY_RUN = False for full fine-tuning
DRY_RUN         = False
DRY_EPOCHS      = 2
DRY_BATCHES     = 10
DRY_VAL_BATCHES = 5

print('FT_CFG:', json.dumps(FT_CFG, indent=2))
print(f'DRY_RUN = {DRY_RUN}')


# Mixup and CutMix utility functions
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mixup_cutmix_data(x, y, mixup_alpha=0.2, cutmix_alpha=1.0, cutmix_prob=0.5):
    index = torch.randperm(x.size(0), device=x.device)
    if np.random.random() < cutmix_prob and cutmix_alpha > 0:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        x1, y1, x2, y2 = rand_bbox(x.size(), lam)
        x_mixed = x.clone()
        x_mixed[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(2) * x.size(3)))
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
        x_mixed = lam * x + (1 - lam) * x[index]
    return x_mixed, y, y[index], lam
# Start the fine-tuning training loop
n_epochs      = DRY_EPOCHS if DRY_RUN else FT_CFG['epochs']
warmup_epochs = FT_CFG['warmup_epochs']
criterion = nn.CrossEntropyLoss(label_smoothing=FT_CFG['label_smooth'])
optimizer = optim.AdamW(model.parameters(),
                         lr=FT_CFG['lr'], weight_decay=FT_CFG['weight_decay'])
cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(1, n_epochs - warmup_epochs), eta_min=1e-6)
warmup_sched = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
scheduler    = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_sched, cosine_sched],
    milestones=[warmup_epochs])
scaler    = GradScaler(device=DEVICE.type)

best_acc1  = 0.
patience_c = 0
history    = {'train_loss': [], 'val_acc1': [], 'val_acc5': []}

for epoch in range(1, n_epochs + 1):
    model.train()
    t0, run_loss, total = time.time(), 0., 0

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        if DRY_RUN and batch_idx >= DRY_BATCHES:
            break
        imgs   = imgs.to(DEVICE, non_blocking=True,
                         memory_format=torch.channels_last)
        labels = labels.to(DEVICE, non_blocking=True)

        # CutMix / Mixup
        imgs_m, y_a, y_b, lam = mixup_cutmix_data(
            imgs, labels,
            mixup_alpha=FT_CFG['mixup_alpha'],
            cutmix_alpha=FT_CFG['cutmix_alpha'],
            cutmix_prob=FT_CFG['cutmix_prob'],
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=DEVICE.type):
            out  = model(imgs_m)
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), FT_CFG['grad_clip'])
        scaler.step(optimizer)
        scaler.update()

        run_loss += loss.item() * imgs.size(0)
        total    += imgs.size(0)

    scheduler.step()

    # Validation
    model.eval()
    correct1, correct5, n_val = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(val_loader):
            if DRY_RUN and batch_idx >= DRY_VAL_BATCHES:
                break
            imgs   = imgs.to(DEVICE, non_blocking=True,
                             memory_format=torch.channels_last)
            labels = labels.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type):
                out = model(imgs)
            correct1 += (out.argmax(1) == labels).sum().item()
            _, top5   = out.topk(5, dim=1)
            correct5 += sum(labels[i].item() in top5[i].tolist()
                            for i in range(labels.size(0)))
            n_val    += labels.size(0)

    val_acc1 = correct1 / n_val
    val_acc5 = correct5 / n_val

    history['train_loss'].append(run_loss / total)
    history['val_acc1'].append(val_acc1)
    history['val_acc5'].append(val_acc5)

    print(f'  Epoch [{epoch:03d}/{n_epochs}]  '
          f'Loss={run_loss/total:.4f}  '
          f'ValAcc@1={val_acc1*100:.2f}%  ValAcc@5={val_acc5*100:.2f}%  '
          f't={time.time()-t0:.1f}s')

    if val_acc1 > best_acc1:
        best_acc1  = val_acc1
        patience_c = 0
        ckpt_path  = MODELS_DIR / 'nas_best_finetuned.pth'
        torch.save({
            'arch':        ARCH,
            'epoch':       epoch,
            'model_state': model.state_dict(),
            'val_acc1':    val_acc1,
            'val_acc5':    val_acc5,
        }, ckpt_path)
    else:
        patience_c += 1
        if patience_c >= FT_CFG['patience']:
            print(f'  Early stopping at epoch {epoch}')
            break

print(f'\nBest val Acc@1 : {best_acc1*100:.2f}%')
print(f'Saved → {MODELS_DIR / "nas_best_finetuned.pth"}')
# Plot the training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history['train_loss'], color='#3498DB', lw=2)
axes[0].set_title('Training Loss', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].grid(alpha=0.3)

axes[1].plot([v * 100 for v in history['val_acc1']], color='#E74C3C', lw=2)
axes[1].set_title('Validation Acc@1', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)'); axes[1].grid(alpha=0.3)

axes[2].plot([v * 100 for v in history['val_acc5']], color='#2ECC71', lw=2)
axes[2].set_title('Validation Acc@5', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Accuracy (%)'); axes[2].grid(alpha=0.3)

plt.suptitle('NAS Fine-Tune Training Curves', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'nas_finetuning_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved → nas_finetuning_curves.png')
# Perform latency benchmark for the fine-tuned model
model.eval()
dummy = torch.randn(1, 3, 64, 64, device=DEVICE)
lats  = []

with torch.no_grad():
    for _ in range(10):   # warm-up
        _ = model(dummy)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model(dummy)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)

lat_ms = float(np.median(lats))

# Save final summary
final_summary = {
    'arch':         ARCH,
    'op_names':     [OP_NAMES[i] for i in ARCH],
    'params_M':     round(n_params / 1e6, 3),
    'best_acc1':    round(best_acc1 * 100, 2),
    'latency_ms':   round(lat_ms, 2),
    'epochs_run':   len(history['train_loss']),
}
with open(RESULTS_DIR / 'nas_final_summary.json', 'w') as f:
    json.dump(final_summary, f, indent=2)

print('\n── NAS Final Summary ──────────────────────────────────────')
print(json.dumps(final_summary, indent=2))
print('Saved → nas_final_summary.json')