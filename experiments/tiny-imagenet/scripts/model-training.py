import os
import json
import time
import random
import copy
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as tvm

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Setup GPU optimizations for RTX 4060 / Ampere
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark     = True   # fastest conv algo for fixed input sizes
    torch.backends.cudnn.deterministic = False
    # TF32: ~10% free throughput gain on Ampere — no other code changes needed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    torch.set_float32_matmul_precision('high')
    print(f'TF32   : matmul={torch.backends.cuda.matmul.allow_tf32}  '
          f'cudnn={torch.backends.cudnn.allow_tf32}')

# Configure dataset and output paths
BASE_DIR    = Path('.')
DATASET_DIR = Path("/raid/home/dgxuser15/datasets/tiny-imagenet-200")
MODELS_DIR  = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_DIR  = DATASET_DIR / 'train'
VAL_DIR    = DATASET_DIR / 'val'
VAL_ANNOT  = VAL_DIR / 'val_annotations.txt'
WNIDS_FILE = DATASET_DIR / 'wnids.txt'
STATS_FILE = RESULTS_DIR / 'dataset_stats.json'

# Load normalization stats from dataset statistics or use fallbacks
if STATS_FILE.exists():
    with open(STATS_FILE) as _f:
        _s = json.load(_f)
    MEAN = tuple(_s['rgb_mean'])
    STD  = tuple(_s['rgb_std'])
else:
    MEAN = (0.4802, 0.4481, 0.3975)
    STD  = (0.2770, 0.2691, 0.2821)
    warnings.warn('dataset_stats.json not found — using published reference values.')

# Create class map and load WordNet IDs
with open(WNIDS_FILE) as _f:
    _wnids = [l.strip() for l in _f if l.strip()]
CLASS_MAP   = {wnid: idx for idx, wnid in enumerate(sorted(_wnids))}
NUM_CLASSES = len(CLASS_MAP)

print(f'Classes : {NUM_CLASSES}  |  mean={MEAN}  |  std={STD}')

# Define data augmentation and transformation pipelines
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


# Define PyTorch dataset classes for train and validation data
class TinyImageNetTrain(Dataset):
    def __init__(self):
        self.samples: list[tuple[Path, int]] = []
        for wnid, label in CLASS_MAP.items():
            img_dir = TRAIN_DIR / wnid / 'images'
            if img_dir.exists():
                for p in img_dir.glob('*.JPEG'):
                    self.samples.append((p, label))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return TRAIN_TRANSFORM(Image.open(path).convert('RGB')), label


class TinyImageNetVal(Dataset):
    def __init__(self):
        self.samples: list[tuple[Path, int]] = []
        img_dir = VAL_DIR / 'images'
        with open(VAL_ANNOT) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[1] in CLASS_MAP:
                    self.samples.append((img_dir / parts[0], CLASS_MAP[parts[1]]))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Skipping corrupted image: {path}")
            return self.__getitem__((idx + 1) % len(self.samples))

        return VAL_TRANSFORM(img), label


# Training configuration parameters
CFG = dict(
    batch_size      = 256, 
    epochs          = 100,
    lr              = 3e-3,
    weight_decay    = 5e-4,
    label_smoothing = 0.1,
    mixup_alpha     = 0.2,
    cutmix_alpha    = 1.0,
    cutmix_prob     = 0.5,     # 50% chance CutMix vs Mixup each batch
    patience        = 25,
    warmup_epochs   = 5,
    num_workers     = 8,
    pin_memory      = True,
    grad_clip       = 5.0,
)

DRY_RUN         = False
DRY_EPOCHS      = 2
DRY_BATCHES     = 10
DRY_VAL_BATCHES = 5

MODELS_TO_TRAIN = ['mobilenetv2', 'shufflenetv2', 'efficientnet_b0']

print('Config:', json.dumps(CFG, indent=2))
print(f'DRY_RUN = {DRY_RUN}')

# Initialize PyTorch DataLoaders
train_loader = DataLoader(
    TinyImageNetTrain(),
    batch_size=CFG['batch_size'],
    shuffle=True,
    num_workers=CFG['num_workers'],
    pin_memory=CFG['pin_memory'],
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=True,
)
val_loader = DataLoader(
    TinyImageNetVal(),
    batch_size=CFG['batch_size'],   # FIX: consistent batch size (was 256 in eval loop)
    shuffle=False,
    num_workers=CFG['num_workers'],
    pin_memory=CFG['pin_memory'],
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=False,
)

print(f'Train batches : {len(train_loader)} | Val batches : {len(val_loader)}')


# Define function to build model architecture
def build_model(name: str, compile_model: bool = False) -> nn.Module:
    if name == 'mobilenetv2':
        model = tvm.mobilenet_v2(weights=None, num_classes=NUM_CLASSES)
        model.features[0][0].stride = (1, 1)
    elif name == 'shufflenetv2':
        model = tvm.shufflenet_v2_x1_0(weights=None, num_classes=NUM_CLASSES)
        model.conv1[0].stride = (1, 1)
    elif name == 'efficientnet_b0':
        model = tvm.efficientnet_b0(weights=None, num_classes=NUM_CLASSES)
        model.features[0][0].stride = (1, 1)
    else:
        raise ValueError(f'Unknown model: {name}')

    # channels_last memory layout: ~5-15% throughput gain for CNN models on CUDA
    # Tensors are stored as NHWC instead of NCHW — better fits GPU memory access patterns
    model = model.to(memory_format=torch.channels_last)
    model = model.to(DEVICE)

    # torch.compile: ~15-20% speedup via kernel fusion (Python 3.11+, PyTorch 2.0+)
    # Adds ~60s compilation overhead on first batch — worthwhile for full training runs
    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print(f'  [compile] torch.compile() applied to {name}')
        except Exception as e:
            print(f'  [compile] torch.compile() skipped: {e}')

    return model


USE_COMPILE = True #(not DRY_RUN) and hasattr(torch, 'compile')
print(f'build_model() ready | channels_last=True | torch.compile={USE_COMPILE}')


# Mixup and CutMix data augmentation utilities
def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
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
    """Randomly apply either Mixup or CutMix per batch."""
    index = torch.randperm(x.size(0), device=x.device)
    if np.random.random() < cutmix_prob and cutmix_alpha > 0:
        # CutMix
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        x1, y1, x2, y2 = rand_bbox(x.size(), lam)
        x_mixed = x.clone()
        x_mixed[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(2) * x.size(3)))
    else:
        # Mixup
        lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
        x_mixed = lam * x + (1 - lam) * x[index]
    return x_mixed, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Measure model inference latency
@torch.no_grad()
def measure_latency(model: nn.Module, n_runs: int = 100) -> float:
    """Median inference latency (ms) at batch=1."""
    model.eval()
    # Use channels_last for the dummy input to match model format
    dummy = torch.randn(1, 3, 64, 64, device=DEVICE).to(
        memory_format=torch.channels_last)
    latencies = []
    for _ in range(10):        # warm-up
        _ = model(dummy)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model(dummy)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
    return float(np.median(latencies))


# Evaluate model on the validation dataset
@torch.no_grad()
def eval_epoch(model: nn.Module, loader, criterion) -> tuple[float, float, float]:
    model.eval()
    total_loss, correct1, correct5, total = 0., 0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        if DRY_RUN and batch_idx >= DRY_VAL_BATCHES:
            break
        imgs   = imgs.to(DEVICE, non_blocking=True,
                         memory_format=torch.channels_last)
        labels = labels.to(DEVICE, non_blocking=True)  
        with autocast(device_type=DEVICE.type):
            out  = model(imgs)
            loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
        correct1   += (out.argmax(dim=1) == labels).sum().item()
        _, top5     = out.topk(5, dim=1)
        correct5   += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
    return total_loss / total, correct1 / total, correct5 / total


# Main training loop for all specified models
all_metrics = []

for model_name in MODELS_TO_TRAIN:
    # Skip if checkpoint and metrics already exist (resume-friendly after partial failure)
    ckpt_path = MODELS_DIR / f'{model_name}_best.pth'
    json_path = RESULTS_DIR / f'baseline_{model_name}_metrics.json'
    if ckpt_path.exists() and json_path.exists():
        with open(json_path, 'r') as f:
            metrics = json.load(f)
        print(f"\n  ⏭  {model_name.upper()} — checkpoint exists "
              f"(acc={metrics.get('best_val_acc1', 0)}%). Skipping.")
        all_metrics.append(metrics)
        continue

    print(f"\n{'='*60}\n  Training : {model_name.upper()}  |  Device: {DEVICE}\n{'='*60}")

    model    = build_model(model_name, compile_model=USE_COMPILE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parameters : {n_params / 1e6:.2f} M')

    criterion = nn.CrossEntropyLoss(label_smoothing=CFG['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(),
                             lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    n_epochs      = DRY_EPOCHS if DRY_RUN else CFG['epochs']
    warmup_epochs = CFG['warmup_epochs']
    # Cosine schedule after warmup
    cosine_sched  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, n_epochs - warmup_epochs), eta_min=1e-6)
    warmup_sched  = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    scheduler     = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs])
    scaler    = GradScaler(device=DEVICE.type)   # AMP fp16 scaler

    best_acc1  = 0.
    patience_c = 0
    history    = {'train_loss': [], 'val_loss': [], 'val_acc1': [], 'val_acc5': [], 'lr': []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0, running_loss, total = time.time(), 0., 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            if DRY_RUN and batch_idx >= DRY_BATCHES:
                break

            # channels_last layout + non_blocking transfer for maximum GPU utilization
            imgs   = imgs.to(DEVICE, non_blocking=True,
                             memory_format=torch.channels_last)
            labels = labels.to(DEVICE, non_blocking=True)

            imgs_m, y_a, y_b, lam = mixup_cutmix_data(
                imgs, labels,
                mixup_alpha=CFG['mixup_alpha'],
                cutmix_alpha=CFG['cutmix_alpha'],
                cutmix_prob=CFG['cutmix_prob'],
            )

            optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()
            with autocast(device_type=DEVICE.type):
                out  = model(imgs_m)
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            total        += imgs.size(0)

        scheduler.step()
        train_loss = running_loss / total
        val_loss, val_acc1, val_acc5 = eval_epoch(model, val_loader, criterion)
        lr_now = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc1'].append(val_acc1)
        history['val_acc5'].append(val_acc5)
        history['lr'].append(lr_now)

        print(f"Epoch [{epoch:02d}/{n_epochs}]  "
              f"TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  "
              f"Acc@1={val_acc1*100:.2f}%  Acc@5={val_acc5*100:.2f}%  "
              f"lr={lr_now:.5f}  t={time.time()-t0:.1f}s")

        if val_acc1 > best_acc1:
            best_acc1  = val_acc1
            patience_c = 0
            ckpt_path  = MODELS_DIR / f'{model_name}_best.pth'
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_acc1': val_acc1, 'val_acc5': val_acc5}, ckpt_path)
        else:
            patience_c += 1
            if patience_c >= CFG['patience']:
                print(f'  Early stopping at epoch {epoch}')
                break

    # Save per-model metrics and performance stats
    if DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(DEVICE)
    latency_ms = measure_latency(model)
    peak_mem   = (torch.cuda.max_memory_allocated(DEVICE) / 1024**2
                  if DEVICE.type == 'cuda' else -1.)
    ckpt_path  = MODELS_DIR / f'{model_name}_best.pth'
    model_mb   = ckpt_path.stat().st_size / 1024**2 if ckpt_path.exists() else -1

    metrics = {
        'model':         model_name,
        'params_M':      round(n_params / 1e6, 3),
        'best_val_acc1': round(best_acc1 * 100, 2),
        'best_val_acc5': round(max(history['val_acc5']) * 100, 2),
        'latency_ms':    round(latency_ms, 2),
        'peak_mem_MB':   round(peak_mem, 1),
        'model_size_MB': round(model_mb, 1),
        'epochs_run':    len(history['train_loss']),
        'history':       history,
    }

    out_json = RESULTS_DIR / f'baseline_{model_name}_metrics.json'
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'  ✓  Metrics → {out_json}')
    all_metrics.append(metrics)

    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

# Print summary comparison table
print('\n✓  All models trained.')
print(f"  {'Model':<20} {'Acc@1':>8} {'Acc@5':>8} {'Params(M)':>10} "
      f"{'Lat(ms)':>9} {'Size(MB)':>9}")
print('─' * 80)
for m in all_metrics:
    print(f"  {m['model']:<20} {m['best_val_acc1']:>8.2f} {m['best_val_acc5']:>8.2f} "
          f"{m['params_M']:>10.2f} "
          f"{m['latency_ms']:>9.2f} {m['model_size_MB']:>9.1f}")
print('─' * 80)

# Plot training curves for evaluation metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['#3498DB', '#E74C3C', '#2ECC71']

for ax, key, title, ax_label in [
    (axes[0], 'val_acc1',   'Validation Accuracy (Top-1)', 'Accuracy'),
    (axes[1], 'val_loss',   'Validation Loss',             'Loss'),
    (axes[2], 'train_loss', 'Training Loss',               'Loss'),
]:
    for m, color in zip(all_metrics, colors):
        h = m['history'][key]
        ax.plot(range(1, len(h) + 1), h, color=color, lw=2, label=m['model'])
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ax_label, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'baseline_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved → baseline_training_curves.png')

# Generate comparison bar charts for accuracy, latency, and parameters
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
names  = [m['model'] for m in all_metrics]
clrs   = ['#3498DB', '#E74C3C', '#2ECC71'][:len(names)]

for ax, (key, ylabel) in zip(axes, [
    ('best_val_acc1', 'Top-1 Accuracy (%)'),
    ('latency_ms',    'Latency (ms, batch=1)'),
    ('params_M',      '# Parameters (M)'),
]):
    vals = [m[key] for m in all_metrics]
    bars = ax.bar(names, vals, color=clrs, edgecolor='white', linewidth=1.2)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

plt.suptitle('Baseline Model Comparison — Tiny-ImageNet-200',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'baseline_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved → baseline_comparison.png')