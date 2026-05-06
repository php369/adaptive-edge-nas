"""
Phase 1: Dataset Understanding & Exploratory Data Analysis (EDA)
Project: Hardware-Aware NAS for Edge Devices
Dataset: Tiny-ImageNet-200
"""

import json
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")

# Project configuration and directory setup
BASE_DIR    = Path(__file__).parent.parent   # project root (one level above scripts/)
DATASET_DIR = Path("/raid/home/dgxuser15/datasets/tiny-imagenet-200")
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_DIR  = DATASET_DIR / "train"
VAL_DIR    = DATASET_DIR / "val"
TEST_DIR   = DATASET_DIR / "test"
WNIDS_FILE = DATASET_DIR / "wnids.txt"
WORDS_FILE = DATASET_DIR / "words.txt"
VAL_ANNOT  = VAL_DIR / "val_annotations.txt"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Load WordNet IDs and metadata
if __name__ == "__main__":
    word_map = {}
    with open(WORDS_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                word_map[parts[0]] = parts[1]

    with open(WNIDS_FILE) as f:
        wnids = [l.strip() for l in f if l.strip()]

    # Collect image paths for training and validation sets
    train_data: dict[str, list[Path]] = {}
    for cls_dir in sorted(TRAIN_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        img_dir = cls_dir / "images"
        train_data[cls_dir.name] = sorted(img_dir.glob("*.JPEG")) if img_dir.exists() else []

    val_data: dict[str, list[Path]] = {}
    img_dir_val = VAL_DIR / "images"
    with open(VAL_ANNOT) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                val_data.setdefault(parts[1], []).append(img_dir_val / parts[0])

    test_imgs = list((TEST_DIR / "images").glob("*.JPEG"))

    # Step 1: Perform integrity check to ensure dataset is complete
    stats = {}

    train_classes = list(train_data.keys())
    counts_list   = [len(v) for v in train_data.values()]
    stats.update({
        "n_train_classes":          len(train_classes),
        "train_img_per_class_min":  min(counts_list),
        "train_img_per_class_max":  max(counts_list),
        "train_img_per_class_mean": round(np.mean(counts_list), 2),
        "train_total_images":       sum(counts_list),
    })
    print(f"Train: {sum(counts_list):,} images / {len(train_classes)} classes "
          f"(min={min(counts_list)}, max={max(counts_list)})")

    missing_boxes = [c for c in train_classes if not (TRAIN_DIR / c / f"{c}_boxes.txt").exists()]
    stats["missing_box_files"] = len(missing_boxes)

    val_classes  = set(val_data.keys())
    expected_set = set(wnids)
    val_counts   = [len(v) for v in val_data.values()]
    stats.update({
        "val_classes_found":      len(val_classes),
        "val_label_extra":        len(val_classes - expected_set),
        "val_label_missing":      len(expected_set - val_classes),
        "val_total_images":       sum(len(v) for v in val_data.values()),
        "val_img_per_class_min":  min(val_counts) if val_counts else 0,
        "val_img_per_class_max":  max(val_counts) if val_counts else 0,
        "val_img_per_class_mean": round(np.mean(val_counts), 2) if val_counts else 0,
        "test_total_images":      len(test_imgs),
    })
    print(f"Val: {stats['val_total_images']:,} images | Test: {len(test_imgs):,} images (unlabelled)")

    # Check for corrupt images in a sample set
    all_train_paths = [p for paths in train_data.values() for p in paths]
    sample_check    = random.sample(all_train_paths, min(2000, len(all_train_paths)))
    corrupt = []
    for p in sample_check:
        try:
            with Image.open(p) as img:
                img.verify()
        except Exception:
            corrupt.append(p)
    stats["corrupt_image_count_sampled"] = len(corrupt)
    stats["corrupt_sample_size"]         = len(sample_check)
    print(f"Corruption check ({len(sample_check):,}-image sample): "
          f"{'0 corrupt ✓' if not corrupt else f'{len(corrupt)} corrupt'}")

    # Step 2A: Sample image size distribution
    size_sample = random.sample(all_train_paths, min(1000, len(all_train_paths)))
    sizes = Counter()
    for p in size_sample:
        try:
            with Image.open(p) as img:
                sizes[img.size] += 1
        except Exception:
            pass

    # Step 2B: Generate class distribution chart
    wnids_sorted = sorted(train_data.keys(), key=lambda k: -len(train_data[k]))
    bar_counts   = [len(train_data[w]) for w in wnids_sorted]

    fig, ax = plt.subplots(figsize=(24, 8))
    ax.bar(range(len(bar_counts)), bar_counts,
           color=plt.cm.viridis(np.linspace(0.15, 0.9, len(bar_counts))),
           edgecolor="none", width=1.0)
    ax.axhline(500, color="#FF4444", lw=1.5, ls="--", label="Expected (500)")
    ax.set_xlabel("Class (sorted by count)", fontsize=13)
    ax.set_ylabel("Training Images", fontsize=13)
    ax.set_title("Tiny-ImageNet-200 — Class Distribution (Training Set)", fontsize=15, fontweight="bold")
    ax.set_xticks([])
    ax.set_xlim(-1, len(bar_counts))
    ax.legend(fontsize=11)
    ax.annotate(f"max={max(bar_counts)}", xy=(0, max(bar_counts)),
                xytext=(10, max(bar_counts)+3), fontsize=9, color="navy")
    ax.annotate(f"min={min(bar_counts)}", xy=(len(bar_counts)-1, min(bar_counts)),
                xytext=(len(bar_counts)-50, min(bar_counts)+3), fontsize=9, color="darkred")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Step 2C: Calculate pixel intensity histogram and RGB stats
    print("Computing RGB statistics (5 000-image sample) …")

    rgb_sample = random.sample(all_train_paths, min(5000, len(all_train_paths)))
    r_vals, g_vals, b_vals = [], [], []
    for p in rgb_sample:
        try:
            img = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            r_vals.append(img[:, :, 0].ravel())
            g_vals.append(img[:, :, 1].ravel())
            b_vals.append(img[:, :, 2].ravel())
        except Exception:
            continue

    r, g, b = np.concatenate(r_vals), np.concatenate(g_vals), np.concatenate(b_vals)
    means = [r.mean(), g.mean(), b.mean()]
    stds  = [r.std(),  g.std(),  b.std()]

    print(f"RGB Mean : R={means[0]:.4f}  G={means[1]:.4f}  B={means[2]:.4f}")
    print(f"RGB Std  : R={stds[0]:.4f}  G={stds[1]:.4f}  B={stds[2]:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, name, color in zip(axes,
                                      [r, g, b],
                                      ["Red", "Green", "Blue"],
                                      ["#E74C3C", "#2ECC71", "#3498DB"]):
        ax.hist(data, bins=256, range=(0, 1), color=color, alpha=0.75, density=True)
        ax.axvline(data.mean(),     color="black", lw=1.5, ls="--", label=f"mean={data.mean():.3f}")
        ax.axvline(np.median(data), color="gray",  lw=1.2, ls=":",  label=f"median={np.median(data):.3f}")
        ax.set_title(f"{name} Channel  (std={data.std():.3f})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Pixel value (normalised 0–1)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    fig.suptitle("Tiny-ImageNet-200 — Pixel Intensity Distribution (5 000-image sample)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pixel_intensity_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Step 2D: Generate random 10x10 sample grid of images
    fig, axes = plt.subplots(10, 10, figsize=(20, 20),
                              gridspec_kw={"hspace": 0.05, "wspace": 0.05})
    for ax, path in zip(axes.ravel(), random.sample(all_train_paths, 100)):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.axis("off")
    fig.suptitle("Tiny-ImageNet-200 — Random Sample Grid (100 images)",
                 fontsize=16, fontweight="bold", y=1.00)
    plt.savefig(RESULTS_DIR / "sample_grid.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Step 2E: Generate per-class sample strip (200 classes in a 20x10 grid)
    fig, axes = plt.subplots(10, 20, figsize=(40, 20),
                              gridspec_kw={"hspace": 0.6, "wspace": 0.05})
    for ax, wnid in zip(axes.ravel(), sorted(train_data.keys())):
        ax.imshow(Image.open(random.choice(train_data[wnid])).convert("RGB"))
        ax.set_title(word_map.get(wnid, wnid).split(",")[0][:14], fontsize=6, pad=2)
        ax.axis("off")
    for ax in axes.ravel()[len(train_data):]:
        ax.axis("off")
    fig.suptitle("Tiny-ImageNet-200 — One Sample per Class",
                 fontsize=18, fontweight="bold", y=1.01)
    plt.savefig(RESULTS_DIR / "per_class_samples.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Persist the collected statistics to JSON
    stats["rgb_mean"]          = [round(m, 6) for m in means]
    stats["rgb_std"]           = [round(s, 6) for s in stds]
    stats["expected_mean_ref"] = [0.480, 0.448, 0.398]
    stats["expected_std_ref"]  = [0.277, 0.269, 0.282]

    out = {}
    for k, v in stats.items():
        if isinstance(v, list):
            out[k] = [float(x) if hasattr(x, 'item') else x for x in v]
        elif hasattr(v, 'item'):
            out[k] = v.item()
        else:
            out[k] = v

    with open(RESULTS_DIR / "dataset_stats.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"✓  All outputs saved to {RESULTS_DIR}/")