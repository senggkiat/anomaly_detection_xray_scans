import os
import zipfile

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
POS_DIR = os.path.join(DATA_DIR, "DvXray_Positive_Samples")
NEG_DIR = os.path.join(DATA_DIR, "DvXray_Negative_Samples")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# DvXray dataset download configuration
DVXRAY_URLS = {
    "positive": "https://drive.google.com/uc?id=1NK1DWLMztROwRkJIlYAnexWLgyFv_gDF",
    "negative": "https://drive.google.com/uc?id=18QJyRNVDG6jguNmV04GRuZM98IGdizUb",
}
DVXRAY_FILENAMES = {
    "positive": "DvXray_Positive_Samples.zip",
    "negative": "DvXray_Negative_Samples.zip",
}

def get_result_directory():
    """Returns the directory for the results folder"""
    return RESULTS_DIR

def check_dvxray_exists():
    """Check if DvXray data directories exist."""
    return os.path.isdir(NEG_DIR) and os.path.isdir(POS_DIR)

def get_directories():
    """Returns directories for DvXray_Negative_Samples and DvXray_Positive_Samples"""
    return (NEG_DIR, POS_DIR)

def _download_gdrive_file(url, dest):
    """Download a file from Google Drive, showing progress."""
    try:
        import gdown
        # gdown handles Google Drive authentication properly
        gdown.download(url, dest, quiet=False)
    except ImportError:
        import subprocess
        import sys
        print("  Installing gdown package for reliable Google Drive downloads...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
        gdown.download(url, dest, quiet=False)


def download_and_extract_dvxray():
    """Download and extract the DvXray dataset to DATA_DIR."""
    os.makedirs(DATA_DIR, exist_ok=True)

    for split, url in DVXRAY_URLS.items():
        zip_path = os.path.join(DATA_DIR, DVXRAY_FILENAMES[split])
        if os.path.exists(zip_path):
            print(f"  Archive already exists: {zip_path}")
        else:
            print(f"  Downloading {split} samples...")
            _download_gdrive_file(url, zip_path)

        # Validate that the file is actually a zip file
        if not zipfile.is_zipfile(zip_path):
            # Remove the invalid file
            os.remove(zip_path)
            raise RuntimeError(
                f"Downloaded file is not a valid zip: {zip_path}\n"
                f"This may be due to Google Drive download restrictions. "
                f"Please download the dataset manually from:\n"
                f"  Positive samples: {DVXRAY_URLS['positive']}\n"
                f"  Negative samples: {DVXRAY_URLS['negative']}\n"
                f"Extract to: {DATA_DIR}"
            )

        # Check if extraction is needed
        target_dir = os.path.join(DATA_DIR, f"DvXray_{split.capitalize()}_Samples")
        if os.path.isdir(target_dir) and os.listdir(target_dir):
            print(f"  {split.capitalize()} samples already extracted to {target_dir}")
            continue

        print(f"  Extracting {split} samples...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract with progress bar
            for member in tqdm(zf.infolist(), desc=f"  Extracting {split}", unit="file"):
                zf.extract(member, DATA_DIR)
        print(f"  Extracted to {DATA_DIR}")


def stratified_subset(dataset, n_samples, seed=42):
    """Stratified sampling: equal samples per class, fixed seed for reproducibility."""
    np.random.seed(seed)
    labels = dataset.labels.squeeze()
    classes = np.unique(labels)
    per_class = n_samples // len(classes)
    indices = []
    for c in classes:
        class_idx = np.where(labels == c)[0]
        chosen = np.random.choice(class_idx, size=min(per_class, len(class_idx)), replace=False)
        indices.extend(chosen)
    np.random.shuffle(indices)
    return torch.utils.data.Subset(dataset, indices)


def visualize_samples(dataset, n_classes=4):
    """Display one sample per class from DvXrayDataset (dual-view: OL + SD) with bounding boxes."""
    import os, json
    from PIL import Image
    import matplotlib.patches as patches

    # Extract pos_dir from the samples (positive samples have label >= 1)
    pos_dir = None
    for ol_path, sd_path, label in dataset.samples:
        if label >= 1:  # Positive sample
            pos_dir = os.path.dirname(ol_path)
            break

    if not pos_dir:
        print("⚠ No positive samples found in dataset.")
        return

    found = {}

    for fname in os.listdir(pos_dir):
        if not fname.endswith("_OL.png") or len(found) >= n_classes:
            continue
        json_path = os.path.join(pos_dir, fname.replace("_OL.png", ".json"))
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            data = json.load(f)
        objects = data.get("objects", [])
        if not objects:
            continue
        # Get all threat labels (exclude "Benign")
        threat_labels = [obj["label"] for obj in objects if obj["label"] != "Benign"]
        if not threat_labels:
            continue
        # Use the first threat label as the representative class
        label = threat_labels[0]
        if label not in found:
            ol_path = os.path.join(pos_dir, fname)
            sd_path = os.path.join(pos_dir, fname.replace("_OL.png", "_SD.png"))
            found[label] = (ol_path, sd_path, objects)

    if not found:
        print("⚠ No samples with threat objects found.")
        return
    
    fig, axes = plt.subplots(len(found), 2, figsize=(10, 5 * len(found)))
    
    # Handle edge case when only 1 class found (axes won't be 2D array)
    if len(found) == 1:
        axes = axes.reshape(1, -1)

    for i, cls in enumerate(list(found.keys())):
        ol_path, sd_path, objects = found[cls]
        ol_img = Image.open(ol_path).convert('RGB')
        sd_img = Image.open(sd_path).convert('RGB')
        
        axes[i, 0].imshow(ol_img)
        axes[i, 0].set_title(f"{cls} — OL", fontsize=12)
        axes[i, 1].imshow(sd_img)
        axes[i, 1].set_title(f"{cls} — SD", fontsize=12)

        # Draw bounding boxes for both views
        for view_idx, (ax, img) in enumerate([(axes[i, 0], ol_img), (axes[i, 1], sd_img)]):
            for obj in objects:
                # Determine which view we're drawing for
                bbox_key = "ol_bb" if view_idx == 0 else "sd_bb"

                if bbox_key in obj:
                    bbox = obj[bbox_key]
                    # Bounding box format: [x_min, y_min, x_max, y_max] (absolute pixels)
                    x_min, y_min, x_max, y_max = bbox

                    # Calculate width and height from coordinates
                    width_px = x_max - x_min
                    height_px = y_max - y_min

                    rect = patches.Rectangle(
                        (x_min, y_min), width_px, height_px,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    # Add label text
                    ax.text(x_min, y_min - 5, obj["label"],
                           color='red', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                    
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


def plot_comparison(results):
    """
    Plot training loss curves and test accuracy bar chart.

    Args:
        results: dict with keys like 'CNN + Basic', 'CNN + Augmented',
                 'ViT + Basic', 'ViT + Augmented'.
                 Each value is a dict with 'losses' (list) and 'accuracy' (float).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    for name, data in results.items():
        ax1.plot(data['losses'], label=name)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy bar chart
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] * 100 for n in names]
    colors = ['#2196F3', '#1565C0', '#FF9800', '#E65100']
    bars = ax2.bar(names, accuracies, color=colors[:len(names)])
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy Comparison')
    ax2.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    plt.show()
