import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import torch


# DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), "data")
# POS_DIR = os.path.join(DATA_DIR, "DvXray_Positive_Samples")
# NEG_DIR = os.path.join(DATA_DIR, "DvXray_Negative_Samples")

POS_DIR = "/content/DvXray_Positive_Samples"
NEG_DIR = "/content/DvXray_Negative_Samples"

# DvXray dataset download configuration
DVXRAY_URLS = {
    "positive": "https://drive.google.com/uc?id=1NK1DWLMztROwRkJIlYAnexWLgyFv_gDF",
    "negative": "https://drive.google.com/uc?id=18QJyRNVDG6jguNmV04GRuZM98IGdizUb",
}
DVXRAY_FILENAMES = {
    "positive": "DvXray_Positive_Samples.zip",
    "negative": "DvXray_Negative_Samples.zip",
}


def check_dvxray_exists():
    """Check if DvXray data directories exist."""
    return os.path.isdir(NEG_DIR) and os.path.isdir(POS_DIR)

def get_directories():
    """Returns directories for DvXray_Negative_Samples and DvXray_Positive_Samples"""
    return (NEG_DIR, POS_DIR)

def _download_gdrive_file(url, dest):
    """Download a file from Google Drive, showing progress."""
    import requests

    session = requests.Session()
    response = session.get(url, stream=True)

    # Handle Google Drive virus scan warning
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
    if token:
        response = session.get(url, params={"confirm": token}, stream=True)

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = downloaded / total_size * 100
                print(f"\r  Downloading: {percent:.1f}%", end="")
    print()


def download_and_extract_dvxray(root_dir):
    """Download and extract the DvXray dataset to root_dir."""
    os.makedirs(root_dir, exist_ok=True)

    for split, url in DVXRAY_URLS.items():
        zip_path = os.path.join(root_dir, DVXRAY_FILENAMES[split])
        if os.path.exists(zip_path):
            print(f"  Archive already exists: {zip_path}")
        else:
            print(f"  Downloading {split} samples...")
            _download_gdrive_file(url, zip_path)

        print(f"  Extracting {split} samples...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root_dir)
        print(f"  Extracted to {root_dir}")


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
