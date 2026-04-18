# CNN Model – DvXray Anomaly Detection

This notebook implements multiple models for anomaly detection on the DvXray dual-view X-ray baggage dataset.

## Getting Started

### For New Users – Downloading the Dataset

**Run Cell 5 and above** to automatically download the DvXray dataset. Cell 5 contains the `DvXrayDataset` initialization with `download=True`, which fetches the dataset from Google Drive if it is not already present on your machine.

```python
# Cell 5 — this triggers the download if data is missing
dataset = DvXrayDataset(transform=transform, download=True)
```

### Prerequisites

Install the required dependencies:

```bash
pip install -r ../requirement.txt
```

### Notebook Structure

| Cell | Description                              |
|------|------------------------------------------|
| 1    | Imports and CUDA device setup            |
| 2    | Device detection (GPU / MPS / CPU)       |
| 3-4  | `DvXrayDataset` class definition         |
| **5**| **Dataset loading & download** ⬅️ start here |
| 6+   | Model definition, training, and evaluation |

## Dataset

**DvXray** – A large-scale dual-view X-ray baggage dataset for prohibited item detection.

- **Views:** Overlook (OL) & Side (SD) X-ray images
- **15 threat classes:** Gun, Knife, Hammer, Battery, etc.
- **Negative samples:** Benign baggage
- **Annotations:** JSON with bounding boxes




