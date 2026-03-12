# YOLOv8 DAVIS Segmentation

A complete pipeline for training YOLOv8 instance segmentation models on the DAVIS dataset.

## Features

- **Automatic DAVIS to YOLO format conversion** - Convert DAVIS annotations to YOLOv8 segmentation format
- **Flexible model sizes** - Support for YOLOv8 nano (n) to extra-large (x) models
- **GPU acceleration** - CUDA support for fast training
- **Validation pipeline** - Built-in model validation after training
- **Configurable parameters** - Control epochs, batch size, train/val split, and more

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA 11.8+ (optional, but recommended for training speed)
- 10+ GB free disk space (for DAVIS dataset and model outputs)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/AlexAvramis/YoloV8-Segmentation.git
cd YoloV8-Segmentation
```

### 2. Create virtual environment
```bash
python -m venv .venv
```

### 3. Activate virtual environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

**Note:** YOLOv8 models are automatically downloaded on first use and cached in `~/.cache/ultralytics/` 

### 5. (Optional) Install CUDA-enabled PyTorch

For CUDA 12.4 (compatible with CUDA 13.2):
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

For CUDA 11.8:
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Dataset Setup

### 1. Download DAVIS dataset
Visit https://davischallenge.org/ and download the DAVIS 2017 dataset.

### 2. Place in project directory
```
YoloV8-Segmentation/
├── DAVIS/
│   ├── Annotations/
│   ├── JPEGImages/
│   ├── ImageSets/
│   ├── README.md
│   └── ...
├── main.py
├── train_yolov8.py
└── ...
```

## Usage

### Option 1: Full Pipeline (Convert + Train)
```bash
python main.py --epochs 100 --batch-size 16 --model-size n
```

### Option 2: Only Convert Dataset
```bash
python main.py --only-conversion
```

### Option 3: Skip Conversion (use existing dataset)
```bash
python main.py --skip-conversion --epochs 100
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--davis-path` | `./DAVIS` | Path to DAVIS dataset |
| `--output-path` | `./yolo_dataset` | Path to output YOLO dataset |
| `--model-size` | `n` | YOLOv8 model size: `n`, `s`, `m`, `l`, `x` |
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `16` | Batch size for training |
| `--val-split` | `0.2` | Validation split ratio (0-1) |
| `--skip-conversion` | `False` | Skip dataset conversion |
| `--only-conversion` | `False` | Only perform conversion, no training |

## Training Examples

```bash
# Train nano model for 50 epochs
python main.py --model-size n --epochs 50

# Train medium model with batch size 32
python main.py --model-size m --batch-size 32

# Train large model for 200 epochs (skip conversion, use existing dataset)
python main.py --skip-conversion --model-size l --epochs 200

# Custom paths
python main.py --davis-path /path/to/davis --output-path /path/to/output
```

## Training on GPU

The script automatically detects CUDA support. To verify GPU usage:

1. Training output will show `Using device: cuda` if GPU is available
2. During training, check GPU memory in the `GPU_mem` column
3. Monitor GPU usage with: `nvidia-smi` (Windows/Linux)

If GPU is not detected, reinstall PyTorch with CUDA support (see [Installation](#installation)).

**Note:** CPU training is very slow. Default batch size of 16 may cause memory issues on CPU. If training on CPU, reduce batch size with `--batch-size 4`.

## Output

Training outputs are saved to: `runs/segment/yolov8_segmentation/<run_name>/`

Contents:
- `weights/best.pt` - Best model weights
- `weights/last.pt` - Last epoch weights
- `results.csv` - Training metrics
- `confusion_matrix.png` - Confusion matrix
- `val_predictions/` - Validation predictions (if enabled)

## Project Structure

```
YoloV8-Segmentation/
├── main.py                      # Main pipeline orchestrator
├── train_yolov8.py             # YOLOv8 training functions
├── convert_davis_to_yolo.py    # DAVIS to YOLO conversion
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
├── README.md                   # This file
├── DAVIS/                      # DAVIS dataset (download required)
└── yolo_dataset/               # Generated YOLO format dataset
```

## Troubleshooting

### CUDA not detected
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
If False, reinstall PyTorch with CUDA support (see [Installation](#installation))

### Out of memory (OOM)
- Reduce `--batch-size` (e.g., 16 → 8)
- Use smaller model (`--model-size n` instead of `l`)
- Use `cache='disk'` in `train_yolov8.py` instead of RAM

### DAVIS dataset conversion errors
- Verify DAVIS dataset structure matches expected format
- Check disk space availability
- Ensure all PNG masks and JPG images are present

## Performance

Training time estimates (on NVIDIA GPU):
- **Nano (n)**: ~15 min/epoch
- **Small (s)**: ~25 min/epoch
- **Medium (m)**: ~45 min/epoch
- **Large (l)**: ~60 min/epoch
- **XLarge (x)**: ~90 min/epoch

Times vary based on GPU type and batch size.

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [DAVIS Challenge](https://davischallenge.org/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

## License

This project uses YOLOv8 which is licensed under AGPL-3.0. See LICENSE files in dependencies for details.

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.
python convert_davis_to_yolo.py
```

This creates:
- `yolo_dataset/images/train/` - Training images
- `yolo_dataset/images/val/` - Validation images
- `yolo_dataset/labels/train/` - Training segmentation masks
- `yolo_dataset/labels/val/` - Validation segmentation masks
- `yolo_dataset/data.yaml` - Dataset configuration

#### Train Model

```bash
python train_yolov8.py
```

## Project Structure

```
YoloV8-Segmentation/
├── main.py                     # Main orchestration script
├── convert_davis_to_yolo.py    # Dataset conversion script
├── train_yolov8.py            # Training script
├── environment.yml            # Conda environment specification
├── README.md                  # This file
├── DAVIS/                     # DAVIS dataset (user provided)
│   ├── JPEGImages/480p/
│   │   ├── train/
│   │   └── val/
│   └── Annotations/480p/
│       ├── train/
│       └── val/
├── yolo_dataset/              # Converted YOLO format dataset (created)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
└── yolov8_segmentation/       # Training outputs (created during training)
    └── davis_yolov8*_seg/
        ├── weights/
        │   ├── best.pt        # Best model weights
        │   └── last.pt        # Last epoch weights
        ├── results.csv
        └── runs/
```

## Training Configuration

### Default Parameters
- **Model Size**: Nano (yolov8n-seg) - ~3.2M parameters
- **Image Size**: 640x640
- **Epochs**: 100
- **Batch Size**: 16 (adjust down if GPU memory is limited)
- **Optimizer**: AdamW
- **Initial Learning Rate**: 0.001

### GPU Memory Requirements (Approximate)

| Model | VRAM Required | Batch Size 16 |
|-------|---------------|---------------|
| nano (n) | 2 GB | ✓ |
| small (s) | 4 GB | ✓ |
| medium (m) | 8 GB | ✓ |
| large (l) | 16 GB | ✓ |
| x-large (x) | 24 GB | Reduce batch size |

### Adjusting for Your Hardware

If you encounter GPU memory errors, reduce batch size:

```bash
python main.py --batch-size 8 --model-size n
```

## Outputs

After training completes, find results in:
- **Best Model**: `yolov8_segmentation/davis_yolov8*_seg/weights/best.pt`
- **Training Metrics**: `yolov8_segmentation/davis_yolov8*_seg/results.csv`
- **Visualizations**: `yolov8_segmentation/davis_yolov8*_seg/` (plots, confusion matrix, etc.)

## Using Trained Model for Inference

See [train_yolov8.py](train_yolov8.py) - the `predict_on_test_set()` function for inference examples.

```python
from ultralytics import YOLO

model = YOLO('yolov8_segmentation/davis_yolov8n_seg/weights/best.pt')
results = model.predict('path/to/image.jpg', conf=0.25)
```

## Troubleshooting

### CUDA/GPU Issues
- Verify GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`
- If False, reinstall PyTorch with CUDA support

### Out of Memory Error
- Reduce batch size: `--batch-size 8`
- Use smaller model: `--model-size n`
- Reduce image size in train_yolov8.py: change `'imgsz': 640` to `'imgsz': 416`

### Dataset Conversion Issues
- Ensure DAVIS folder structure matches expected format
- Check that JPEGImages/480p and Annotations/480p exist
- Verify train/val splits exist in both folders

## Dataset Format

DAVIS dataset structure expected:
```
DAVIS/
├── JPEGImages/
│   └── 480p/
│       └── train/  (and other splits)
└── Annotations/
    └── 480p/
        └── train/  (and other splits)
```

## References

- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [DAVIS Dataset](https://davischallenge.org/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

## Training Results

After training, the best model weights will be saved in:
`runs/segment/train/weights/best.pt`

## Inference

To run inference on new images:
```python
from ultralytics import YOLO
model = YOLO('path/to/best.pt')
results = model.predict('path/to/images', save=True)
```

## Notes

- The DAVIS dataset contains video sequences, so we're treating each frame as an independent image
- All objects in DAVIS are labeled as class 0 ("object") since DAVIS doesn't have predefined classes
- Training time depends on your hardware (GPU recommended)
- Monitor training progress in the `runs/` directory