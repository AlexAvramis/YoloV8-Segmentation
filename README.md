# YOLOv8 DAVIS Segmentation

A complete pipeline for training YOLOv8 instance segmentation models on the DAVIS dataset.

## Features

- Automatic DAVIS to YOLO format conversion
- Uses official DAVIS 2017 train/val splits (60 train, 30 val sequences)
- Flexible model sizes: YOLOv8 n, s, m, l, x
- GPU acceleration with CUDA when available
- Built-in validation after training
- Configurable cache mode and optional training plots

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA (optional, recommended)
- Enough disk space for DAVIS and training outputs

## Installation

1. Clone the repository

```bash
git clone https://github.com/AlexAvramis/YoloV8-Segmentation.git
cd YoloV8-Segmentation
```

2. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Optional: install CUDA-enabled PyTorch build

CUDA 12.4 wheels:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

CUDA 11.8 wheels:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Dataset Setup

Download DAVIS 2017 from https://davischallenge.org/ and place it under:

```text
YoloV8-Segmentation/
  DAVIS/
    JPEGImages/
    Annotations/
    ImageSets/
```

The pipeline reads split definitions from:

- DAVIS/ImageSets/2017/train.txt
- DAVIS/ImageSets/2017/val.txt

## Usage

Full pipeline (convert + train):

```bash
python main.py --epochs 100 --batch-size 16 --model-size n
```

Only conversion:

```bash
python main.py --only-conversion
```

Skip conversion (use existing yolo_dataset):

```bash
python main.py --skip-conversion --epochs 100
```

## Command Line Arguments

| Argument | Default | Description |
|---|---|---|
| --davis-path | ./DAVIS | Path to DAVIS dataset root |
| --output-path | ./yolo_dataset | Path to generated YOLO dataset |
| --model-size | n | YOLOv8 model size: n, s, m, l, x |
| --epochs | 100 | Number of training epochs |
| --batch-size | 16 | Batch size for training |
| --cache | true | Image cache mode: true, false, or disk |
| --plots | False | Enable Ultralytics training plots |
| --skip-conversion | False | Skip conversion step |
| --only-conversion | False | Run conversion only |

## Output

Training outputs are saved under:

- yolov8_segmentation/davis_yolov8{size}_seg/

Typical files:

- weights/best.pt
- weights/last.pt
- results.csv
- confusion_matrix.png (if plotting enabled)

## Notes

- Plot generation is disabled by default to avoid matplotlib/Pillow plotting crashes in some environments.
- Use --plots only when you want training visualizations.
- All DAVIS objects are trained as a single class named object.

## Troubleshooting

CUDA check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Out-of-memory fixes:

- Lower batch size
- Use model-size n or s
- Use --cache disk

Dataset conversion checks:

- Verify DAVIS/ImageSets/2017/train.txt and val.txt exist
- Verify DAVIS/JPEGImages/480p and DAVIS/Annotations/480p exist

## References

- https://docs.ultralytics.com/
- https://davischallenge.org/
- https://github.com/ultralytics/ultralytics
