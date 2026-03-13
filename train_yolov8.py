from ultralytics import YOLO, settings
import torch
from pathlib import Path

def train_yolov8_segmentation(data_yaml_path, model_size='n', epochs=100, batch_size=16, cache='true', plots=False):
    """
    Train YOLOv8 segmentation model on DAVIS dataset

    Args:
        data_yaml_path (str): Path to data.yaml file
        model_size (str): Model size ('n', 's', 'm', 'l', 'x' for nano to extra-large)
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        cache (str|bool): Whether to cache images for faster training. Can be 'true', 'false', or 'disk'.
        plots (bool): Whether to save training plots. Disable if matplotlib/Pillow plotting crashes.
    """

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Configure Ultralytics to use a project-local weights directory
    weights_dir = Path.cwd() / 'weights'
    weights_dir.mkdir(exist_ok=True)
    settings.update({'weights_dir': str(weights_dir)})

    # Load model
    model_name = f'yolov8{model_size}-seg.pt'
    model = YOLO(model_name)

    # Training configuration
    training_args = {
        'data': data_yaml_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': 640,  # Input image size
        'device': device,
        'workers': 4,  # Number of worker threads
        'project': 'yolov8_segmentation',
        'name': f'davis_yolov8{model_size}_seg',
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'cache': cache,  # Cache images for faster training
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Classification loss gain
        'dfl': 1.5,  # Distribution focal loss gain
        'nbs': 64,  # Nominal batch size
        'overlap_mask': True,  # Masks should overlap during training
        'mask_ratio': 4,  # Mask downsample ratio
        'dropout': 0.0,
        'val': True,  # Enable validation during training
        'plots': plots,  # Save plots during training
    }

    # Start training
    print("Starting YOLOv8 segmentation training...")
    results = model.train(**training_args)

    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}")

    return results

def validate_model(model_path, data_yaml_path):
    """
    Validate trained model on validation set

    Args:
        model_path (str): Path to trained model weights
        data_yaml_path (str): Path to data.yaml file
    """
    # Configure Ultralytics cache directory
    weights_dir = Path.cwd() / 'weights'
    weights_dir.mkdir(exist_ok=True)
    settings.update({'weights_dir': str(weights_dir)})
    
    model = YOLO(model_path)
    results = model.val(data=data_yaml_path)
    return results

def predict_on_test_set(model_path, test_images_path, output_path):
    """
    Run inference on test images

    Args:
        model_path (str): Path to trained model weights
        test_images_path (str): Path to test images
        output_path (str): Path to save predictions
    """
    model = YOLO(model_path)

    # Run prediction
    results = model.predict(
        source=test_images_path,
        save=True,
        save_txt=True,
        save_conf=True,
        project=output_path,
        name='predictions',
        conf=0.25,  # Confidence threshold
        iou=0.7,    # IoU threshold
        max_det=100,  # Maximum detections per image
        agnostic_nms=False,
        augment=False,
        visualize=False,
        retina_masks=True,  # Use high-resolution segmentation masks
    )

    return results

if __name__ == "__main__":
    # Configuration - use absolute path based on script location
    script_dir = Path(__file__).resolve().parent
    DATA_YAML = str(script_dir / "yolo_dataset" / "data.yaml")
    MODEL_SIZE = 'n'  # nano model
    EPOCHS = 100
    BATCH_SIZE = 16

    # Train model
    print("Training YOLOv8 segmentation model...")
    train_results = train_yolov8_segmentation(DATA_YAML, MODEL_SIZE, EPOCHS, BATCH_SIZE)

    # Validate model
    print("Validating model...")
    best_model_path = str(Path(train_results.save_dir) / 'weights' / 'best.pt')
    val_results = validate_model(best_model_path, DATA_YAML)

    print("Training and validation complete!")
    print(f"Validation results: {val_results}")