import os
import cv2
import numpy as np
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
import shutil

def create_yolo_segmentation_format(davis_path, output_path, split='train', year='2017'):
    """
    Convert DAVIS dataset to YOLOv8 segmentation format.

    Args:
        davis_path (str): Path to DAVIS dataset root
        output_path (str): Path to output YOLO format dataset
        split (str): Which DAVIS split to convert ('train', 'val', 'test')
        year (str): DAVIS year ('2016', '2017')
    """

    # DAVIS paths
    images_path = Path(davis_path) / 'JPEGImages' / '480p'
    annotations_path = Path(davis_path) / 'Annotations' / '480p'
    split_file = Path(davis_path) / 'ImageSets' / year / f'{split}.txt'

    # Output paths
    output_images = Path(output_path) / 'images' / split
    output_labels = Path(output_path) / 'labels' / split

    # Create directories
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    # Read split file to get sequence names
    if not split_file.exists():
        print(f"Warning: Split file not found at {split_file}")
        print(f"Available years: 2016, 2017")
        return
    
    with open(split_file, 'r') as f:
        sequence_names = [line.strip() for line in f if line.strip()]

    print(f"Found {len(sequence_names)} sequences in {split} split (year {year})")

    for seq_idx, seq_name in enumerate(sequence_names):
        seq_images_dir = images_path / seq_name
        seq_masks_dir = annotations_path / seq_name
        
        if not seq_images_dir.exists() or not seq_masks_dir.exists():
            print(f"Warning: Sequence {seq_name} not found in dataset, skipping")
            continue
        
        print(f"Processing sequence {seq_idx+1}/{len(sequence_names)}: {seq_name}")

        # Get all frames in sequence
        seq_images = sorted(seq_images_dir.glob('*.jpg'))
        seq_masks = sorted(seq_masks_dir.glob('*.png'))

        if len(seq_images) != len(seq_masks):
            print(f"Warning: Mismatch in {seq_name} - {len(seq_images)} images, {len(seq_masks)} masks")
            continue

        for img_path, mask_path in zip(seq_images, seq_masks):
            # Copy image
            img_filename = f"{seq_name}_{img_path.stem}.jpg"
            shutil.copy(str(img_path), str(output_images / img_filename))

            # Process mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"Warning: Could not read mask {mask_path}, skipping")
                continue
            
            # Ensure mask is 2D - handle case where imread returns 3D despite IMREAD_GRAYSCALE
            if len(mask.shape) != 2:
                # Convert to grayscale if needed
                if len(mask.shape) == 3 and mask.shape[2] == 1:
                    mask = mask[:, :, 0]
                elif len(mask.shape) == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                else:
                    print(f"Warning: Unexpected mask shape {mask.shape} for {mask_path}, skipping")
                    continue
            
            # Get unique object IDs (excluding background 0)
            object_ids = np.unique(mask)
            object_ids = object_ids[object_ids != 0]

            # Create label file
            label_filename = f"{seq_name}_{img_path.stem}.txt"
            label_path = output_labels / label_filename

            with open(label_path, 'w') as f:
                for obj_id in object_ids:
                    # Create binary mask for this object
                    obj_mask = (mask == obj_id).astype(np.uint8)

                    # Find contours
                    contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        # Take the largest contour
                        contour = max(contours, key=cv2.contourArea)

                        # Convert to YOLO format (normalized coordinates)
                        height, width = mask.shape
                        contour = contour.flatten()

                        # Convert to normalized coordinates
                        normalized_contour = []
                        for i in range(0, len(contour), 2):
                            x = contour[i] / width
                            y = contour[i+1] / height
                            normalized_contour.extend([x, y])

                        # Write to file: class_id x1 y1 x2 y2 ... xn yn
                        if len(normalized_contour) >= 6:  # At least 3 points (6 coordinates)
                            line = f"0 {' '.join(map(str, normalized_contour))}\n"
                            f.write(line)

def create_data_yaml(output_path, classes=['object']):
    """
    Create data.yaml file for YOLOv8 training
    """
    # Paths should be relative to the data.yaml file location

    data = {
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }

    yaml_path = Path(output_path) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Created data.yaml at {yaml_path}")

def split_train_val(output_path, val_split=0.2):
    """
    Split training data into train and validation sets
    """
    images_train = Path(output_path) / 'images' / 'train'
    labels_train = Path(output_path) / 'labels' / 'train'

    images_val = Path(output_path) / 'images' / 'val'
    labels_val = Path(output_path) / 'labels' / 'val'

    images_val.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = list(images_train.glob('*.jpg'))
    image_names = [f.stem for f in image_files]

    if len(image_names) < 2:
        print(f"Warning: Only {len(image_names)} images in train set. Skipping train/val split.")
        print("Make sure the DAVIS dataset was properly converted.")
        return

    # Split
    train_names, val_names = train_test_split(image_names, test_size=val_split, random_state=42)

    # Move validation files
    for name in val_names:
        # Move image
        src_img = images_train / f"{name}.jpg"
        dst_img = images_val / f"{name}.jpg"
        if src_img.exists():
            shutil.move(src_img, dst_img)

        # Move label
        src_label = labels_train / f"{name}.txt"
        dst_label = labels_val / f"{name}.txt"
        if src_label.exists():
            shutil.move(src_label, dst_label)

    print(f"Split complete: {len(train_names)} train, {len(val_names)} val")

if __name__ == "__main__":
    # Configuration - using relative paths for GitHub compatibility
    DAVIS_PATH = "./DAVIS"
    OUTPUT_PATH = "./yolo_dataset"

    # Convert train split
    print("Converting DAVIS train split to YOLO format...")
    create_yolo_segmentation_format(DAVIS_PATH, OUTPUT_PATH, split='train')

    # Split into train/val
    print("Splitting train data into train/val sets...")
    split_train_val(OUTPUT_PATH, val_split=0.2)

    # Create data.yaml
    print("Creating data.yaml...")
    create_data_yaml(OUTPUT_PATH)

    print("Conversion complete!")