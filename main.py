"""
Main orchestration script for YOLOv8 DAVIS segmentation training pipeline
"""

import sys
import argparse
from pathlib import Path
from convert_davis_to_yolo import create_yolo_segmentation_format, create_data_yaml
from train_yolov8 import train_yolov8_segmentation, validate_model

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 DAVIS Segmentation Training Pipeline')
    parser.add_argument('--davis-path', type=str, 
                       default='./DAVIS',
                       help='Path to DAVIS dataset')
    parser.add_argument('--output-path', type=str,
                       default='./yolo_dataset',
                       help='Path to output YOLO format dataset')
    parser.add_argument('--model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--cache', type=str, default='true',
                       choices=['true', 'false', 'disk'],
                       help="Whether to cache images for faster training (true/false/disk). Default: true")
    parser.add_argument('--plots', action='store_true',
                       help='Enable Ultralytics training plots (disabled by default to avoid plotting crashes)')
    parser.add_argument('--skip-conversion', action='store_true',
                       help='Skip dataset conversion (dataset already converted)')
    parser.add_argument('--only-conversion', action='store_true',
                       help='Only perform dataset conversion, do not train')
    
    args = parser.parse_args()
    
    davis_path = Path(args.davis_path)
    output_path = Path(args.output_path)
    
    # Validate paths
    if not davis_path.exists():
        print(f"Error: DAVIS dataset not found at {davis_path}")
        sys.exit(1)
    
    # Convert DAVIS to YOLO format
    if not args.skip_conversion:
        print("=" * 60)
        print("Step 1: Converting DAVIS dataset to YOLO format...")
        print("=" * 60)
        
        try:
            print("Converting DAVIS train split...")
            create_yolo_segmentation_format(str(davis_path), str(output_path), split='train')
            
            print("Converting DAVIS val split...")
            create_yolo_segmentation_format(str(davis_path), str(output_path), split='val')

            print("\nCreating data.yaml configuration...")
            create_data_yaml(str(output_path))
            
            print("\n✓ Dataset conversion complete!")
        except Exception as e:
            print(f"Error during dataset conversion: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Skipping dataset conversion (using existing dataset)")
    
    if args.only_conversion:
        print("\n✓ Conversion complete. Dataset ready for training.")
        return
    
    # Train model
    data_yaml_path = str(output_path / 'data.yaml')
    
    if not Path(data_yaml_path).exists():
        print(f"Error: data.yaml not found at {data_yaml_path}")
        print("Please run dataset conversion first or use --skip-conversion flag with existing dataset")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Step 2: Training YOLOv8 segmentation model...")
    print("=" * 60)
    print(f"Model size: {args.model_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    try:
        train_results = train_yolov8_segmentation(
            data_yaml_path,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            cache=args.cache,
            plots=args.plots
        )

        print("=" * 60)
        print("Step 3: Validating trained model...")
        print("=" * 60)

        best_model_path = str(Path(train_results.save_dir) / 'weights' / 'best.pt')
        if Path(best_model_path).exists():
            val_results = validate_model(best_model_path, data_yaml_path)
            print("✓ Validation complete!")
        else:
            print(f"Warning: Best model weights not found at {best_model_path}")

        print("\n" + "=" * 60)
        print("✓ Training pipeline complete!")
        print("=" * 60)
        print(f"Outputs saved to: {train_results.save_dir}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
