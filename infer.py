"""
Run inference with a trained U-Net model on drone images.

Usage:
    python infer.py --input test/ --output outputs/visual/
    python infer.py --input path/to/image.png --output results/predictions/ --show
"""

import os
import argparse
from glob import glob

import torch

import config
from model import UNet
from dataset import get_val_transform
from utils import visualize_prediction


def parse_args():
    parser = argparse.ArgumentParser(description="Run U-Net inference on drone images")
    parser.add_argument("--model", type=str, default="unet_drone_segmentation.pth",
                        help="Path to trained model weights (.pth)")
    parser.add_argument("--input", type=str, default="test/",
                        help="Path to input image or directory of images")
    parser.add_argument("--output", type=str, default="outputs/visual/",
                        help="Directory to save prediction visualizations")
    parser.add_argument("--show", action="store_true",
                        help="Display each prediction in a window")
    return parser.parse_args()


def load_model(weights_path, device):
    """Load a trained UNet model from a .pth file."""
    model = UNet(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {weights_path}")
    return model


def get_image_paths(input_path):
    """Get list of image paths from a file or directory."""
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        paths = sorted(
            glob(f"{input_path}/*.png") +
            glob(f"{input_path}/*.jpg") +
            glob(f"{input_path}/*.jpeg")
        )
        if not paths:
            raise FileNotFoundError(f"No images found in {input_path}")
        return paths

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main():
    args = parse_args()
    device = config.DEVICE
    print(f"Device: {device}")

    # Load model
    model = load_model(args.model, device)
    transform = get_val_transform()

    # Get images
    image_paths = get_image_paths(args.input)
    print(f"Found {len(image_paths)} image(s)")

    # Run inference
    os.makedirs(args.output, exist_ok=True)

    for i, img_path in enumerate(image_paths, 1):
        name = os.path.basename(img_path)
        save_path = os.path.join(args.output, name)

        visualize_prediction(
            img_path, model, transform,
            device=device,
            save_path=save_path,
            show=args.show,
        )

        print(f"[{i}/{len(image_paths)}] Processed {name}")

    print(f"\n✓ All predictions saved to {args.output}")


if __name__ == "__main__":
    main()