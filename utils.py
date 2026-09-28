import os
import csv
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

import config


def dice_score(pred, target, num_classes=None):
    num_classes = num_classes or config.NUM_CLASSES
    pred = torch.argmax(pred, dim=1)
    dice = 0.0

    for cls in range(num_classes):
        pred_c = (pred == cls).float()
        target_c = (target == cls).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice += (2 * intersection + 1e-6) / (union + 1e-6)

    return dice / num_classes


def per_class_iou(pred, target, num_classes=None):
    num_classes = num_classes or config.NUM_CLASSES
    pred = torch.argmax(pred, dim=1)
    ious = {}

    for cls in range(num_classes):
        pred_c = (pred == cls).float()
        target_c = (target == cls).float()
        intersection = (pred_c * target_c).sum().item()
        union = (pred_c + target_c).clamp(0, 1).sum().item()
        iou = (intersection + 1e-6) / (union + 1e-6)
        ious[config.CLASS_NAMES[cls]] = round(iou, 4)

    return ious


def create_color_mask(pred_mask, color_map=None):
    color_map = color_map or config.COLOR_MAP
    h, w = pred_mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in color_map.items():
        color_img[pred_mask == cls] = color
    return color_img


def visualize_prediction(image_path, model, transform, device=None,
                         save_path=None, show=False):
    device = device or config.DEVICE
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    augmented = transform(image=img_rgb)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    pred = cv2.resize(
        pred.astype(np.uint8),
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    color_mask = create_color_mask(pred)
    overlay = cv2.addWeighted(img_rgb, 0.6, color_mask, 0.4, 0)
    combined = np.hstack([img_rgb, color_mask, overlay])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    if show:
        cv2.imshow("Prediction", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return combined


def save_training_log(log_path, history):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(log_path, save_path=None):
    epochs, train_losses, val_losses, val_dices = [], [], [], []

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            val_dices.append(float(row["val_dice"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label="Train Loss", marker="o", markersize=3)
    ax1.plot(epochs, val_losses, label="Val Loss", marker="o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_dices, label="Val Dice", marker="o", markersize=3, color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Validation Dice Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
