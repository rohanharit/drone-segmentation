import os
import argparse

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from model import UNet
from dataset import get_dataloaders
from utils import dice_score, save_training_log, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for drone segmentation")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--image_dir", type=str, default=config.IMAGE_DIR)
    parser.add_argument("--mask_dir", type=str, default=config.MASK_DIR)
    parser.add_argument("--save_dir", type=str, default=config.WEIGHTS_DIR)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = criterion(preds, masks)

        total_loss += loss.item()
        total_dice += dice_score(preds, masks).item()

    n = len(loader)
    return total_loss / n, total_dice / n


def main():
    args = parse_args()
    device = config.DEVICE

    train_loader, val_loader = get_dataloaders(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        batch_size=args.batch_size,
    )

    model = UNet(num_classes=config.NUM_CLASSES).to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    best_dice = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_dice": round(val_dice, 4),
            "lr": current_lr,
        })

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.1e}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            best_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)

    final_path = os.path.join(args.save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)

    log_path = os.path.join(config.RESULTS_DIR, "training_log.csv")
    save_training_log(log_path, history)

    plot_path = os.path.join(config.RESULTS_DIR, "training_curves.png")
    plot_training_curves(log_path, save_path=plot_path)


if __name__ == "__main__":
    main()