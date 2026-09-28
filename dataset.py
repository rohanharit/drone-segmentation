import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


def remap_mask(mask):
    new_mask = np.zeros_like(mask)
    for new_class, old_classes in config.GROUPED_CLASSES.items():
        for c in old_classes:
            new_mask[mask == c] = new_class
    return new_mask


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], 0)
        mask = remap_mask(mask)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        return img, mask.long()


def get_train_transform(image_size=None):
    size = image_size or config.IMAGE_SIZE
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])


def get_val_transform(image_size=None):
    size = image_size or config.IMAGE_SIZE
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(),
        ToTensorV2(),
    ])


def get_dataloaders(image_dir=None, mask_dir=None, batch_size=None,
                    num_workers=None, test_size=None):
    image_dir = image_dir or config.IMAGE_DIR
    mask_dir = mask_dir or config.MASK_DIR
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS
    test_size = test_size or config.TEST_SIZE

    image_paths = sorted(glob(f"{image_dir}/*.png") + glob(f"{image_dir}/*.jpg"))
    mask_paths = sorted(glob(f"{mask_dir}/*.png"))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {image_dir}")
    if len(mask_paths) == 0:
        raise FileNotFoundError(f"No masks found in {mask_dir}")

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths,
        test_size=test_size,
        random_state=config.RANDOM_STATE,
    )

    train_dataset = SegmentationDataset(train_imgs, train_masks, get_train_transform())
    val_dataset = SegmentationDataset(val_imgs, val_masks, get_val_transform())

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader
