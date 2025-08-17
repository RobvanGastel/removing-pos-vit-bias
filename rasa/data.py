from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation as VOCBaseSeg


class VOCSegmentation(VOCBaseSeg):
    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transform=transform,
            target_transform=None,
        )

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        # Fetch the image and mask using the original implementation
        img = self.images[index]
        mask = self.masks[index]

        img = np.array(Image.open(img).convert("RGB"))
        mask = np.array(Image.open(mask))

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

        return {"images": img, "masks" : mask }


class VOCDataModule(pl.LightningDataModule):
    CLASS_IDX_TO_NAME = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        val_transform: Optional[Callable] = None,
        train_transform: Optional[Callable] = None,
        shuffle: bool = False,
        year: str = "2012",
        return_masks: bool = False,
        drop_last: bool = False,
        download: bool = False,
    ):
        """
        Uses torchvision.datasets.VOCSegmentation. If 
        return_masks=False, the training dataset applies
        `transform` to images only. For joint img+mask
        transforms, pass them via `transforms`.
        """
        super().__init__()
        self.year = year
        self.shuffle = shuffle
        self.download = download
        self.drop_last = drop_last
        self.return_masks = return_masks
        self.root = data_dir  # expects <data_dir>/VOC2012 or <data_dir>/VOCdevkit/VOC2012
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform

        # Train dataset
        self.voc_train = VOCSegmentation(
            root=self.root, year=self.year, image_set="train",
            transform=self.train_transform, download=self.download
        )

        # Val dataset: always return (img, mask)
        self.voc_val = VOCSegmentation(
            root=self.root, year=self.year, image_set="val",
            transform=self.val_transform, download=self.download
        )

    def __len__(self):
        return len(self.voc_train)

    def class_id_to_name(self, i: int):
        return self.CLASS_IDX_TO_NAME[i]

    def setup(self, stage: Optional[str] = None):
        return

    def train_dataloader(self):
        return DataLoader(
            self.voc_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.voc_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=True,
        )

    def get_train_dataset_size(self):
        return len(self.voc_train)

    def get_val_dataset_size(self):
        return len(self.voc_val)

    def get_num_classes(self):
        return len(self.CLASS_IDX_TO_NAME)


def get_training_data(dataset_config, batch_size, num_workers) -> VOCDataModule:
    # Extract configurations
    input_size = dataset_config["size_crops"]
    min_scale_factor = dataset_config.get("min_scale_factor", 0.25)
    max_scale_factor = dataset_config.get("max_scale_factor", 1.0)
    blur_strength = dataset_config.get("blur_strength", 1.0)
    jitter_strength = dataset_config.get("jitter_strength", 0.4)
    val_size = dataset_config["size_crops_val"]

    # Training augmentation pipeline
    train_transform = A.Compose(
        [
            A.RandomResizedCrop(
                height=input_size,
                width=input_size,
                scale=(min_scale_factor, max_scale_factor),
                p=1.0,
            ),
            A.ColorJitter(
                brightness=0.8 * jitter_strength,
                contrast=0.8 * jitter_strength,
                saturation=0.8 * jitter_strength,
                hue=0.2 * jitter_strength,
                p=0.8,
            ),
            A.ToGray(p=0.2),
            A.GaussianBlur(
                blur_limit=(3, 7),  # Common kernel size range
                sigma_limit=(blur_strength * 0.1, blur_strength * 2.0),
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.255],
            ),
            ToTensorV2(),
        ]
    )

    # Validation pipeline
    val_transform = A.Compose(
        [
            A.Resize(height=val_size, width=val_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

    # TODO: Make more flexible by passing the dataset object
    # currently only works for VOC.
    data_module = VOCDataModule(
        data_dir=dataset_config["data_path"],
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=train_transform,
        val_transform=val_transform,
    )
    return data_module
