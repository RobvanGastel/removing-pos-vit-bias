from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation


class VOCSeg(VOCSegmentation):
    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transform=transform,
            target_transform=target_transform,
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        The overridden __getitem__ method.
        It fetches the image and mask and applies the transforms.
        """
        # Fetch the image and mask using the original implementation
        img = self.images[index]
        mask = self.masks[index]

        img = np.array(Image.open(img).convert("RGB"))
        mask = np.array(Image.open(mask))
        # img = to_tensor(img)
        # mask = to_tensor(mask)

        # Apply the image transform if provided
        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.transform is not None:
        #     # Albumentations expects a dictionary as input
        #     augmented = self.transform(image=img, mask=mask)
        #     img = augmented['image']
        #     mask = augmented['mask']

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

        # if self.target_transform is not None:
        #     mask = self.target_transform(image=mask)['image']

        # Apply the target transform if provided
        # if self.target_transform is not None:
        #     mask = self.target_transform(mask)

        return img, mask

class VOCDataModule(pl.LightningDataModule):

    CLASS_IDX_TO_NAME = [
        "background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
        "chair","cow","diningtable","dog","horse","motorbike","person","pottedplant",
        "sheep","sofa","train","tvmonitor",
    ]

    def __init__(
        self,
        data_dir: str,
        train_split: str,
        val_split: str,
        train_image_transform: Optional[Callable],
        batch_size: int,
        num_workers: int,
        val_image_transform: Optional[Callable] = None,
        val_target_transform: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        shuffle: bool = False,
        return_masks: bool = False,
        drop_last: bool = False,
        year: str = "2012",
        download: bool = False,
    ):
        """
        Uses torchvision.datasets.VOCSegmentation.
        If return_masks=False, the training dataset applies `transform` to images only.
        For joint img+mask transforms, pass them via `transforms`.
        """
        super().__init__()
        self.root = data_dir  # expects <data_dir>/VOC2012 or <data_dir>/VOCdevkit/VOC2012
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_image_transform = train_image_transform
        self.val_transforms = val_transforms
        self.val_image_transform = val_image_transform
        self.val_target_transform = val_target_transform
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.return_masks = return_masks
        self.year = year
        self.download = download

        # Train dataset: choose between joint transforms vs image-only transform
        self.voc_train = VOCSeg(
            root=self.root, year=self.year, image_set="train",
            transform=self.train_image_transform, download=False
        )

        # Val dataset: always return (img, mask)
        if self.val_transforms is not None:
            self.voc_val = VOCSeg(
                root=self.root, year=self.year, image_set="train",
                transforms=self.val_transforms,  download=False
            )
        else:
            self.voc_val = VOCSeg(
                root=self.root, year=self.year, image_set="train",
                transform=self.val_image_transform,
                target_transform=self.val_target_transform,
                download=False
            )

    def __len__(self):
        return len(self.voc_train)

    def class_id_to_name(self, i: int):
        return self.CLASS_IDX_TO_NAME[i]

    def setup(self, stage: Optional[str] = None):
        print(f"Train size {len(self.voc_train)}")
        print(f"Val size {len(self.voc_val)}")

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


class TrainXVOCValDataModule(pl.LightningDataModule):
    def __init__(self, train_datamodule: pl.LightningDataModule, val_datamodule: VOCDataModule):
        super().__init__()
        self.train_datamodule = train_datamodule
        self.val_datamodule = val_datamodule

    def setup(self, stage: str = None):
        self.train_datamodule.setup(stage)
        self.val_datamodule.setup(stage)

    def class_id_to_name(self, i: int):
        return self.val_datamodule.class_id_to_name(i)

    def __len__(self):
        return len(self.train_datamodule)

    def train_dataloader(self):
        return self.train_datamodule.train_dataloader()

    def val_dataloader(self):
        return self.val_datamodule.val_dataloader()
