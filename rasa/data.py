
import random
from PIL import Image, ImageFilter

from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomApply,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)
from torchvision.transforms.functional import InterpolationMode

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from rasa.voc.data import VOCDataModule, TrainXVOCValDataModule

import torch
import torchvision.transforms.functional as F

def gaussian_blur(img, sigma):
    """
    Applies Gaussian blur to a PyTorch tensor image.

    Args:
        img (torch.Tensor): A PyTorch tensor representing the image.
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        torch.Tensor: The blurred image tensor.
    """
    # The torchvision.transforms.functional.gaussian_blur function
    # requires an odd kernel size. We'll use a standard one.
    kernel_size = int(2 * sigma) * 2 + 1
    return F.gaussian_blur(img, kernel_size, [sigma, sigma])


def get_training_data(config) -> TrainXVOCValDataModule:
    # Extract configurations
    input_size = config.yml["data"]["size_crops"]

    min_scale_factor = config.yml["data"].get("min_scale_factor", 0.25)
    max_scale_factor = config.yml["data"].get("max_scale_factor", 1.0)
    blur_strength = config.yml["data"].get("blur_strength", 1.0)
    jitter_strength = config.yml["data"].get("jitter_strength", 0.4)

    # Training Augmentation Pipeline
    # This pipeline is used for training data and includes both geometric and pixel-level augmentations.
    train_transforms = A.Compose(
        [
            # Crop a random part of the image and resize it to the given size.
            A.RandomResizedCrop(
                height=input_size,
                width=input_size,
                scale=(min_scale_factor, max_scale_factor),
                p=1.0,
            ),
            # Randomly change the brightness, contrast, and saturation.
            # Albumentations takes a tuple for the range [-val, +val].
            A.ColorJitter(
                brightness=0.8 * jitter_strength,
                contrast=0.8 * jitter_strength,
                saturation=0.8 * jitter_strength,
                hue=0.2 * jitter_strength,
                p=0.8,
            ),
            # Randomly convert the image to grayscale.
            A.ToGray(p=0.2),
            # Apply Gaussian blur with a random sigma.
            A.GaussianBlur(
                blur_limit=(3, 7),  # Common kernel size range
                sigma_limit=(blur_strength * 0.1, blur_strength * 2.0),
                p=0.5,
            ),
            # Horizontally flip the image.
            A.HorizontalFlip(p=0.5),
            # Vertically flip the image.
            A.VerticalFlip(p=0.5),
            # Randomly rotate the image by 90 degrees.
            A.RandomRotate90(p=0.5),
            # Normalize the image using the ImageNet mean and standard deviation.
            # This is applied before converting to a tensor.
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.255],
            ),
            # Convert the NumPy array to a PyTorch tensor.
            ToTensorV2(),
        ]
    )

    val_size = config.yml["data"]["size_crops_val"]

    # Validation Image Augmentation Pipeline
    # This pipeline is used for the validation set images.
    val_image_transforms = A.Compose(
        [
            # Resize the image to the validation size.
            A.Resize(height=val_size, width=val_size),
            # Normalize the image pixels.
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            # Convert the NumPy arrays to PyTorch tensors.
            ToTensorV2(),
        ]
    )

    # Validation Target Augmentation Pipeline
    # This pipeline is used for the validation set masks.
    val_target_transforms = A.Compose(
        [
            # Resize the mask to the validation size using nearest neighbor interpolation.
            A.Resize(
                height=val_size,
                width=val_size,
                interpolation=cv2.INTER_NEAREST,
            ),
            # Convert the NumPy arrays to PyTorch tensors.
            ToTensorV2(),
        ]
    )
    val_data_module = VOCDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_split="trainaug",
        val_split="val",
        data_dir=config.yml["data"]["voc_data_path"],
        train_image_transform=train_transforms,
        val_image_transform=val_image_transforms,
        val_target_transform=val_target_transforms,
    )

    train_data_module = VOCDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_split="trainaug",
        val_split="val",
        data_dir=config.yml["data"]["voc_data_path"],
        train_image_transform=train_transforms,
        val_image_transform=val_image_transforms,
        val_target_transform=val_target_transforms,
        drop_last=True,
    )

    num_images = 10582
    return TrainXVOCValDataModule(train_data_module, val_data_module), num_images
