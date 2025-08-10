
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


from rasa.voc.data import VOCDataModule, TrainXVOCValDataModule

class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709 following
    https://github.com/facebookresearch/swav/blob/5e073db0cc69dea22aa75e92bfdd75011e888f28/src/multicropdataset.py#L64
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x: Image):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_training_data(config) -> TrainXVOCValDataModule:
    # Extract configurations
    input_size = config.yml["data"]["size_crops"]

    min_scale_factor = config.yml["data"].get("min_scale_factor", 0.25)
    max_scale_factor = config.yml["data"].get("max_scale_factor", 1.0)
    blur_strength = config.yml["data"].get("blur_strength", 1.0)
    jitter_strength = config.yml["data"].get("jitter_strength", 0.4)

    train_transforms = Compose([
        RandomResizedCrop(size=(input_size, input_size), scale=(min_scale_factor, max_scale_factor)),
        RandomApply([ColorJitter(
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.2 * jitter_strength,
        )], p=0.8),
        RandomGrayscale(p=0.2),
        RandomApply([GaussianBlur(sigma=[blur_strength * 0.1, blur_strength * 2.0])], p=0.5),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomApply([RandomRotation(90, interpolation=InterpolationMode.NEAREST, fill=0)], p=0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255]),
    ])

    val_size = config.yml["data"]["size_crops_val"]
    val_image_transforms = Compose([
        Resize((val_size, val_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_target_transforms = Compose([
        Resize((val_size, val_size), interpolation=InterpolationMode.NEAREST),
        ToTensor(),
    ])

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
