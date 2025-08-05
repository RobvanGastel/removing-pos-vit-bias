

def get_training_data(data_config, train_config, val_config, num_workers=12) -> TrainXVOCValDataModule:
    # Init data modules and tranforms
    data_dir = data_config["data_dir"]
    dataset_name = data_config["dataset_name"]
    input_size = data_config["size_crops"]
    # Setup data
    min_scale_factor = data_config.get("min_scale_factor", 0.25)
    max_scale_factor = data_config.get("max_scale_factor", 1.0)

    blur_strength = data_config.get("blur_strength", 1.0)
    jitter_strength = data_config.get("jitter_strength", 0.4)
    color_jitter = ColorJitter(
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.2 * jitter_strength,
    )
    train_transforms = Compose(
        [
            RandomResizedCrop(
                size=(input_size, input_size),
                scale=(min_scale_factor, max_scale_factor),
            ),
            RandomApply([color_jitter], p=0.8),
            RandomGrayscale(p=0.2),
            RandomApply([GaussianBlur(sigma=[blur_strength * 0.1, blur_strength * 2.0])], p=0.5),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomApply(
                [
                    RandomRotation(
                        90,
                        interpolation=InterpolationMode.NEAREST,
                        expand=False,
                        center=None,
                        fill=0,
                    )
                ],
                p=0.5,
            ),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255]),
        ]
    )

    # Setup voc dataset used for evaluation
    val_size = data_config["size_crops_val"]
    val_image_transforms = Compose(
        [
            Resize((val_size, val_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_target_transforms = Compose(
        [
            Resize((val_size, val_size), interpolation=InterpolationMode.NEAREST),
            ToTensor(),
        ]
    )
    val_batch_size = val_config.get("val_batch_size", train_config["batch_size"])

    val_data_module = VOCDataModule(
        batch_size=val_batch_size,
        num_workers=num_workers,
        train_split="trainaug",
        val_split="val",
        data_dir=data_config["voc_data_path"],
        train_image_transform=train_transforms,
        # val_transforms=val_image_transforms
        val_image_transform=val_image_transforms,
        val_target_transform=val_target_transforms,
    )
    num_images = None
    # Setup train data
    if dataset_name == "coco":
        num_images = None
        file_list = os.listdir(os.path.join(data_dir, "train2017"))
        train_data_module = CocoDataModule(
            batch_size=train_config["batch_size"],
            num_workers=num_workers,
            file_list=file_list,
            data_dir=data_dir,
            train_transforms=train_transforms,
            val_transforms=None,
        )
    elif dataset_name == "imagenet100":
        num_images = 126689
        with open("path/to/imagenet100.txt") as f:
            class_names = [line.rstrip("\n") for line in f]
        train_data_module = ImageNetDataModule(
            train_transforms=train_transforms,
            batch_size=train_config["batch_size"],
            class_names=class_names,
            num_workers=num_workers,
            data_dir=data_dir,
            num_images=num_images,
        )
    elif dataset_name == "imagenet1k":
        num_images = 1281167
        data_dir = os.path.join(data_dir, "train")
        class_names = os.listdir(data_dir)
        assert len(class_names) == 1000
        train_data_module = ImageNetDataModule(
            train_transforms=train_transforms,
            batch_size=train_config["batch_size"],
            class_names=class_names,
            num_workers=num_workers,
            data_dir=data_dir,
            num_images=num_images,
        )
    elif dataset_name == "voc":
        num_images = 10582
        train_data_module = VOCDataModule(
            batch_size=train_config["batch_size"],
            num_workers=num_workers,
            train_split="trainaug",
            val_split="val",
            data_dir=data_config["voc_data_path"],
            train_image_transform=train_transforms,
            val_image_transform=val_image_transforms,
            val_target_transform=val_target_transforms,
            drop_last=True,
        )
    else:
        raise ValueError(f"Data set {dataset_name} not supported")

    # Use data module wrapper to have train_data_module provide train loader and voc data module the val loader
    return TrainXVOCValDataModule(train_data_module, val_data_module), num_images
