import yaml
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from rasa import RASA
from rasa.data import get_training_data

def post_train_rasa(config : argparse.Namespace, encoder):
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(config.seed)

    prev_model = None
    for i in range(config.start_pos_layers, config.end_pos_layers):
        datamodule = get_training_data(config.dataset, config.batch_size, config.num_workers)

        # Set config
        config.n_pos_layers = i
        config.num_classes = datamodule.get_num_classes()
        config.n_samples = datamodule.get_train_dataset_size()
        model = RASA(config, encoder=encoder)

        if prev_model is not None:
            print(f"Loading weights from previous model for pos layer {i}.")
            # Setup the new model with the previous model's pre_pos_layers and the pos_pred layer as the
            # new pre_pos_layers and create a newly initialized pos_pred layer
            # 1) Move the pos_pred layer to the list of pre_pos_layers in the current model
            prev_model.model.head.pre_pos_layers.append(prev_model.model.head.pos_pred)
            # 2) Reinitialize the pos_pred layer from the previous model
            prev_model.model.head.pos_pred = model.model.head.pos_pred
            # 3) Load the state dict of the previous model's head to the current model's head
            msg = model.model.head.load_state_dict(prev_model.model.head.state_dict(), strict=False)
            print(
                f"Loaded Updated Previous Model Weights to a newly one for pos layer {i}:",
                msg,
            )

        csv_logger = CSVLogger(save_dir=config.output, name=config.exp_name)
        trainer = pl.Trainer(
            logger=csv_logger,
            default_root_dir=config.output,
            check_val_every_n_epoch=config.check_val_every,
            max_epochs=config.epochs,
            accelerator="cuda",
            fast_dev_run=False,
            log_every_n_steps=20,
            benchmark=True,
            deterministic=False,
            num_sanity_val_steps=0,
            detect_anomaly=False, 
            callbacks=[
                ModelCheckpoint(
                    dirpath=config.output,
                    save_top_k=0,
                    save_last=True,
                    verbose=True,
                )
            ]
        )

        trainer.fit(model, datamodule=datamodule)
        # Iterate this process
        prev_model = model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="rasa_baseline",
        help="Experiment name",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./configs/rasa.yml",
        help="Post-training RASA setup"
    )
    config = parser.parse_args()

    # Load data, train, encoder config
    with open(config.path, "r") as f:
        data = yaml.safe_load(f)

    for k, v in data["train"].items():
        setattr(config, k, v)
    config.dataset = data["data"]

    # Load DINOv2, DINOv3 encoder dynamically
    encoder_args = dict(
        repo_or_dir=data["encoder"]["repo_or_dir"],
        model=data["encoder"]["model"]
    )
    if "source" in data["encoder"]:
        encoder_args["source"] = data["encoder"]["source"]
    if data["encoder"].get("weights"):
        encoder_args["weights"] = data["encoder"]["weights"]
    encoder = torch.hub.load(**encoder_args).cuda()

    post_train_rasa(config, encoder)


