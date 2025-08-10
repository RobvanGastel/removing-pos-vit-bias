import yaml
import argparse

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

from rasa import RASA
from rasa.data import get_training_data

def post_train_rasa(config : argparse.Namespace, encoder):
    torch.set_float32_matmul_precision("medium")
    seed_everything(config.seed)

    for i in range(config.start_pos_layers, config.end_pos_layers):
        # TODO: Data config

        config.n_pos_layers = i
        data_module, n_img = get_training_data(config) 
        model = RASA(config, encoder=encoder)

        trainer = Trainer(
            check_val_every_n_epoch=1,
            # logger=..., # weights and biases?
            max_epochs=config.epochs,
            accelerator="cuda",
            fast_dev_run=True,
            log_every_n_steps=400,
            benchmark=True,
            deterministic=False,
            num_sanity_val_steps=0,
            detect_anomaly=False, 
            callbacks=[
                ModelCheckpoint(
                    dirpath=config.output,
                    save_top_k=-1,
                    verbose=True,
                    save_on_train_epoch_end=True,
                )
            ]
        )

        trainer.fit(model, datamodule=data_module)
        # prev_model = model, iterate this process


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
        default="./configs/rasa_baseline.yml",
        help="Post-training RASA setup"
    )
    config = parser.parse_args()

    # TODO: Configurable
    encoder = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model=f"dinov2_vits14_reg",
    ).cuda()

    # TODO: Configurable
    config.seed = 42
    config.start_pos_layers = 0
    config.end_pos_layers = 22
    config.lr_head = 0.0002
    config.final_lr = 0.
    config.weight_decay = 0.
    config.epochs = 5
    config.num_workers = 8
    config.batch_size = 8
    config.output = f"/output/{config.exp_name}"

    with open(config.path, "r") as f:
        data = yaml.safe_load(f)
    config.yml = data

    post_train_rasa(config, encoder)
