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

    prev_model = None
    for i in range(config.start_pos_layers, config.end_pos_layers):
        config.n_pos_layers = i

        data_module, config.n_samples = get_training_data(config) 
        model = RASA(config, encoder=encoder)

        if prev_model is not None:
            # TODO: How is this step done?
            print(f"Loading weights from previous model for pos layer {i}.")
            # Setup the new model with the previous model's pre_pos_layers and the pos_pred layer as the
            # new pre_pos_layers and create a newly initialized pos_pred layer
            # 1) Move the pos_pred layer to the list of pre_pos_layers in the current model
            prev_model.head.pre_pos_layers.append(prev_model.head.pos_pred)
            # 2) Reinitialize the pos_pred layer from the previous model
            prev_model.head.pos_pred = model.head.pos_pred
            # 3) Load the state dict of the previous model's head to the current model's head
            msg = model.head.load_state_dict(prev_model.head.state_dict(), strict=False)
            print(
                f"Loaded Updated Previous Model Weights to a newly one for pos layer {i}:",
                msg,
            )

        trainer = Trainer(
            check_val_every_n_epoch=1,
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
                    save_top_k=-1,
                    verbose=True,
                    save_on_train_epoch_end=True,
                )
            ]
        )

        trainer.fit(model, datamodule=data_module)
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
        default="./configs/rasa_baseline.yml",
        help="Post-training RASA setup"
    )
    config = parser.parse_args()

    # TODO: Configurable
    encoder = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model=f"dinov2_vits14_reg",
    ).cuda()

    # TODO: Add to yml config
    config.seed = 42
    config.start_pos_layers = 0
    config.end_pos_layers = 22
    config.lr_head = 0.0002
    config.final_lr = 0.
    config.weight_decay = 0.
    config.epochs = 9
    config.num_workers = 8
    config.batch_size = 8
    config.output = f"./logs/output/{config.exp_name}"

    with open(config.path, "r") as f:
        data = yaml.safe_load(f)
    config.yml = data

    post_train_rasa(config, encoder)
