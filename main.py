import argparse

import torch

from pytorch_lightning import Trainer, seed_everything

from rasa import RASA
# from rasa.data import ...

def post_train_rasa(config : argparse.Namespace, encoder):

    # from torchvision.datasets import FakeData
    # from torchvision.transforms import Compose, Resize, ToTensor
    # transform = Compose([Resize((224,224)), ToTensor()])
    # data = FakeData(transform=transform, size=500)
    # loader = DataLoader(data, batch_size=16, shuffle=True)


    torch.set_float32_matmul_precision("medium")
    seed_everything(config["seed"])

    # TODO: For loop over ...
    # for i in range(train_config["start_pos_layers"], train_config["end_pos_layers"]):

    model = RASA(config, encoder=encoder)

    # TODO: Checkpoint callback
    
    # data_module = ... 

    # TODO: Check model loading

    trainer = Trainer(
        check_val_every_n_epoch=1,
        # logger=..., # weights and biases?
        max_epochs=config["epochs"],
        accelerator="cuda",
        fast_dev_run=True,
        log_every_n_steps=400,
        benchmark=True,
        deterministic=False,
        num_sanity_val_steps=0,
        detect_anomaly=False, 
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
    config = parser.parse_args()

    # TODO: Configurable
    encoder = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model=f"dinov2_vits14_reg",
    ).cuda()

    # TODO: Configurable
    config.lr = 1e-3
    config.seed = 42
    config.lr_head = 1e-3
    config.final_lr = 1e-3
    config.weight_decay = 1e-5
    config.epochs = 9
    config.train_iters_per_epoch = 10

    post_train_rasa(config, encoder)
