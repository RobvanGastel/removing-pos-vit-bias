import argparse

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

from rasa.model import RASAModel 

class RASA(pl.LightningModule):
    def __init__(
        self,
        config: argparse.Namespace,
        encoder
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.model = RASAModel(config, encoder=encoder)

    def configure_optimizers(self, config, model : nn.Module):
        # Freeze rest of head and encoder
        for name, param in model.head.named_parameters():
            if not name.startswith("pos_pred"):
                param.requires_grad_(False)
        for param in model.encoder.parameters():
            param.requires_grad_(False)

        # TODO: Exclude norm and bias for weight decay?
        head_params_named = [
            param
            for name, param in model.head.named_parameters()
            if name.startswith("pos_pred")
        ]
        params = [{"params": head_params_named, "lr": config.lr_head}]

        optimizer = torch.optim.AdamW(params, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.train_iters_per_epoch * config.epochs,
            eta_min=config.final_lr,
        )

        return optimizer, scheduler
