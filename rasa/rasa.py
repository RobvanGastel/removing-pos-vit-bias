import math
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl

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
        self.train_iters_per_epoch = 10582 // 12
        # TODO: Set global batch size


    def configure_optimizers(self):
        # Freeze rest of head and encoder
        for name, param in self.model.head.named_parameters():
            if not name.startswith("pos_pred"):
                param.requires_grad_(False)
        for param in self.model.encoder.parameters():
            param.requires_grad_(False)

        # TODO: Exclude norm and bias for weight decay?
        head_params_named = [
            param
            for name, param in self.model.head.named_parameters()
            if name.startswith("pos_pred")
        ]
        params = [{"params": head_params_named, "lr": self.config.lr_head}]

        optimizer = torch.optim.AdamW(params, weight_decay=0.)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.train_iters_per_epoch * self.config.epochs,
            eta_min=self.config.final_lr,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # TODO: Double check which weights to update, might skip weight decay is not
    # used.
    # def optimizer_step(
    #     self,
    #     epoch: int,
    #     batch_idx: int,
    #     optimizer: Union[Optimizer, LightningOptimizer],
    #     optimizer_closure: Optional[Callable[[], Any]] = None,
    # ):
    #     pass

    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        
        # print(batch[0].shape, batch[1].shape)
        out_dict = self.model.encoder.forward_features(batch[0])
        x = out_dict["x_norm_patchtokens"]
        print(x.shape)
        ps = x.shape[1]
        bs = x.shape[0]
        x = self.model.head.forward(x, use_pos_pred=False, return_pos_info=False)
        y = self.model.head.forward_pos_pred(x)  # shape: (B, N, D) -> (B, N, 2)
        # Create the target positions
        ps_1d = int(math.sqrt(ps))
        r = torch.arange(ps_1d, device=x.device, dtype=torch.float) / (ps_1d - 1)
        c = torch.arange(ps_1d, device=x.device, dtype=torch.float) / (ps_1d - 1)
        r, c = torch.meshgrid(r, c, indexing="ij")
        tgt = torch.stack(
            (
                r.flatten().unsqueeze(0).repeat(bs, 1),
                c.flatten().unsqueeze(0).repeat(bs, 1),
            ),
            dim=-1,
        ).float()  # shape: (B, N, 2) ???

        # Mean Squared Error between the predicted position and the actual position
        loss = F.mse_loss(y, tgt.to(device=x.device))

        # self.log(
        #     "lr_heads",
        #     self.optimizers().param_groups[1]["lr"],
        #     on_step=True,
        #     on_epoch=False,
        # )  # TODO: Maybe the number of the param group is not now 2 because we removed the backbone weoghts
        # self.log(
        #     "weight_decay",
        #     self.optimizers().param_groups[0]["weight_decay"],
        #     on_step=True,
        #     on_epoch=False,
        # )
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        pass