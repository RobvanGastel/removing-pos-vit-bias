import math
import argparse

import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from rasa.model import RASAModel 
from rasa.metrics import PredsmIoUKmeans

class RASA(pl.LightningModule):
    def __init__(
        self,
        config: argparse.Namespace,
        encoder
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])

        self.config = config
        self.val_iters = config.val_iters
        self.n_clusters = config.num_clusters_kmeans_miou
        self.train_iters_per_epoch = config.n_samples // config.batch_size

        self.model = RASAModel(config, encoder=encoder)
        self.preds_miou_x = PredsmIoUKmeans(self.n_clusters, config.num_classes)
        self.preds_miou_pos_x = PredsmIoUKmeans(self.n_clusters, config.num_classes)
        self.preds_miou_no_pos_x = PredsmIoUKmeans(self.n_clusters, config.num_classes)


    def on_train_epoch_start(self):
        self.val_loss_mse = []
        self.val_loss_mse_pos_x = []
        self.val_pos_x_to_x_sim = []
        self.val_loss_mse_no_pos_x = []
        self.val_pos_x_to_pos_emb_sim = []
        self.val_no_pos_x_to_x_sim = []
        self.val_no_pos_x_to_pos_emb_sim = []

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

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        out_dict = self.model.encoder.forward_features(batch["images"])
        x = out_dict["x_norm_patchtokens"]

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

        self.log("train/loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        b, c, h, w = batch["images"].shape
        out_dict = self.model.encoder.forward_features(batch["images"])
        x = out_dict["x_norm_patchtokens"]

        ps = x.shape[1]
        bs = x.shape[0]
        x_pre = self.model.head.forward(x, use_pos_pred=False, return_pos_info=False)
        y = self.model.head.forward_pos_pred(x_pre)  # shape: (B, N, D) -> (B, N, 2)
        
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
        ).float() 
        loss_mse = F.mse_loss(y, tgt)

        # Original evaluations
        # Eval 1: Our target loss we try to minimize is the MSE loss
        # between the predicted position and the actual position of an image patch
        self.val_loss_mse.append(loss_mse)
        x_pos, x_no_pos = self.model.head.decompose_pos_2D(x_pre, ll_weight=self.model.head.pos_pred.weight)

        # Eval 2: How good is the prediction with the pred layer when using the positional information ONLY?
        pos = self.model.head.forward_pos_pred(x_pos)
        loss_mse_pos_x = F.mse_loss(pos, tgt)
        self.val_loss_mse_pos_x.append(loss_mse_pos_x)

        # Eval 3: How good is the prediction with the pred layer after removing the positional information?
        pos = self.model.head.forward_pos_pred(x_no_pos)
        loss_mse_no_pos_x = F.mse_loss(pos, tgt)
        self.val_loss_mse_no_pos_x.append(loss_mse_no_pos_x)

        # Eval 4: How similar is the positional information to the positional embedding of the backbone?
        # Eval 5: How similar is the positional information after
        # removing the positional information to the positional embedding of the backbone?

        # Eval 6: How similar is the positional information after
        # removing the positional information to the positional information?
        no_pos_x_to_x_sim = torch.nn.functional.cosine_similarity(x_no_pos, x, dim=-1, eps=1e-8)  # shape: (B, N)
        self.val_no_pos_x_to_x_sim.append(no_pos_x_to_x_sim.mean())

        # Eval 7: How similar is the positional information to original backbone embeddings
        # after removing the positional information?
        pos_x_to_x_sim = torch.nn.functional.cosine_similarity(x_pos, x, dim=-1, eps=1e-8)  # shape: (B, N)
        self.val_pos_x_to_x_sim.append(pos_x_to_x_sim.mean())

        if self.val_iters is None or batch_idx < self.val_iters:
            self.segmentation_validation_step(x, batch["masks"], self.preds_miou_x)
            self.segmentation_validation_step(x_pos, batch["masks"], self.preds_miou_pos_x)
            self.segmentation_validation_step(x_no_pos, batch["masks"], self.preds_miou_no_pos_x)

    def segmentation_validation_step(self, embs: torch.Tensor, mask: torch.Tensor, preds_miou) -> None:
        # Validate for self.val_iters. Constrained to only parts of the validation set as mIoU calculation
        # would otherwise take too long.
        with torch.no_grad():
            # Process gt seg masks
            bs = mask.size(0)
            gt = mask.float()

            # mask to remove object boundary class
            valid = gt != 255

            # store embeddings, valid masks and gt for clustering after validation end
            res_w = int(np.sqrt(embs.size(1)))
            embs = embs.permute(0, 2, 1).reshape(bs, self.model.encoder.embed_dim, res_w, res_w)
            preds_miou.update(valid, embs, gt)

    def segmentation_validation_epoch_end(self, preds_miou):
        # Trigger computations for rank 0 process
        res_kmeans = preds_miou.compute(self.trainer.is_global_zero)
        preds_miou.reset()
        
        if res_kmeans is not None:  # res_kmeans is none for all processes with rank != 0
            for k, name, res_k in res_kmeans:
                miou_kmeans, tp, fp, fn, _, matched_bg = res_k
                print("miou: ", miou_kmeans)
                self.log(f"val/K={name}_miou", round(miou_kmeans, 8))

                # Log precision and recall values for each class
                for i, (tp_class, fp_class, fn_class) in enumerate(zip(tp, fp, fn)):
                    class_name = self.trainer.datamodule.class_id_to_name(i)
                
                self.log(f"val/K={name}_{class_name}_precision", round(tp_class / max(tp_class + fp_class, 1e-8), 8))
                self.log(f"val/K={name}_{class_name}_recall", round(tp_class / max(tp_class + fn_class, 1e-8), 8))
                if k > self.num_classes:
                    # Log percentage of clusters assigned to background class
                    self.log(f"val/K={name}-percentage-bg-cluster", round(matched_bg, 8))

    def on_validation_epoch_end(self) -> None:
        # Average the validation losses
        loss_mse = torch.stack(self.val_loss_mse).mean()
        loss_mse_pos_x = torch.stack(self.val_loss_mse_pos_x).mean()
        loss_mse_no_pos_x = torch.stack(self.val_loss_mse_no_pos_x).mean()
        no_pos_x_to_x_sim = torch.stack(self.val_no_pos_x_to_x_sim).mean()
        pos_x_to_x_sim = torch.stack(self.val_pos_x_to_x_sim).mean()

        self.log("val/loss_mse", loss_mse, prog_bar=True)
        self.log("val/loss_mse_pos_x", loss_mse_pos_x, prog_bar=True)
        self.log("val/loss_mse_no_pos_x", loss_mse_no_pos_x, prog_bar=True)
        self.log("val/no_pos_x_to_x_sim", no_pos_x_to_x_sim, prog_bar=True)
        self.log("val/pos_x_to_x_sim", pos_x_to_x_sim, prog_bar=True)

        # TODO: Add these statistics
        # pos_x_to_pos_emb_sim = torch.stack(self.val_pos_x_to_pos_emb_sim).mean()
        # no_pos_x_to_pos_emb_sim = torch.stack(self.val_no_pos_x_to_pos_emb_sim).mean()
        # self.log("val/no_pos_x_to_pos_emb_sim", no_pos_x_to_pos_emb_sim, prog_bar=True)
        # self.log("val/pos_x_to_pos_emb_sim", pos_x_to_pos_emb_sim, prog_bar=True)

        # Reset the validation metrics
        self.val_loss_mse = []
        self.val_loss_mse_pos_x = []
        self.val_pos_x_to_x_sim = []
        self.val_loss_mse_no_pos_x = []
        self.val_pos_x_to_pos_emb_sim = []
        self.val_no_pos_x_to_x_sim = []
        self.val_no_pos_x_to_pos_emb_sim = []
        
        self.segmentation_validation_epoch_end(self.preds_miou_x)
        self.segmentation_validation_epoch_end(self.preds_miou_pos_x)
        self.segmentation_validation_epoch_end(self.preds_miou_no_pos_x)
            