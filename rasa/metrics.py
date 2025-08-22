import time
from collections import defaultdict
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric


class PredsmIoU(Metric):
    """
    Subclasses Metric. Computes mean Intersection over Union (mIoU) given ground-truth and predictions.
    .update() can be called repeatedly to add data from multiple validation loops.
    """

    def __init__(self, num_pred_classes: int, num_gt_classes: int):
        """
        :param num_pred_classes: The number of predicted classes.
        :param num_gt_classes: The number of gt classes.
        """
        super().__init__(dist_sync_on_step=False)
        self.num_pred_classes = num_pred_classes
        self.num_gt_classes = num_gt_classes
        self.add_state("gt", [])
        self.add_state("pred", [])
        self.n_jobs = -1

    def update(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        self.gt.append(gt)
        self.pred.append(pred)

    def compute(
        self,
        is_global_zero: bool,
        many_to_one: bool = False,
        precision_based: bool = False,
        linear_probe: bool = False,
    ) -> Tuple[float, List[np.int64], List[np.int64], List[np.int64], List[np.int64], float]:
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param is_global_zero: Flag indicating whether process is rank zero. Computation of metric is only triggered
        if True.
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt,  percentage of clusters matched to background class. 1/self.num_pred_classes
        if self.num_pred_classes == self.num_gt_classes.
        """
        if is_global_zero:
            pred = torch.cat(self.pred).cpu().numpy().astype(int)
            gt = torch.cat(self.gt).cpu().numpy().astype(int)
            assert len(np.unique(pred)) <= self.num_pred_classes
            assert np.max(pred) <= self.num_pred_classes
            return self.compute_miou(
                gt,
                pred,
                self.num_pred_classes,
                self.num_gt_classes,
                many_to_one=many_to_one,
                precision_based=precision_based,
                linear_probe=linear_probe,
            )

    def compute_miou(
        self,
        gt: np.ndarray,
        pred: np.ndarray,
        num_pred: int,
        num_gt: int,
        many_to_one=False,
        precision_based=False,
        linear_probe=False,
    ) -> Tuple[float, List[np.int64], List[np.int64], List[np.int64], List[np.int64], float]:
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param gt: numpy array with all flattened ground-truth class assignments per pixel
        :param pred: numpy array with all flattened class assignment predictions per pixel
        :param num_pred: number of predicted classes
        :param num_gt: number of ground truth classes
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt,  percentage of clusters matched to background class. 1/self.num_pred_classes
        if self.num_pred_classes == self.num_gt_classes.
        """
        assert pred.shape == gt.shape
        print(f"seg map preds have size {gt.shape}")
        tp = [0] * num_gt
        fp = [0] * num_gt
        fn = [0] * num_gt
        jac = [0] * num_gt

        if linear_probe:
            reordered_preds = pred
            matched_bg_clusters = {}
        else:
            if many_to_one:
                match = self._original_match(num_pred, num_gt, pred, gt, precision_based=precision_based)
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, matched_preds in match.items():
                    for pred_i in matched_preds:
                        reordered_preds[pred == int(pred_i)] = int(target_i)
                matched_bg_clusters = len(match[0]) / num_pred
            else:
                match = self._hungarian_match(num_pred, num_gt, pred, gt)
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, pred_i in zip(*match):
                    reordered_preds[pred == int(pred_i)] = int(target_i)
                # merge all unmatched predictions to background
                for unmatched_pred in np.delete(np.arange(num_pred), np.array(match[1])):
                    reordered_preds[pred == int(unmatched_pred)] = 0
                matched_bg_clusters = 1 / num_gt

        # tp, fp, and fn evaluation
        for i_part in range(0, num_gt):
            tmp_all_gt = gt == i_part
            tmp_pred = reordered_preds == i_part
            tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
            fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
            fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)

        # Calculate IoU per class
        for i_part in range(0, num_gt):
            jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

        return (
            np.mean(jac),
            tp,
            fp,
            fn,
            reordered_preds.astype(int).tolist(),
            matched_bg_clusters,
        )

    @staticmethod
    def get_score(
        flat_preds: np.ndarray,
        flat_targets: np.ndarray,
        c1: int,
        c2: int,
        precision_based: bool = False,
    ) -> float:
        """
        Calculates IoU given gt class c1 and prediction class c2.
        :param flat_preds: flattened predictions
        :param flat_targets: flattened gt
        :param c1: ground truth class to match
        :param c2: predicted class to match
        :param precision_based: flag to calculate precision instead of IoU.
        :return: The score if gt-c1 was matched to predicted c2.
        """
        tmp_all_gt = flat_targets == c1
        tmp_pred = flat_preds == c2
        tp = np.sum(tmp_all_gt & tmp_pred)
        fp = np.sum(~tmp_all_gt & tmp_pred)
        if not precision_based:
            fn = np.sum(tmp_all_gt & ~tmp_pred)
            jac = float(tp) / max(float(tp + fp + fn), 1e-8)
            return jac
        else:
            prec = float(tp) / max(float(tp + fp), 1e-8)
            return prec

    def compute_score_matrix(
        self,
        num_pred: int,
        num_gt: int,
        pred: np.ndarray,
        gt: np.ndarray,
        precision_based: bool = False,
    ) -> np.ndarray:
        """
        Compute score matrix. Each element i, j of matrix is the score if i was matched j. Computation is parallelized
        over self.n_jobs.
        :param num_pred: number of predicted classes
        :param num_gt: number of ground-truth classes
        :param pred: flattened predictions
        :param gt: flattened gt
        :param precision_based: flag to calculate precision instead of IoU.
        :return: num_pred x num_gt matrix with A[i, j] being the score if ground-truth class i was matched to
        predicted class j.
        """
        print("Parallelizing iou computation")
        start = time.time()
        score_mat = [
            self.get_score(pred, gt, c1, c2, precision_based=precision_based)
            for c2 in range(num_pred)
            for c1 in range(num_gt)
        ]
        print(f"took {time.time() - start} seconds")
        score_mat = np.array(score_mat)
        return score_mat.reshape((num_pred, num_gt)).T

    def _hungarian_match(self, num_pred: int, num_gt: int, pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # do hungarian matching. If num_pred > num_gt match will be partial only.
        iou_mat = self.compute_score_matrix(num_pred, num_gt, pred, gt)
        match = linear_sum_assignment(1 - iou_mat)
        print("Matched clusters to gt classes:")
        print(match)
        return match

    def _original_match(self, num_pred, num_gt, pred, gt, precision_based=False) -> Dict[int, list]:
        score_mat = self.compute_score_matrix(num_pred, num_gt, pred, gt, precision_based=precision_based)
        preds_to_gts = {}
        preds_to_gt_scores = {}
        # Greedily match predicted class to ground-truth class by best score.
        for pred_c in range(num_pred):
            for gt_c in range(num_gt):
                score = score_mat[gt_c, pred_c]
                if (pred_c not in preds_to_gts) or (score > preds_to_gt_scores[pred_c]):
                    preds_to_gts[pred_c] = gt_c
                    preds_to_gt_scores[pred_c] = score
        gt_to_matches = defaultdict(list)
        for k, v in preds_to_gts.items():
            gt_to_matches[v].append(k)
        print("matched clusters to gt classes:")
        return gt_to_matches


class PredsmIoUKmeans(PredsmIoU):
    """
    Used to track k-means cluster correspondence to ground-truth categories during fine-tuning.
    """

    def __init__(
        self,
        clustering_granularities: List[int],
        num_gt_classes: int,
        pca_dim: int = 50,
    ):
        """
        :param clustering_granularities: list of clustering granularities for embeddings
        :param num_gt_classes: number of ground-truth classes
        :param pca_dim: target dimensionality of PCA
        """
        super(PredsmIoU, self).__init__(dist_sync_on_step=False)  # Init Metric super class
        self.pca_dim = pca_dim
        self.num_pred_classes = clustering_granularities
        self.num_gt_classes = num_gt_classes
        self.add_state("masks", [])
        self.add_state("embeddings", [])
        self.add_state("gt", [])
        self.n_jobs = -1  # num_jobs = num_cores
        self.num_train_pca = 4000000  # take num_train_pca many vectors at max for training pca

    def update(self, masks: torch.Tensor, embeddings: torch.Tensor, gt: torch.Tensor) -> None:
        self.masks.append(masks)
        self.embeddings.append(embeddings)
        self.gt.append(gt)

    def predict_only(
        self,
        is_global_zero: bool,
        seed: int = 1,
        remap_to_gt: bool = False,
        precision_based: bool = False,
    ):
        """
        Returns predictions for each k in self.num_pred_classes.
        If remap_to_gt=True, clusters are remapped to GT ids (no IoU is computed/returned).

        Output: List[Tuple[int, np.ndarray]] where each item is (k, preds_1d),
                preds_1d has length equal to number of valid pixels (mask==True).
        """
        if not is_global_zero:
            return None

        # 1) Prep embeddings to pixel grid
        embeddings = torch.cat([e.cpu() for e in self.embeddings], dim=0)
        valid_masks = torch.cat(self.masks, dim=0).cpu().numpy()
        res_w = valid_masks.shape[2]
        embeddings = nn.functional.interpolate(embeddings, size=(res_w, res_w), mode="bilinear")
        embeddings = embeddings.permute(0, 2, 3, 1).reshape(valid_masks.shape[0] * res_w**2, -1).numpy()

        # 2) Normalize + PCA (FAISS)
        normalized = (embeddings - np.mean(embeddings, axis=0)) / (np.std(embeddings, axis=0, ddof=0) + 1e-5)
        d_orig = embeddings.shape[1]
        pca = faiss.PCAMatrix(d_orig, self.pca_dim)
        pca.train(normalized[: self.num_train_pca])
        assert pca.is_trained
        feats = pca.apply_py(normalized)

        # Optional GT (only needed if remapping)
        if remap_to_gt:
            gt_flat = torch.cat(self.gt, dim=0).cpu().numpy()[valid_masks]

        results = []
        for k in self.num_pred_classes:
            # 3) KMeans for this k
            kmeans = faiss.Kmeans(
                self.pca_dim, k, niter=50, nredo=5, seed=seed, verbose=False, gpu=False, spherical=False
            )
            kmeans.train(feats)
            _, pred_labels = kmeans.index.search(feats, 1)  # [N, 1]
            clusters = pred_labels.squeeze()                # [N]

            # 4) Apply valid mask and flatten
            pred_flat = clusters.reshape(valid_masks.shape[0], res_w, res_w)[valid_masks]

            if remap_to_gt:
                # Remap clusters -> GT ids (vectorized LUT)
                if k == self.num_gt_classes:
                    rows, cols = self._hungarian_match(k, self.num_gt_classes, pred_flat, gt_flat)
                    lut = np.zeros(k, dtype=np.int64)  # unmatched -> background 0
                    lut[cols] = rows
                    pred_flat = lut[pred_flat.astype(np.int64)]
                else:
                    match = self._original_match(k, self.num_gt_classes, pred_flat, gt_flat,
                                                 precision_based=precision_based)
                    lut = np.zeros(k, dtype=np.int64)
                    for gt_id, pred_ids in match.items():
                        lut[np.array(pred_ids, dtype=np.int64)] = int(gt_id)
                    pred_flat = lut[pred_flat.astype(np.int64)]

            results.append((k, pred_flat.astype(np.int64)))

        return results

    def compute(self, is_global_zero: bool, seed=1) -> List[any]:
        if is_global_zero:
            # interpolate embeddings to match ground-truth masks spatially
            embeddings = torch.cat([e.cpu() for e in self.embeddings], dim=0)  # move everything to cpu before catting
            valid_masks = torch.cat(self.masks, dim=0).cpu().numpy()
            res_w = valid_masks.shape[2]
            embeddings = nn.functional.interpolate(embeddings, size=(res_w, res_w), mode="bilinear")
            embeddings = embeddings.permute(0, 2, 3, 1).reshape(valid_masks.shape[0] * res_w**2, -1).numpy()

            # Normalize embeddings and reduce dims of embeddings by PCA
            normalized_embeddings = (embeddings - np.mean(embeddings, axis=0)) / (np.std(embeddings, axis=0, ddof=0) + 1e-5)
            d_orig = embeddings.shape[1]
            pca = faiss.PCAMatrix(d_orig, self.pca_dim)
            pca.train(normalized_embeddings[: self.num_train_pca])
            assert pca.is_trained
            transformed_feats = pca.apply_py(normalized_embeddings)

            # Cluster transformed feats with kmeans
            results = []
            gt = torch.cat(self.gt, dim=0).cpu().numpy()[valid_masks]
            for k in self.num_pred_classes:
                kmeans = faiss.Kmeans(
                    self.pca_dim,
                    k,
                    niter=50,
                    nredo=5,
                    seed=seed,
                    verbose=True,
                    gpu=False,
                    spherical=False,
                )
                kmeans.train(transformed_feats)
                _, pred_labels = kmeans.index.search(transformed_feats, 1)
                clusters = pred_labels.squeeze()

                # Filter predictions by valid masks (removes voc boundary gt class)
                pred_flattened = clusters.reshape(valid_masks.shape[0], res_w, res_w)[valid_masks]
                # TODO: Uncoment the following line for checking that all clusters are used.
                # assert len(np.unique(pred_flattened)) == k
                # assert np.max(pred_flattened) == k - 1

                # Calculate mIoU. Do many-to-one matching if k > self.num_gt_classes.
                if k == self.num_gt_classes:
                    results.append(
                        (
                            k,
                            k,
                            self.compute_miou(
                                gt,
                                pred_flattened,
                                k,
                                self.num_gt_classes,
                                many_to_one=False,
                            ),
                        )
                    )
                else:
                    results.append(
                        (
                            k,
                            k,
                            self.compute_miou(
                                gt,
                                pred_flattened,
                                k,
                                self.num_gt_classes,
                                many_to_one=True,
                            ),
                        )
                    )
                    results.append(
                        (
                            k,
                            f"{k}_prec",
                            self.compute_miou(
                                gt,
                                pred_flattened,
                                k,
                                self.num_gt_classes,
                                many_to_one=True,
                                precision_based=True,
                            ),
                        )
                    )
            return results
