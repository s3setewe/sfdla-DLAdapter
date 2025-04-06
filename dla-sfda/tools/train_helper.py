"""
File: dla-sfda/tools/train_helper.py
Description: Functions and Classes for Training and Optimizing
"""
from typing import Any, Dict, List

import torch
from detectron2.data import build_detection_train_loader
from torch.utils.data import Sampler

from model.CustomDatasetMapper import CustomDatasetMapper


def build_custom_optimizer(cfg: Any, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Description:
        builds a custom optimizer for training a model with specific learning rates for different parameter groups
    Params:
        cfg: Configuration object containing solver settings like learning rates, momentum, and weight decay
        model (torch.nn.Module): The model whose parameters will be optimized
    Return:
        torch.optim.Optimizer: Configured optimizer instance
    """
    backbone_params = []
    roi_head_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        elif "roi_heads" in name:
            roi_head_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": cfg.SOLVER.BACKBONE_LR},
            {"params": roi_head_params, "lr": cfg.SOLVER.ROI_HEAD_LR},
            {"params": other_params, "lr": cfg.SOLVER.BASE_LR},
        ],
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    return optimizer


def compute_entropy_metrics(cfg: Any, model: torch.nn.Module) -> Dict[str, float]:
    """
    Description:
        Computes entropy metrics for a model on the training dataset
    Params:
        cfg: configuration object
        model (torch.nn.Module): model for feature extraction and prediction
    Return:
        Dict[str, float]: dictionary mapping image IDs to their corresponding mean entropy values
    """
    data_loader = build_detection_train_loader(
        cfg,
        mapper=CustomDatasetMapper.from_config(cfg, is_train=True),
        num_workers=0
    )
    num_images = len(data_loader.dataset.dataset.dataset)
    print(f"num_images: {num_images}")
    model.eval()
    metrics = {}
    for data, iteration in zip(data_loader, range(0, num_images)):
        print(f"iteration: {iteration}")
        image_id = data[0]['image_id']
        with torch.no_grad():
            dynamic_teacher_features, dynamic_teacher_proposals, _, dynamic_teacher_results, _ = model(
                data, mode="train", aug="weak"
            )
            dt_box_features = model.roi_heads._shared_roi_transform(
                [dynamic_teacher_features['res4']],
                [dynamic_teacher_proposals[0].proposal_boxes]
            )
            dt_roih_logits = model.roi_heads.box_predictor(
                dt_box_features.mean(dim=[2, 3])
            )[0]  # shape: [N, C]

        probs = torch.softmax(dt_roih_logits, dim=-1)  # [N, C]
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)  # [N]
        entropy_mean = entropy.mean().item()  # Skalar-Wert
        metrics[str(image_id)] = entropy_mean

    return metrics


class SortedSampler(Sampler):
    """
    Description:
        Sampler that sorts the dataset based on a metric (e.g. entropy) before sampling.
    Params:
        dataset (List[Dict]): list of dataset dictionaries
        metrics (Dict[str, float]): dictionary mapping image_id to metric values
        sort_ascending (bool): sort in ascending order
    """
    def __init__(self, dataset: List[Dict], metrics: Dict[str, float], sort_ascending: bool = True):
        self.dataset = dataset
        self.metrics = metrics
        self.sort_ascending = sort_ascending
        self.indices = self.sort_indices()

    def sort_indices(self) -> List[int]:
        indexed_metrics = []
        for idx, data_dict in enumerate(self.dataset):
            image_id = data_dict['image_id']
            entropy = self.metrics.get(str(image_id), 0.0)
            indexed_metrics.append((idx, entropy))

        sorted_indices = sorted(
            indexed_metrics,
            key=lambda x: x[1],
            reverse=not self.sort_ascending
        )
        return [idx for idx, _ in sorted_indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.dataset)
