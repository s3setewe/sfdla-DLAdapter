"""
File: dla-sfda/tools/loss_function.py
Description: Different Loss Functions (constrastive loss, distillation loss, entropy loss, soft label loss)
"""

import torch
import torch.nn.functional as F
from typing import Dict


def compute_contrastive_loss(t_features: Dict[str, torch.Tensor], s_features: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
    """
    Description:
        Computes the contrastive loss between teacher and student features
    Params:
        t_features (Dict[str, torch.Tensor]): teacher feature
        s_features (Dict[str, torch.Tensor]): student feature
        temperature (float): temperature parameter for scaling logits
    Return:
        contrastive_loss (torch.Tensor): total contrastive loss.
    """
    total_loss = 0

    for key in t_features.keys():
        t_feat = t_features[key]
        s_feat = s_features[key]

        t_feat = t_feat.to(s_feat.device)  # Ensure teacher features are on the same device as student features

        N, C, H, W = t_feat.shape

        # Transpose Shape: (N*H*W, C)
        t_feat = (t_feat.view(N, C, -1)).permute(0, 2, 1).reshape(-1, C)
        s_feat = (s_feat.view(N, C, -1)).permute(0, 2, 1).reshape(-1, C)

        t_feat_norm = torch.nn.functional.normalize(t_feat, dim=1)
        s_feat_norm = torch.nn.functional.normalize(s_feat, dim=1)

        logits = torch.matmul(s_feat_norm, t_feat_norm.T)
        labels = torch.arange(logits.shape[0], device=logits.device)
        logits = logits / temperature
        loss = torch.nn.functional.cross_entropy(logits, labels)
        total_loss += loss

    return total_loss


def feature_distillation_loss(t_features: Dict[str, torch.Tensor], s_features: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Description:
        Computes distillation loss as the mean squared error (MSE) loss between teacher and student features.
    Params:
        t_features (Dict[str, torch.Tensor]): teacher feature
        s_features (Dict[str, torch.Tensor]): student feature
    Return:
        distillation_loss (torch.Tensor): total distillation_loss (MSE)
    """
    student_device = next(iter(s_features.values())).device
    loss = 0.0
    for key in t_features.keys():
        loss += F.mse_loss(s_features[key], t_features[key].to(student_device))
    return loss

def entropy_loss(student_logits: torch.Tensor) -> torch.Tensor:
    """
    Description:
        Minimizes the entropy of student predictions to encourage confident predictions
    Params:
        student_logits (torch.Tensor): Logits of student model predictions (N, num_classes)
    Return:
        entropy_loss (torch.Tensor): total entropy_loss
    """
    student_probs = F.softmax(student_logits, dim=1)
    ent = -(student_probs * student_probs.log()).sum(dim=1).mean()
    return ent



def soft_label_kl_distillation_loss(student_class_logits: torch.Tensor, selected_teacher_pseudo_results: list) -> torch.Tensor:
    """
    Description:
        computes the KL-divergence loss between student predictions and teacher pseudo-label distributions.
    Params:
        student_class_logits (torch.Tensor): logits of student predictions (N, num_classes).
        selected_teacher_pseudo_results (list): list of objects containing teacher pseudo-labels with soft_cls_distribution attribute.
    Return:
        soft_label_kl_distillation_loss (torch.Tensor): KL-divergence loss as soft_label_loss
    """

    # Check if the teacher pseudo-results have the required soft_cls_distribution attribute
    # If no pseudo-labels are available, return zero loss
    if not hasattr(selected_teacher_pseudo_results[0], 'soft_cls_distribution'):
        return torch.tensor(0.0, device=student_class_logits.device)

    # Extract the teacher's soft class distributions
    # If the teacher's soft distributions are empty, return zero loss
    teacher_distributions = selected_teacher_pseudo_results[0].soft_cls_distribution.to(student_class_logits.device)
    if teacher_distributions.numel() == 0:
        return torch.tensor(0.0, device=student_class_logits.device)

    # Convert student logits to probability distributions using softmax
    student_probs = torch.softmax(student_class_logits, dim=1)

    # Compute KL-divergence between student_probs and teacher_distributions
    # Assumption: The number of student proposals (N) matches the number of teacher proposals (M), If N != M, we slice both distributions to match the smaller size
    N = min(student_probs.size(0), teacher_distributions.size(0))
    student_probs = student_probs[:N]
    teacher_distributions = teacher_distributions[:N]

    # Calculate KL-divergence loss with batch mean reduction
    loss = torch.nn.functional.kl_div(student_probs.log(), teacher_distributions, reduction='batchmean')
    return loss
