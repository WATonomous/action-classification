import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import warnings

class FocalLoss():
    """
    modified from https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
    """
    def __init__(self, action_class_counts):
        """
        We set alpha to the inverse frequency of each class id label.
        """
        self.alpha = [0]*len(action_class_counts)
        total = sum(action_class_counts.values())
        for action_id, count in action_class_counts.items():
            self.alpha[action_id] = total / count
        self.alpha = torch.Tensor(self.alpha)

    def compute_focal_loss(
        self,
        input_logits: torch.Tensor,
        target: torch.Tensor,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        r"""Criterion that computes Focal loss.

        According to :cite:`lin2018focal`, the Focal loss is computed as follows:
        .. math::
            \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
        Where:
        - :math:`p_t` is the model's estimated probability for each class.
        Args:
            input_logits: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
            target: one-hot labels tensor with shape :math:`(N, C *)`.
            alpha: Weighting factor for each class (already initialized at class instatiation).
            gamma: Focusing parameter :math:`\gamma >= 0`.
            reduction: Specifies the reduction to apply to the
            output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the sum of the output will be divided by
            the number of elements in the output, ``'sum'``: the output will be
            summed.
        Return:
            the computed loss.
        """

        if not isinstance(input_logits, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input_logits)}")

        if not len(input_logits.shape) >= 2:
            raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input_logits.shape}")

        if input_logits.size(0) != target.size(0):
            raise ValueError(f'Expected input batch_size ({input_logits.size(0)}) to match target batch_size ({target.size(0)}).')

        n = input_logits.size(0)
        out_size = (n,) + input_logits.size()[1:]
        if target.size()[1:] != input_logits.size()[1:]:
            raise ValueError(f'Expected target size {out_size}, got {target.size()}')

        if not input_logits.device == target.device:
            raise ValueError(f"input and target must be in the same device. Got: {input_logits.device} and {target.device}")

        # compute softmax over the classes axis
        input_soft: torch.Tensor = F.softmax(input_logits, dim=1)
        log_input_soft: torch.Tensor = F.log_softmax(input_logits, dim=1)

        # create the labels one hot tensor
        target_one_hot = target

        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1.0, gamma)
        self.alpha = self.alpha.to(weight)
        focal = -self.alpha.expand(weight.shape[0],weight.shape[1]) * weight * log_input_soft
        loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

        if reduction == 'none':
            loss = loss_tmp
        elif reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {reduction}")
        return loss



def ava_pose_softmax_func(logits):
    pose_logits = nn.Softmax(dim=1)(logits[:, :13])
    interact_logits = nn.Sigmoid()(logits[:, 13:])
    logits = torch.cat([pose_logits, interact_logits], dim=1)
    logits = torch.clamp(logits, min=0., max=1.)
    return logits


def ava_pose_softmax_criterion(logits, targets):
    logits = ava_pose_softmax_func(logits)
    return F.binary_cross_entropy(logits, targets)


def ava_criterion(pose_softmax=False):
    if pose_softmax:
        return ava_pose_softmax_criterion, ava_pose_softmax_func
    return nn.BCEWithLogitsLoss(), nn.Sigmoid()
