import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torchvision.ops import sigmoid_focal_loss
import warnings


def sum_reduction_sigmoid_focal_criterion(logits, targets):
    """
    sigmoid focal loss which sums over the class dimension and then
    takes the mean over the batch dimension. The effect of this is that
    the magnitude of the loss is greater than taking a mean over the entire
    matrix. Additionally, suppose that the model correctly learns to predict
    the absence of common classes, in the case where there is a difficult, uncommon class,
    averaging over the class dimension would result in a lower loss 
    (weighted down by the low loss of common true negative predictions) whereas a sum over the 
    class dimension would not reduce a high loss produced by the incorrect prediction
    of an uncommon class.
    """
    loss = sigmoid_focal_loss(logits, targets, reduction="none").sum(dim=1)
    return torch.mean(loss, dim=0)


def sum_reduction_sigmoid_focal():
    return sum_reduction_sigmoid_focal_criterion, torch.sigmoid


def custom_sigmoid_focal_criterion(logits, targets):
    """
    sigmoid focal loss that divides loss by number of positive classes in the
    ground truth. The effect of this may be that the datapoints with fewer
    positive classes have a larger effect.
    """
    loss = sigmoid_focal_loss(logits, targets, reduction="none").sum(dim=1)
    pos_nums = targets.sum(dim=1)
    pos_nums[pos_nums == 0] = 1
    loss = torch.mean(loss / pos_nums)
    return loss


def custom_sigmoid_focal():
    return custom_sigmoid_focal_criterion, torch.sigmoid


def bce_sigmoid_criterion(logits, targets):
    logits = torch.sigmoid(logits)
    return F.binary_cross_entropy(logits, targets)


def bce_sigmoid():
    return bce_sigmoid_criterion, torch.sigmoid


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
