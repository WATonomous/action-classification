from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torchvision.ops import sigmoid_focal_loss
import warnings

class wBCE():
    """
    Implementation of loss in Argus++ (https://arxiv.org/pdf/2201.05290.pdf)
    """
    def __init__(self, action_class_counts, total_boxes):
        self.eps = 0.5
        # weight balancing frequency of a class' occurrence relative to other classes
        self.freq_weight = torch.zeros(len(action_class_counts), device=torch.cuda.current_device())
        # weight balancing instances where a class appears vs instances it does not appear
        self.pn_weight = torch.zeros(len(action_class_counts), device=torch.cuda.current_device())

        for action_id, count in action_class_counts.items():
            self.freq_weight[action_id] = 1 / count
            self.pn_weight[action_id] = (total_boxes - count) / count
        self.freq_weight = len(action_class_counts) * (self.freq_weight / torch.sum(self.freq_weight))
        #self.freq_weight = F.softmax(self.freq_weight, dim=0)
        #self.pn_weight = F.softmax(self.pn_weight, dim=0)

        #self.freq_weight = self.freq_weight - self.freq_weight.min()
        #self.freq_weight = self.freq_weight / self.freq_weight.max()
        #self.freq_weight  = self.freq_weight + self.eps

        #self.pn_weight = self.pn_weight - self.pn_weight.min()
        #self.pn_weight = self.pn_weight / self.pn_weight.max()
        #self.pn_weight  = self.pn_weight + self.eps
    
        # idea: use log scale for weight frequencies
        self.freq_weight = torch.log(self.freq_weight + 1)
        self.pn_weight = torch.log(self.pn_weight)
    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        # sigmoid + bce that is numerically stable
        return F.binary_cross_entropy_with_logits(logits, targets, 
                                weight = self.freq_weight, pos_weight = self.pn_weight, reduction = 'mean')


def max_reduction_sigmoid_focal_criterion(logits, targets):
    loss = sigmoid_focal_loss(logits,targets, reduction = "none").max(dim =1).values
    return torch.mean(loss, dim = 0)

def max_reduction_sigmoid_focal():
    return max_reduction_sigmoid_focal_criterion, torch.sigmoid

def sum_reduction_sigmoid_focal_criterion(logits, targets):
    loss = sigmoid_focal_loss(logits,targets, reduction = "none").sum(dim =1)
    return torch.mean(loss, dim = 0)

def sum_reduction_sigmoid_focal():
    return sum_reduction_sigmoid_focal_criterion, torch.sigmoid


def custom_sigmoid_focal_criterion(logits, targets):
    loss = sigmoid_focal_loss(logits,targets, reduction = "none").sum(dim =1)
    pos_nums = targets.sum(dim = 1)
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

def softmax_func(logits):

    return F.softmax(logits, dim = 1)

def bce_softmax_criterion(logits, targets):
    logits = softmax_func(logits)
    targets /= torch.sum(targets, dim = 1).view(targets.shape[0],1).expand(targets.shape[0], targets.shape[1])
    targets = torch.nan_to_num(targets)
    return F.cross_entropy(logits, targets)

def bce_softmax():
    return bce_softmax_criterion, softmax_func

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
