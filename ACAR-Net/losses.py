from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torchvision.ops import sigmoid_focal_loss
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
            self.alpha[action_id] = .25 #tf sigmoid focal crossentropy says do this #total / count 
        self.alpha = torch.Tensor(self.alpha)
        # now, normalize between 0 and 1
        #self.alpha = F.normalize(self.alpha, p  = 1, dim = 0)
        #self.alpha -= self.alpha.min()
        #self.alpha /= self.alpha.max()
        #if (torch.isnan(self.alpha).sum() > 0):
        #    print('alpha contains a Nan')
        #    raise ValueError

    def compute_focal_loss(self,
        input_logits: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean'
        ) -> torch.Tensor:

        return self.stupid_focal_loss(input_logits, target, reduction=reduction)

    def stupid_focal_loss(self, 
        input_logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = 'mean'
        ) -> torch.Tensor:

        """
        0 'Red', 1 'Amber', 2 'Green', 3 'MovAway', 4 'MovTow', 5 'Mov', 6 'Rev', 7 'Brake',
        8 'Stop', 9 'IncatLft', 10 'IncatRht', 11 'HazLit', 12 'TurLft', 13 'TurRht', 14 'MovRht',
        15 'MovLft', 16 'Ovtak', 17 'Wait2X', 18 'XingFmLft', 19 'XingFmRht', 20 'Xing', 21 'PushObj']
        """

        part1 = F.softmax(input_logits[:13], dim=1)
        part2 = F.sigmoid(input_logits[13:])
        p = torch.cat((part1, part2), dim =1)
        
        ce_loss = F.binary_cross_entropy(p, targets, reduction="none")
        ce_loss = ce_loss.view(targets.shape[0],1).expand(targets.shape[0], targets.shape[1])
        
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            # 0.25 for positives and 0.75 for negatives
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

            loss = alpha_t * loss

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def modified_focal_loss(self,
        input_logits: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean'
        ) -> torch.Tensor:

        """
        0 'Red', 1 'Amber', 2 'Green', 3 'MovAway', 4 'MovTow', 5 'Mov', 6 'Rev', 7 'Brake',
        8 'Stop', 9 'IncatLft', 10 'IncatRht', 11 'HazLit', 12 'TurLft', 13 'TurRht', 14 'MovRht',
        15 'MovLft', 16 'Ovtak', 17 'Wait2X', 18 'XingFmLft', 19 'XingFmRht', 20 'Xing', 21 'PushObj']
        """

        traffic_input = input_logits[:3]
        traffic_target = target[:3]
        traffic_loss = self._compute_focal_loss_softmax(traffic_input, traffic_target, reduction=reduction)
        
        mov_distance_input = input_logits[3:6]
        mov_distance_target = target[3:6]
        mov_distance_loss = self._compute_focal_loss_softmax(mov_distance_input, mov_distance_target, reduction=reduction)
        
        incat_input = input_logits[9:11]
        incat_target = target[9:11]
        incat_loss = self._compute_focal_loss_softmax(incat_input, incat_target, 
                                                        reduction=reduction)
        
        tur_direction_input = input_logits[12:14]
        tur_direction_target = target[12:14]
        tur_loss = self._compute_focal_loss_softmax(tur_direction_input, tur_direction_target, 
                                                        reduction=reduction)
        
        mov_direction_input = input_logits[14:16]
        mov_direction_target = target[14:16]
        mov_direction_loss = self._compute_focal_loss_softmax(mov_direction_input, mov_direction_target, 
                                                        reduction=reduction)

        xing_direction_input = input_logits[18:21]
        xing_direction_target = target[18:21]
        xing_direction_loss = self._compute_focal_loss_softmax(xing_direction_input, xing_direction_target,
                                                        reduction=reduction)

        other_input = torch.cat((input_logits[6:9],input_logits[11:12], input_logits[16:18], input_logits[21:]))
        other_target = torch.cat((target[6:9],target[11:12], target[16:18], target[21:]))
        other_loss = sigmoid_focal_loss(other_input, other_target, reduction = reduction)

        all_losses = torch.tensor((traffic_loss, mov_distance_loss, incat_loss, tur_loss, 
                                    mov_direction_loss, xing_direction_loss, other_loss), requires_grad = True)
        all_losses = all_losses.to(torch.cuda.current_device())

        # this is valid because all operations are means
        return torch.mean(all_losses)
    
    def compute_focal_loss_softmax(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
    ):
        """

        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        p = F.softmax(inputs, dim = 1)

        targets /= torch.sum(targets, dim = 1).view(targets.shape[0],1).expand(targets.shape[0], targets.shape[1])
        targets = torch.nan_to_num(targets)
        
        ce_loss = F.cross_entropy(p, targets, reduction="none")
        ce_loss = ce_loss.view(targets.shape[0],1).expand(targets.shape[0], targets.shape[1])
        
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            # 0.25 for positives and 0.75 for negatives
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

            loss = alpha_t * loss

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def deprecated_compute_focal_loss_softmax(
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
            target: labels tensor with shape :math:`(N, C *)`.
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

        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1.0, gamma)
        self.alpha = self.alpha.to(weight)
        focal = -self.alpha.expand(weight.shape[0],weight.shape[1]) * weight * log_input_soft
        loss_tmp = torch.einsum('bc...,bc...->b...', (target, focal))

        if reduction == 'none':
            loss = loss_tmp
        elif reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {reduction}")
        return loss

    def _deprecated_compute_focal_loss_sigmoid(
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

        ce = F.binary_cross_entropy(input_logits, target)
        input_sig: torch.Tensor = F.sigmoid(input_logits)
        log_input_sig: torch.Tensor = F.logsigmoid(input_logits)

        # compute the actual focal loss
        weight = torch.pow(-input_sig + 1.0, gamma)
        self.alpha = self.alpha.to(weight)
        focal = -self.alpha.expand(weight.shape[0], weight.shape[1]) * weight * log_input_sig
        loss_tmp = torch.einsum('bc...,bc...->b...', (target, focal))

        if reduction == 'none':
            loss = loss_tmp
        elif reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {reduction}")
        return loss

    def other_modified_focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
    ):
        """
        Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.


        0 'Red', 1 'Amber', 2 'Green', 3 'MovAway', 4 'MovTow', 5 'Mov', 6 'Rev', 7 'Brake',
        8 'Stop', 9 'IncatLft', 10 'IncatRht', 11 'HazLit', 12 'TurLft', 13 'TurRht', 14 'MovRht',
        15 'MovLft', 16 'Ovtak', 17 'Wait2X', 18 'XingFmLft', 19 'XingFmRht', 20 'Xing', 21 'PushObj']

        """

        traffic_input = inputs[:, :3]
        traffic_loss = F.softmax(traffic_input, dim = 1)
        
        mov_distance_input = inputs[:, 3:6]
        mov_distance_loss = F.softmax(mov_distance_input, dim = 1)
        
        incat_input = inputs[:, 9:11]
        incat_loss = F.softmax(incat_input, dim = 1)
        
        tur_direction_input = inputs[:, 12:14]
        tur_loss = F.softmax(tur_direction_input, dim = 1)
        
        mov_direction_input = inputs[:, 14:16]
        mov_direction_loss = F.softmax(mov_direction_input, dim = 1)

        xing_direction_input = inputs[:, 18:21]
        xing_direction_loss = F.softmax(xing_direction_input, dim = 1)

        other_input = torch.cat((inputs[:, 6:9],inputs[:, 11:12], inputs[:, 16:18], inputs[:, 21:]), dim = 1)
        other_loss = torch.sigmoid(other_input)

        p = torch.cat((traffic_loss, mov_distance_loss, other_loss[:, 0:3], incat_loss, other_loss[:, 3:4], tur_loss, 
                    mov_direction_loss, other_loss[:, 4:6], xing_direction_loss, other_loss[:, 6:]), dim = 1)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss


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

def max_reduction_sigmoid_focal(pose_softmax = False):
    return max_reduction_sigmoid_focal_criterion, torch.sigmoid

def sum_reduction_sigmoid_focal_criterion(logits, targets):
    loss = sigmoid_focal_loss(logits,targets, reduction = "none").sum(dim =1)
    return torch.mean(loss, dim = 0)

def sum_reduction_sigmoid_focal(pose_softmax = False):
    return sum_reduction_sigmoid_focal_criterion, torch.sigmoid


def custom_sigmoid_focal_criterion(logits, targets):
    loss = sigmoid_focal_loss(logits,targets, reduction = "none").sum(dim =1)
    pos_nums = targets.sum(dim = 1)
    pos_nums[pos_nums == 0] = 1
    loss = torch.mean(loss / pos_nums)
    return loss


def custom_sigmoid_focal(pose_softmax = False):
    return custom_sigmoid_focal_criterion, torch.sigmoid


def bce_sigmoid_criterion(logits, targets):
    logits = torch.sigmoid(logits)
    return F.binary_cross_entropy(logits, targets)

def bce_sigmoid(pose_softmax = False):
    return bce_sigmoid_criterion, torch.sigmoid

def softmax_func(logits):

    return F.softmax(logits, dim = 1)

def bce_softmax_criterion(logits, targets):
    logits = softmax_func(logits)
    targets /= torch.sum(targets, dim = 1).view(targets.shape[0],1).expand(targets.shape[0], targets.shape[1])
    targets = torch.nan_to_num(targets)
    return F.cross_entropy(logits, targets)

def bce_softmax(pose_softmax = False):
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
