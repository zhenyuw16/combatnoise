import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def soft_focal_loss(pred,
        target,
        weight=None,
        gamma=2.0,
        alpha=0.25,
        reduction='mean',
        avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    
    target, target_score = target[0], target[1]
    target_oh = torch.zeros((pred_sigmoid.shape[0], pred.shape[1] + 1)).type_as(pred).to(pred.device)
    target_oh.scatter_(1, target[:,None], 1)
    target_oh = target_oh[:,0:-1]
    target = target[:,None]

    target_soft = (target_oh > 0).float() * target_score[:,None]
    pt = target_soft - pred_sigmoid
    focal_weight = ((1 - alpha) + (2*alpha - 1) * target_soft) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target_soft, reduction='none') * focal_weight

    weight = weight.view(-1,1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class SoftFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(SoftFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * soft_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
