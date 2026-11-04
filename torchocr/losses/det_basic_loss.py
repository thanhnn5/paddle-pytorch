import torch
from torch import nn
import torch.nn.functional as F


class BalanceLoss(nn.Module):
    def __init__(self,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 negative_ratio=3,
                 return_origin=False,
                 eps=1e-6,
                 **kwargs):
        """
               The BalanceLoss for Differentiable Binarization text detection
               args:
                   balance_loss (bool): whether balance loss or not, default is True
                   main_loss_type (str): can only be one of ['CrossEntropy','DiceLoss',
                       'Euclidean','BCELoss', 'MaskL1Loss'], default is  'DiceLoss'.
                   negative_ratio (int|float): float, default is 3.
                   return_origin (bool): whether return unbalanced loss or not, default is False.
                   eps (float): default is 1e-6.
               """
        super(BalanceLoss, self).__init__()
        self.balance_loss = balance_loss
        self.main_loss_type = main_loss_type
        self.negative_ratio = negative_ratio
        self.return_origin = return_origin
        self.eps = eps

        if self.main_loss_type == "CrossEntropy":
            self.loss = nn.CrossEntropyLoss()
        elif self.main_loss_type == "Euclidean":
            self.loss = nn.MSELoss()
        elif self.main_loss_type == "DiceLoss":
            self.loss = DiceLoss(self.eps)
        elif self.main_loss_type == "BCELoss":
            self.loss = BCELoss(reduction='none')
        elif self.main_loss_type == "MaskL1Loss":
            self.loss = MaskL1Loss(self.eps)
        else:
            loss_type = [
                'CrossEntropy', 'DiceLoss', 'Euclidean', 'BCELoss', 'MaskL1Loss'
            ]
            raise Exception(
                "main_loss_type in BalanceLoss() can only be one of {}".format(
                    loss_type))

    def forward(self, pred, gt, mask=None):
        """
        The BalanceLoss for Differentiable Binarization text detection
        args:
            pred (variable): predicted feature maps.
            gt (variable): ground truth feature maps.
            mask (variable): masked maps.
        return: (variable) balanced loss
        """
        positive = gt * mask
        negative = (1 - gt) * mask

        positive_count = int(positive.sum())
        negative_count = int(
            min(negative.sum(), positive_count * self.negative_ratio))
        loss = self.loss(pred, gt, mask=mask)

        if not self.balance_loss:
            return loss

        positive_loss = positive * loss
        negative_loss = negative * loss
        negative_loss = torch.reshape(negative_loss, shape=[-1])
        if negative_count > 0:
            sort_loss, _ = negative_loss.sort(descending=True)
            negative_loss = sort_loss[:negative_count]
            balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
                positive_count + negative_count + self.eps)
        else:
            balance_loss = positive_loss.sum() / (positive_count + self.eps)
        if self.return_origin:
            return balance_loss, loss

        return balance_loss


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask, weights=None):
        """
        DiceLoss function.
        """

        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = torch.sum(pred * gt * mask)

        union = torch.sum(pred * mask) + torch.sum(gt * mask) + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        """
        Mask L1 Loss
        """
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        loss = torch.mean(loss)
        return loss


class BCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, label, mask=None, weight=None, name=None):
        loss = F.binary_cross_entropy(input, label, reduction=self.reduction)
        return loss


class MaskedFocalLoss(nn.Module):
    """Binary focal loss averaged over valid (mask=1) pixels.

    Port of PaddleOCR ``ppocr/losses/det_basic_loss.py::MaskedFocalLoss``
    (PP-OCRv6). ``pred`` is a post-sigmoid probability map (DBHead output), so
    it is clamped and inverted back to a logit before the stable
    BCE-with-logits form, mirroring Paddle's ``F.sigmoid_focal_loss``:

        loss = BCEWithLogits(logit, gt) * alpha_t * (1 - p_t) ** gamma
    """

    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-6):
        super(MaskedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, gt, mask):
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)
        logit = torch.log(pred / (1.0 - pred))

        ce = F.binary_cross_entropy_with_logits(logit, gt, reduction='none')
        p = torch.sigmoid(logit)
        p_t = p * gt + (1 - p) * (1 - gt)
        alpha_t = self.alpha * gt + (1 - self.alpha) * (1 - gt)
        loss = alpha_t * ce * torch.pow(1 - p_t, self.gamma)

        return (loss * mask).sum() / (mask.sum() + self.eps)


class DiceFocalLoss(nn.Module):
    """DiceLoss + MaskedFocalLoss (PP-OCRv6 shrink-map loss).

    Drop-in replacement for BalanceLoss: same ``forward(pred, gt, mask)``
    signature, returns a scalar.
    """

    def __init__(self, dice_weight=1.0, focal_weight=1.0, focal_alpha=0.75,
                 focal_gamma=2.0, eps=1e-6):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(eps=eps)
        self.focal_loss = MaskedFocalLoss(alpha=focal_alpha, gamma=focal_gamma, eps=eps)

    def forward(self, pred, gt, mask=None):
        loss_dice = self.dice_loss(pred, gt, mask)
        loss_focal = self.focal_loss(pred, gt, mask)
        return self.dice_weight * loss_dice + self.focal_weight * loss_focal
