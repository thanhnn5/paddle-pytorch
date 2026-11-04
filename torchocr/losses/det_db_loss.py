import torch
from torch import nn

from .det_basic_loss import BalanceLoss, MaskL1Loss, DiceLoss, DiceFocalLoss


class DBLoss(nn.Module):
    """
    Differentiable Binarization (DB) Loss Function
    args:
        param (dict): the super paramter for DB Loss

    PP-OCRv6 additions (ported from PaddleOCR ppocr/losses/det_db_loss.py):
      * main_loss_type='DiceFocalLoss': shrink & binary maps use Dice + masked
        focal loss instead of OHEM-balanced Dice.
      * aux_weight_p4/p3/p2 > 0: deep supervision on the auxiliary DBHead
        outputs ``aux_maps_p4/p3/p2`` (emitted in train mode when the head is
        built with ``aux_in_channels > 0``).
    """

    def __init__(self,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 alpha=5,
                 beta=10,
                 ohem_ratio=3,
                 eps=1e-6,
                 aux_weight_p4=0.0,
                 aux_weight_p3=0.0,
                 aux_weight_p2=0.0,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 dice_weight=1.0,
                 focal_weight=1.0,
                 **kwargs):
        super(DBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = MaskL1Loss(eps=eps)
        if main_loss_type == 'DiceFocalLoss':
            self.bce_loss = DiceFocalLoss(
                dice_weight=dice_weight,
                focal_weight=focal_weight,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                eps=eps)
            self.dice_loss = self.bce_loss
        else:
            # Original behaviour (BalanceLoss also validates main_loss_type).
            self.bce_loss = BalanceLoss(
                balance_loss=balance_loss,
                main_loss_type=main_loss_type,
                negative_ratio=ohem_ratio)
            self.dice_loss = DiceLoss(eps=eps)
        self.aux_weight_p4 = aux_weight_p4
        self.aux_weight_p3 = aux_weight_p3
        self.aux_weight_p2 = aux_weight_p2

    def forward(self, predicts, labels):
        predict_maps = predicts['res']
        label_threshold_map, label_threshold_mask, label_shrink_map, label_shrink_mask = labels[1:]
        shrink_maps = predict_maps[:, 0, :, :]
        threshold_maps = predict_maps[:, 1, :, :]
        binary_maps = predict_maps[:, 2, :, :]

        loss_shrink_maps = self.bce_loss(shrink_maps, label_shrink_map,label_shrink_mask)
        loss_threshold_maps = self.l1_loss(threshold_maps, label_threshold_map,label_threshold_mask)
        loss_binary_maps = self.dice_loss(binary_maps, label_shrink_map, label_shrink_mask)
        loss_shrink_maps = self.alpha * loss_shrink_maps
        loss_threshold_maps = self.beta * loss_threshold_maps
        # CBN loss
        if 'distance_maps' in predicts.keys():
            distance_maps = predicts['distance_maps']
            cbn_maps = predicts['cbn_maps']
            cbn_loss = self.bce_loss(cbn_maps[:, 0, :, :], label_shrink_map,
                                     label_shrink_mask)
        else:
            dis_loss = torch.tensor([0.], device=shrink_maps.device)
            cbn_loss = torch.tensor([0.], device=shrink_maps.device)

        loss_all = loss_shrink_maps + loss_threshold_maps \
                   + loss_binary_maps
        losses = {'loss': loss_all+ cbn_loss, \
                  "loss_shrink_maps": loss_shrink_maps, \
                  "loss_threshold_maps": loss_threshold_maps, \
                  "loss_binary_maps": loss_binary_maps, \
                  "loss_cbn": cbn_loss}

        # PP-OCRv6 auxiliary deep-supervision losses
        for aux_key, aux_w in [
            ('aux_maps_p4', self.aux_weight_p4),
            ('aux_maps_p3', self.aux_weight_p3),
            ('aux_maps_p2', self.aux_weight_p2),
        ]:
            if aux_w > 0 and aux_key in predicts:
                aux_maps = predicts[aux_key]
                l_shrink = self.alpha * self.bce_loss(
                    aux_maps[:, 0, :, :], label_shrink_map, label_shrink_mask)
                l_threshold = self.beta * self.l1_loss(
                    aux_maps[:, 1, :, :], label_threshold_map, label_threshold_mask)
                l_binary = self.dice_loss(
                    aux_maps[:, 2, :, :], label_shrink_map, label_shrink_mask)
                aux_loss = l_shrink + l_threshold + l_binary
                losses['loss_{}'.format(aux_key)] = aux_loss
                losses['loss'] = losses['loss'] + aux_w * aux_loss

        return losses
