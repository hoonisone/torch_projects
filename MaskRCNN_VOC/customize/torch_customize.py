from torchvision.models.detection import roi_heads
from torchvision.models.detection.roi_heads import * 
import torch
import numpy as np

def boundary_mask(mask, filter_size = 1):
    n = filter_size**2
    conv = torch.nn.Conv2d(1, 1, (filter_size, filter_size), bias = False, padding="same").to(mask.device)

    with torch.no_grad():
        conv.weight[0, 0] = 1
    mask = conv(mask)
    mask = torch.abs(mask - np.ceil(n/2)) < n/2-1
    return mask

def my_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs): 
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) 
    """ 
    Arguments: 
        proposals (list[BoxList]) 
        mask_logits (Tensor) 
        targets (list[BoxList]) 

    Return: 
        mask_loss (Tensor): scalar tensor containing the loss 
    """ 

    
    discretization_size = mask_logits.shape[-1] 
    labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)] 
    mask_targets = [ 
        project_masks_on_boxes(m, p, i, discretization_size) 
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs) 
    ] 

    labels = torch.cat(labels, dim=0) 
    mask_targets = torch.cat(mask_targets, dim=0) 

    # torch.mean (in binary_cross_entropy_with_logits) doesn't 
    # accept empty tensors, so handle it separately 
    if mask_targets.numel() == 0: 
        return mask_logits.sum() * 0 

    # Weight
    if BOUNDARY_LOSS == True:
        mask_weights = [boundary_mask(mask_target.unsqueeze(0), BOUNDARY_SIZE).squeeze() for mask_target in mask_targets]
        mask_weights = 1+torch.stack(mask_weights)
    else:
        mask_weights = None

    mask_loss = F.binary_cross_entropy_with_logits( 
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets , weight=mask_weights
    ) 
    return mask_loss 

BOUNDARY_LOSS = False
BOUNDARY_SIZE = 0

MODIFIED = False

def customize_mask_rcnn_roi_heads_loss():
    print("Customized: torchvision.models.detection.roi_heads.maskrcnn_loss")
    roi_heads.maskrcnn_loss = my_maskrcnn_loss

def customize_all():
    if MODIFIED == False:
        customize_mask_rcnn_roi_heads_loss()
        
