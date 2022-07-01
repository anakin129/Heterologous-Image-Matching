import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import build_matcher
from .iou import generalized_box_iou,box_cxcywh_to_xyxy
from utils.config import cfg
class SetCriterion(nn.Module):
    """
    This class computes the loss for TransT
    The process happens in two steps:
        1) we compute assignment between ground truth and the outputs of the model
        2) we supervise each pair of matched ground-truth/predict(supervise class and box)
    """
    def __init__(self,num_classes,matcher,weight_dict,eos_coef,losses):
        """
        Create the criterion
        args:
            num_classes:number of object categories,always be 1 for single object tracking.
            matcher:module able to compute a matching between target and proposals
            weight_dict:dict containing as key the names of the losses and as values their relative weight
            eos_coef:relative classification weight applied to the no-object category
            losses:list of all the losses to be applied.See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes=num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight=torch.ones(self.num_classes+1)
        empty_weight[-1]=self.eos_coef
        self.register_buffer('empty_weight',empty_weight)
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self,outputs,targets,indices,num_boxes,log=True):
        """
        Classifiaction loss(entropy-loss)
        targets dicts must contain the key 'labels' containing a tensor of dim
        """
        assert 'pred_logits' in outputs
        src_logits=outputs['pred_logits']
        idx=self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).type(torch.int64)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        # src_logits.shape: torch.Size([8, 1024, 2])
        # target_classes.shape: torch.Size([8, 1024])
        
        
        ###  cross-entropy loss
        ###  loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight) 
       
        ### info-entropy loss
        
        bs,_=target_classes.shape
        src_logits=F.softmax(src_logits,dim=-1)[:,:,0]
        signal=torch.exp(src_logits*(1-target_classes))
        noise=torch.exp(src_logits*target_classes)
        loss_ce= -1 * torch.log(1.1*signal/ (1.1*signal + 0.1*noise+ 1e-10))
        loss_ce=loss_ce.mean()
        
        
        losses = {'loss_ce': loss_ce}

        return losses
        
        
    def loss_boxes(self,outputs,targets,indices,num_boxes):
        """
        Compute the losses related to the bounding boxes,the L1 regression loss and the GIOU loss.
        target dict must contain the key 'boxes' containing a tensor of dim [nb_target_boxes,4].
        The target boxes are expected in format(center_x,center_y,h,w),normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx=self._get_src_permutation_idx(indices)
        src_boxes=outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
      
        giou, iou = generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes))
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        loss_giou = 1 - giou
        iou = iou
        
        ### ingore the large iou samples
        cfg.TRAIN.thresh=0.95
        mask=iou<cfg.TRAIN.thresh
        losses['iou'] = iou.sum() / num_boxes
        loss_giou=loss_giou*mask
        num_boxes=torch.sum(mask==True)
        if num_boxes==0:
            return 0
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        
        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        return losses
        
        
    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs:{'pred_logits':x1,'pred_boxes':x2}
        # targets:list of dicts, such that len(targets) == batch_size.
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the target
        indices = self.matcher(outputs_without_aux, targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)

        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos))
        ## cls_loss:entropy_loss
        ## reg_loss:l1_loss+giou_loss
        return losses

def transt_loss():
    num_classes = 1
    matcher = build_matcher()
    weight_dict = {'loss_ce': 8.334, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    device =torch.device('cuda' if cfg.TRAIN.use_gpu and torch.cuda.is_available() else 'cpu')
    criterion.to(device)
    return criterion
