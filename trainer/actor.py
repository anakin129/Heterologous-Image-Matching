import torch
from utils.tensor_dict import TensorDict
import numpy as np

class BaseActor:
    """
    Base class for actor.The actor class handles the passing of the data through the network
    and calculate the loss 
    """
    def __init__(self,net,objective):
        self.net=net
        self.objective=objective
    def __call__(self,data:TensorDict):
        """ Called in each training iteration. Should pass in input data through the network, calculate the loss, and
        return the training stats for the input data
        args:
            data - A TensorDict containing all the necessary data blocks.

        returns:
            loss    - loss for the input data
            stats   - a dict containing detailed losses
        """
        raise NotImplementedError
    def to(self,device):
        self.net.to(device)
    def train(self,mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)
        self.objective.train(mode)
    def eval(self):
        self.train(False)

class TranstActor(BaseActor):
    def __call__(self,data):
        outputs=self.net(data['search_images'],data['template_images'])

        #generate labels
        targets=[]
        targets_origin=data['search_anno']
        for i in range(len(targets_origin)):
            h,w=data['search_images'][i][0].shape
            target_origin=targets_origin[i]
            target={}
            target_origin=target_origin.reshape(1,-1)
            target_origin[0][0] += target_origin[0][2] / 2
            target_origin[0][0] /= w
            target_origin[0][1] += target_origin[0][3] / 2
            target_origin[0][1] /= h
            target_origin[0][2] /= w
            target_origin[0][3] /= h
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            targets.append(target)
        
        # compute loss
        # outputs:(center_x, center_y, width, height)
        loss_dict=self.objective(outputs,targets)
        weight_dict=self.objective.weight_dict
        losses=sum(loss_dict[k]*weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'iou': loss_dict['iou'].item()
                 }

        return losses, stats
