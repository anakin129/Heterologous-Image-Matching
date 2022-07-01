from utils.config import cfg
import torch
from torch import nn
import torch.nn.functional as F
from .transt_backbone import build_backbone, build_double_backbone
from utils.nestedtensor import NestedTensor,nested_tensor_from_tensor
from .model_constructor import model_constructor
from .backbone_yolo import build_yolo_double_backbone
from .featurefusion_network import build_featurefusion_network
import time
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TransT(nn.Module): 
    """ This is the TransT module that performs single object tracking """
    def __init__(self,args,backbone, featurefusion_network, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        hidden_dim = featurefusion_network.d_model
        self.args=args
        if args.model_id==1:
            self.backbone=backbone
            num_channels=self.backbone.num_channels
        else:
            self.backbone_A=backbone[0]
            self.backbone_B=backbone[1]
            num_channels=self.backbone_A.num_channels
        
        self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)

        self.featurefusion_network = featurefusion_network
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        self.backbone_time=[]
        self.fusion_time=[]
        self.cls_reg_time=[]
        self.fps=[]
        
    def forward(self, search, template):
        """The forward expects a NestedTensor, which consists of:
               - search.tensors: batched images, of shape [batch_size x 3 x H_search x W_search]
               - search.mask: a binary mask of shape [batch_size x H_search x W_search], containing 1 on padded pixels
               - template.tensors: batched images, of shape [batch_size x 3 x H_template x W_template]
               - template.mask: a binary mask of shape [batch_size x H_template x W_template], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits for all feature vectors.
                                Shape= [batch_size x num_vectors x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all feature vectors, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image.

        """
        b=search.shape[0]
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
        s=time.time()
        sf=time.time()
        feature_search, pos_search = self.backbone(search) if self.args.model_id==1 else self.backbone_A(search)
        feature_template, pos_template = self.backbone(template) if self.args.model_id==1 else self.backbone_B(template)
        self.backbone_time.append(time.time()-s)
        s=time.time()
        src_search, mask_search= feature_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        hs,memory_template,memory_search= self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])
        self.fusion_time.append(time.time()-s)
        s=time.time()
        outputs_class = self.class_embed(hs) # torch.Size([1, 8, 1024, 2])
        outputs_coord = self.bbox_embed(hs).sigmoid()  # torch.Size([1, 8, 1024, 4])
        self.cls_reg_time.append(time.time()-s)
        self.fps.append(time.time()-sf)
        # outputs_class[-1].shape:torch.Size([8, 1024, 2])
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

@model_constructor
def transt_resnet50(args):
    num_classes=1
    if args.model_id==1:
        backbone=build_double_backbone(args)
        backbone_net=backbone
    else:
        backbone_A,backbone_B=build_double_backbone(args)
        backbone_net=tuple((backbone_A,backbone_B))
    featurefusion_network= build_featurefusion_network(args)
    model=TransT(args,backbone_net,featurefusion_network,num_classes=num_classes)
    device=torch.device('cuda' if cfg.TRAIN.use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model