import torch
from torch import nn
import torch.nn.functional as F
from utils.config import cfg
from typing import Dict, List
from utils.nestedtensor import NestedTensor
from .resnet50 import resnet50
from .position_encoding import build_position_encoding
from .backbone_yolo import YOLObackbone
from .mobilenetv3 import mobilenet_backbone

class BackboneBase(nn.Module):
    def __init__(self,backbone,num_channels):
        super().__init__()
        self.body=backbone
        self.num_channels=num_channels

    def forward(self,tensor_list:NestedTensor):
        xs=self.body(tensor_list.tensors)
        out:Dict[str,NestedTensor]={}
        for name,x in xs.items():
            m=tensor_list.mask
            assert m is not None
            mask=F.interpolate(m[None].float(),size=x.shape[-2:]).to(torch.bool)[0]
            out[name]=NestedTensor(x,mask)
        return out

class Backbone(BackboneBase):
    def __init__(self,args):
        if args.backbone_id==0:
            if args.multi_scale:
                backbone=resnet50(output_layers=['layer2','layer3'],pretrained=False)
                num_channels=1024
            else:
                backbone=resnet50(output_layers=['layer3'],pretrained=False)
                num_channels=1024
                
        elif args.backbone_id==1:
            backbone=mobilenet_backbone(args)
            num_channels=960
        elif args.backbone_id==2:
            backbone=YOLObackbone(args,'yolov5m.pt')
            num_channels=768
            
        else:
            raise NotImplementedError

        super().__init__(backbone,num_channels)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []  #¡¡var: type = value
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos

def build_backbone(args):
    position_embedding=build_position_encoding()
    backbone=Backbone(args)
    model=Joiner(backbone,position_embedding)
    model.num_channels=backbone.num_channels
    return model
    
def build_double_backbone(args):
    position_embedding=build_position_encoding()
    if args.model_id==0:           ## low-sep-high-sharing
        backbone_A=Backbone(args=args)
        backbone_B=Backbone(args=args)
        if args.backbone_id==0:    ## resnet
            backbone_B.body.layer2=backbone_A.body.layer2
            backbone_B.body.layer3=backbone_A.body.layer3
        elif args.backbone_id==1:  ##mobilenetv3
            backbone_B.body.layer2=backbone_A.body.layer2
            backbone_A.body.layer2[4].block[1][0].stride=(1,1)
            backbone_A.body.layer2[10].block[1][0].stride=(1,1)
        elif args.backbone_id==2:  ##yolov5
            backbone_B.body.layer2=backbone_A.body.layer2
            backbone_B.body.layer3=backbone_A.body.layer3
            backbone_B.body.layer4=backbone_A.body.layer4
            backbone_A.body.layer3[0].conv.stride=(1,1)
            backbone_A.body.layer4[0].conv.stride=(1,1)
        else:
            raise NotImplementedError
        net_A=Joiner(backbone_A,position_embedding)
        net_B=Joiner(backbone_B,position_embedding)
        net_A.num_channels=backbone_A.num_channels
        net_B.num_channels=backbone_B.num_channels
        return net_A,net_B
    elif args.model_id==1:
        backbone=Backbone(args=args)
        if args.backbone_id==1:
            backbone.body.layer2[4].block[1][0].stride=(1,1)
            backbone.body.layer2[10].block[1][0].stride=(1,1)
        elif args.backbone_id==2:
            backbone.body.layer3[0].conv.stride=(1,1)
            backbone.body.layer4[0].conv.stride=(1,1)
        net=Joiner(backbone,position_embedding)
        net.num_channels=backbone.num_channels
        return net
    elif args.model_id==2:
        backbone_A=Backbone(args=args)
        backbone_B=Backbone(args=args)
        if args.backbone_id==1:
            backbone_A.body.layer2[4].block[1][0].stride=(1,1)
            backbone_A.body.layer2[10].block[1][0].stride=(1,1)
            backbone_B.body.layer2[4].block[1][0].stride=(1,1)
            backbone_B.body.layer2[10].block[1][0].stride=(1,1)
        elif args.backbone_id==2:
            backbone_A.body.layer3[0].conv.stride=(1,1)
            backbone_A.body.layer4[0].conv.stride=(1,1)
            backbone_B.body.layer3[0].conv.stride=(1,1)
            backbone_B.body.layer4[0].conv.stride=(1,1)
        net_A=Joiner(backbone_A,position_embedding)
        net_B=Joiner(backbone_B,position_embedding)
        net_A.num_channels=backbone_A.num_channels
        net_B.num_channels=backbone_B.num_channels
        return net_A,net_B
        