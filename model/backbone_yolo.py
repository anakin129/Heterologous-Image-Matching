import sys
import os
sys.path.append(os.getcwd())
import torch
from torch import nn
sys.path.append('..')
from .position_encoding import build_position_encoding
from .yolo_backbone.yolov5.models.common import DetectMultiBackend
from utils.nestedtensor import NestedTensor

def get_yoloBackbone(model_size : str):
    
        weights = '/data0/liutianqi/Siamese-triplet/tran/model/yolo_backbone/yolov5/' + model_size
        model = DetectMultiBackend(weights)

        backbone = nn.Sequential( *(list(list(model.model.children())[0].children())[0:10]))

        return backbone

class YOLObackbone(nn.Module):
    def __init__(self, args,model_size : str):
        super().__init__()
        tmp   = get_yoloBackbone(model_size)       
        for i in tmp.parameters():
            i.requires_grad=True
        self.layer1 = tmp[0:3]     
        self.layer2 = tmp[3:5]      
        self.layer3 = tmp[5:7]      
        self.layer4 = tmp[7:]     
        if args.multi_scale:
          self.proj1=nn.Conv2d(192,768,1,1,0)
          self.proj2=nn.Conv2d(384,768,1,1,0)  
        # print(self.layer1)
        # print(self.layer2)
        # print(self.layer3)
        # print(self.layer4)
        self.num_channels=768
        self.args=args
    def forward(self, x):   
        out={}      
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        if self.args.multi_scale:
            x=self.proj1(x1)+self.proj2(x2)+x3
        else:
          x=x3
        out['layer4']=x
        return out

def build_yolo_double_backbone(model_size : str = 'yolov5s.pt'):
    #position_embedding=build_position_encoding()

    vis_backbone    = YOLObackbone(model_size)
    lwir_backbone   = YOLObackbone(model_size)

    #lwir_backbone.body.layer1=vis_backbone.body.layer1
    lwir_backbone.layer2=vis_backbone.layer2
    lwir_backbone.layer3=vis_backbone.layer3
    lwir_backbone.layer4=vis_backbone.layer4

    #vis_model=Joiner(vis_backbone,position_embedding)
    #lwir_model=Joiner(lwir_backbone,position_embedding)

    #return vis_model,lwir_model
    return vis_backbone, lwir_backbone
if __name__ == '__main__':
    vis_backbone, lwir_backbone = build_yolo_double_backbone('yolov5s.pt')