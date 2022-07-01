from torchvision.models.mobilenetv3 import mobilenet_v3_large
import torch.nn as nn
class mobilenet_backbone(nn.Module):
    def __init__(self,args):
        super().__init__()
        tmp=mobilenet_v3_large(pretrained=True).features
        self.layer1 = tmp[0:3]     
        self.layer2 = tmp[3:]          
        self.num_channels=960
        self.args=args
        if args.multi_scale:
          self.proj1=nn.Conv2d(80,960,1,1,0)
          self.proj2=nn.Conv2d(160,960,1,1,0)
    def forward(self, x):   
        out={}      
        x = self.layer1(x)
        if self.args.multi_scale:
          x1=self.layer2[0:5](x)
          x2=self.layer2[5:11](x1)
          x3=self.layer2[11:](x2)
          x=x3+self.proj1(x1)+self.proj2(x2)
        else:
          x = self.layer2(x)
        out['layer2']=x
        return out