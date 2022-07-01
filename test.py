import torch
import cv2
from dataset.m3fd import M3FD_Dataset
from dataset.processing import TransTProcessing
from dataset import sampler
from dataset.loader import LTRLoader
from utils.config import cfg
from loss.iou import generalized_box_iou
from model.transt_model import transt_resnet50
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import numpy as np
from model.transt_model import transt_resnet50
import argparse
import os
from dataset.coco import COCO_Dataset
import logging
import random
thresh=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
parser=argparse.ArgumentParser(description='testing configuration')
parser.add_argument('--log',type=str,help='log name',default='test')
parser.add_argument('--gpu_id',type=str,help='gpu id',default='0')
parser.add_argument('--image_source',action="store_true", default=False,help='whether to use images which are same type')
parser.add_argument('--model_id',type=int,default=0,help='0:low-sep-high-sharing; 1:share,2:sep')
parser.add_argument('--backbone_id',type=int,default=0,help='0:resnet;1:mobilenetv3;2:yolov5_backbone')
parser.add_argument('--fusion_network_id',type=int,default=1,help='0:ECA-CFA;1:ECA-search_CFA')
parser.add_argument('--multi_scale',action='store_true',default=False,help='whether to use multi_scale')
parser.add_argument('--pretrained_path',type=str,default='',help='model path')
parser.add_argument('--dataset',type=str,default='m3fd',help='dataset')
args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

logger=logging.getLogger('%s'%args.log)
logger.setLevel(logging.INFO)
sh=logging.StreamHandler()
sh.setLevel(logging.INFO)
fh=logging.FileHandler(filename='./%s.log'%args.log,mode='a')
fh.setLevel(logging.INFO)
#logger.addHandler(sh)
logger.addHandler(fh)

logger.info('--------------------------------------------------------------------')
logger.info('--------------------------------------------------------------------')

logger.info(args)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xywh_to_xyxy(x):
    x1,y1,w,h=x.unbind(-1)
    b = [x1, y1,(x1 +w), (y1+h)]
    return torch.stack(b, dim=-1)


device=torch.device('cuda')
def main():
    if args.dataset=='m3fd':
        test_dataset=M3FD_Dataset(cfg.DATASET.M3FD.m3fd_test_dir)
    elif args.dataset=='coco':
        test_dataset=COCO_Dataset(cfg.DATASET.COCO.coco_dir)
    center_jitter_factor = {'search': 0, 'template': 0}
    scale_jitter_factor = {'search': 0, 'template': 0}
    data_processing_train =TransTProcessing(search_area_factor=cfg.TRAIN.search_area_factor,
                                                      template_area_factor = cfg.TRAIN.template_area_factor,
                                                      search_sz=cfg.TRAIN.search_sz,
                                                      temp_sz=cfg.TRAIN.temp_sz,
                                                      center_jitter_factor=center_jitter_factor,
                                                      scale_jitter_factor=scale_jitter_factor,
                                                      mode='sequence')
    dataset_test = sampler.TransTSampler([test_dataset], [1],
                                samples_per_epoch=1000*cfg.TRAIN.batch_size, max_gap=100, processing=data_processing_train,args=args)
    loader_test = LTRLoader('test', dataset_test, training=False, batch_size=1,num_workers=cfg.TRAIN.num_workers,shuffle=False)
    
    # Create network and actor
    model = transt_resnet50(args)
    checkpiont=torch.load(args.pretrained_path,map_location='cpu')
    model.load_state_dict(checkpiont['net'])
    model.to(device)
    model.eval()
    print(model)
    count=[0]*len(thresh)
    num=0
    iou_m=0
    giou_m=0
    with torch.no_grad():
      for idx,data in enumerate(loader_test):
        with torch.no_grad():
            output=model(data['search_images'].cuda(),data['template_images'].cuda())
        logits=output['pred_logits'].cpu()#b*1024*2
        pred_bbox=output['pred_boxes'].cpu()*256 #b*1024*4
        sch_anno=data['search_anno'].cpu() #b*4
        logits=logits[:,:,0]#b*1024
        idx_max=logits.argmax(-1)
        pred=pred_bbox[[range(len(idx_max))],idx_max,:].squeeze(0)#b*4
        pred=box_cxcywh_to_xyxy(pred)
        gt=box_xywh_to_xyxy(sch_anno)#b*4
        giou, iou = generalized_box_iou(pred,gt)
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        iou_m+=iou.sum().item()
        giou_m+=giou.sum().item()
        for i in range(len(thresh)):
            count[i]+=np.sum([np.array((x>thresh[i])).sum() for x in iou])
            #count[i]+=torch.sum(torch.tensor([torch.tensor((x>thresh[i])).sum() for x in iou])).item()
        num+=len(iou)
    c=[num]+count+[0]
    metric=[]
    for i in range(len(c)-1):
        metric.append(c[i]-c[i+1])
    metric=list(map(lambda x:x*1.0/num,metric))
    logger.info(metric)
    iou_m/=num
    giou_m/=num
    logger.info('mean iou:%s'%iou_m)
    logger.info('mean giou:%s'%giou_m)
    
    logger.info('backbone time:{:.6f}s'.format(sum(model.backbone_time)/len(model.backbone_time)))
    logger.info('fusion network time:{:.6f}s'.format(sum(model.fusion_time)/len(model.fusion_time)))
    logger.info('MLP time:{:.6f}s'.format(sum(model.cls_reg_time)/len(model.cls_reg_time)))

    x = torch.randn(1,3,256,256).to(device)
    y= torch.randn(1,3,128,128).to(device)
   
    flops = FlopCountAnalysis(model, (x,y))
    logger.info("FLOPs:%s"%flops.total())
    logger.info(parameter_count_table(model))

    logger.info('fps:{:.6f}'.format(1.0*len(model.fps)/sum(model.fps)))
if __name__=='__main__':
    main()