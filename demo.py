from sklearn.feature_extraction import img_to_graph
import torch
import cv2
from dataset.m3fd import M3FD_Dataset
from dataset.processing import TransTProcessing
from dataset import sampler
from dataset.loader import LTRLoader
from utils.config import cfg
from PIL import ImageDraw,Image
import numpy as np
from model.transt_model import transt_resnet50
import os
import argparse
from loss.iou import generalized_box_iou
import json

parser=argparse.ArgumentParser(description='demo configuration')
parser.add_argument('--gpu_id',type=str,help='gpu id',default='0')
parser.add_argument('--image_source',action="store_true", default=False,help='whether to use images which are same type')
parser.add_argument('--model_id',type=int,default=0,help='0:low-sep-high-sharing; 1:share,2:sep')
parser.add_argument('--backbone_id',type=int,default=0,help='0:resnet;1:mobilenetv3;2:yolov5_backbone')
parser.add_argument('--fusion_network_id',type=int,default=0,help='0:ECA-CFA;1:ECA-search_CFA')
parser.add_argument('--multi_scale',action='store_true',default=False,help='whether to use multi_scale')
parser.add_argument('--pretrained_path',type=str,default='./checkpoint/different_source/low-sep-high-sharing/resnet/Transt_0_0_0.pth.tar',help='model path')
parser.add_argument('--all_targets',action='store_true',default=False,help='whether to show all targets')
args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

demo_dir='./Data/m3fd/'
save_dir='./demo_all'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    os.mkdir(save_dir+'/result')
    os.mkdir(save_dir+'/template')



show_num=200
device=torch.device('cuda')


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xywh_to_xyxy(x):
    x1,y1,w,h=x.unbind(-1)
    b = [x1, y1,(x1 +w), (y1+h)]
    return torch.stack(b, dim=-1)

def main():
    m3fd_test=M3FD_Dataset(cfg.DATASET.M3FD.m3fd_test_dir)
    center_jitter_factor = {'search': 0, 'template': 0}
    scale_jitter_factor = {'search': 0, 'template': 0}
    data_processing_test =TransTProcessing(search_area_factor=cfg.TRAIN.search_area_factor,
                                                      template_area_factor = cfg.TRAIN.template_area_factor,
                                                      search_sz=cfg.TRAIN.search_sz,
                                                      temp_sz=cfg.TRAIN.temp_sz,
                                                      center_jitter_factor=center_jitter_factor,
                                                      scale_jitter_factor=scale_jitter_factor,
                                                      mode='sequence')
    dataset_test = sampler.TransTSampler([m3fd_test], [1],
                                samples_per_epoch=3636, max_gap=0, processing=data_processing_test,args=args)
    loader_test = LTRLoader('test', dataset_test, training=False, batch_size=1, num_workers=0,shuffle=False)
    
    # Create network and actor
    model = transt_resnet50(args)
    checkpiont=torch.load(args.pretrained_path,map_location='cpu')
    model.load_state_dict(checkpiont['net'])
    model.to(device)
    model.eval()
    j=-1
    anno_path=os.path.join(demo_dir,'test','gt.json')
    with open(anno_path,'r') as f:
        meta_data=json.load(f)
    img_all=list(meta_data.keys())
    gt_all=list(meta_data.values())
    tmp=None
    cv2.namedWindow("demo",0)
    cv2.resizeWindow("demo", 640,480)
    cv2.moveWindow("demo",0,0)
    cv2.namedWindow("template",0)
    cv2.resizeWindow("template", 200, 200)
    cv2.moveWindow("template",0,0)
    
    with torch.no_grad():
        for idx,data in enumerate(loader_test):
            #if idx>=show_num:
             #   break
            j+=1
            img_path=img_all[j].split('_')[0]+'.png'
            if  args.all_targets:
                if img_path==tmp:
                    continue
            output=model(data['search_images'].cuda(),data['template_images'].cuda())
            sort_idx=output['pred_logits'][0].detach().cpu().numpy()
            sort_idx=sort_idx[:,0].argsort()[::-1]
            max_idx=sort_idx[0]
            pred_bbox=output['pred_boxes'][0].detach().cpu().numpy()*256
            pred_bbox=pred_bbox[max_idx]
            sch_anno=data['search_anno'].cpu().numpy()
            sch=data['search_images'].permute(0,2,3,1).numpy()
            pred=torch.tensor(pred_bbox).unsqueeze(0)#b*4
            pred=box_cxcywh_to_xyxy(pred)
            gt=box_xywh_to_xyxy(torch.tensor(sch_anno))#b*4
            giou, iou = generalized_box_iou(pred,gt)
            if iou<0.3:
                continue
            img=cv2.imread(os.path.join(demo_dir,'lwir',img_path))
            gt=gt_all[j]
            x1,y1,x3,y3=int(gt[0]),int(gt[1]),int(gt[0]+gt[2]),int(gt[1]+gt[3])
            img=cv2.rectangle(img,(x1,y1),(x3,y3),color=(0,255,0),thickness=5)
            xc=(x1+x3)//2
            yc=(y1+y3)//2
            scale_x=gt[2]/sch_anno[0][2]
            scale_y=gt[3]/sch_anno[0][3]
            pred_bbox[0]=int(pred_bbox[0])-256//2+xc
            pred_bbox[1]=int(pred_bbox[1])-256//2+yc
            pred_bbox[2]*=scale_x
            pred_bbox[3]*=scale_y
            x1,y1,x3,y3=int(pred_bbox[0]-pred_bbox[2]//2),int(pred_bbox[1]-pred_bbox[3]//2),int(pred_bbox[0]+pred_bbox[2]//2),int(pred_bbox[1]+pred_bbox[3]//2)
            img=cv2.rectangle(img,(x1,y1),(x3,y3),color=(0,0,255),thickness=5)
            img=cv2.putText(img,f'IOU: {iou.item():.3f}',(x1-10,y1-10),fontFace=3,fontScale=1, color=(215,235,250))
            tpl=data['template_images'][0].permute(1,2,0).numpy()
            cv2.imshow('template',tpl)
            cv2.imshow('demo',img)
            cv2.waitKey(100)
            cv2.imwrite(os.path.join(save_dir,'result','%s.png'%j),img)
            cv2.imwrite(os.path.join(save_dir,'template','%s.png'%j),tpl*255)
            tmp=img_path
                

if __name__=='__main__':
    main()

