import sys
sys.path.append('..')
import os
import random
import json
from tkinter import N
from tkinter.messagebox import NO
import torch
from utils.image_loader import opencv_loader
from dataset.base_video_dataset import BaseVideoDataset
from utils.config import cfg

class M3FD_Dataset(BaseVideoDataset):
    def __init__(self,root=None,image_loader=opencv_loader,split='train'):
        root=cfg.DATASET.M3FD.m3fd_dir if root is None else root
        super().__init__('M3FD',root,image_loader)

        self.anno_path=os.path.join(root,'gt.json')
        with open(self.anno_path,'r') as f:
            meta_data=json.load(f)
        self.img_path=list(meta_data.keys())
        self.gt=list(meta_data.values())
        self.transform()
        self.sequence_list=list(range(len(self.img_path)))
        self.seq2name={seq:name for seq,name in enumerate(self.img_path)}
        
    def transform(self):
        for bbox in self.gt:
            bbox[0]=-1*bbox[2]//2+511//2
            bbox[1]=-1*bbox[3]//2+511//2
            
    def _get_frames(self,seq_id,img_type):
        frame_name=self.img_path[seq_id]
        path=os.path.join(self.root,img_type,frame_name+'.png')
        return self.image_loader(path)

    def get_sequence_info(self, seq_id):
        bbox=self.gt[seq_id]
        bbox=torch.Tensor(bbox).view(1,4)
        valid=(bbox[:,2]>0) & (bbox[:,3]>0)
        visible=valid.clone().byte()
        return {'bbox':bbox,'valid':valid,'visible':visible}

    def get_frames(self,seq_id=None,frame_ids=None,anno=None,img_type=None):
        frame=self._get_frames(seq_id,img_type)
        frame_list=[frame.copy( ) for _ in frame_ids]
        if anno is None:
            anno=self.get_sequence_info(seq_id)
        anno_frames={}
        for key,value in anno.items():
            anno_frames[key]=[value[0,...] for _ in frame_ids]
        object_meta=None
        return frame_list,anno_frames,object_meta
    
    def get_num_sequences(self):
        return len(self.sequence_list)
    
    def get_name(self):
        return 'm3fd'
    
    def is_video_sequence(self):
        return False

if __name__=='__main__':
  m3fd=M3FD_Dataset(root=cfg.DATASET.M3FD.m3fd_test_dir)
  template_frame_ids = [1] 
  search_frame_ids = [1]
  seq_id=random.randint(0,m3fd.get_num_sequences()-1)
  seq_info_dict=m3fd.get_sequence_info(seq_id)
  vis_info=m3fd.get_frames(seq_id,template_frame_ids,seq_info_dict,'vis')
  lwir_info=m3fd.get_frames(seq_id,template_frame_ids,seq_info_dict,'lwir') 
  vis_img=vis_info[0][0]
  lwir_img=lwir_info[0][0]
  vis_gt=vis_info[1]['bbox'][0]
  lwir_gt=lwir_info[1]['bbox'][0]
  
  from PIL import Image, ImageOps, ImageStat, ImageDraw
  import numpy as np
  p=Image.fromarray(np.uint8(vis_img))
  draw=ImageDraw.Draw(p)
  x1,y1,x3,y3=vis_gt[0],vis_gt[1],vis_gt[0]+vis_gt[2],vis_gt[1]+vis_gt[3]
  draw.line([(x1, y1), (x3, y1), (x3, y3), (x1, y3), (x1, y1)], width=3, fill='green')
  save_path='./vis.jpg'
  p.save(save_path)
  
  p=Image.fromarray(np.uint8(lwir_img))
  draw=ImageDraw.Draw(p)
  x1,y1,x3,y3=lwir_gt[0],lwir_gt[1],lwir_gt[0]+lwir_gt[2],lwir_gt[1]+lwir_gt[3]
  draw.line([(x1, y1), (x3, y1), (x3, y3), (x1, y3), (x1, y1)], width=3, fill='green')
  save_path='./lwir.jpg'
  p.save(save_path)