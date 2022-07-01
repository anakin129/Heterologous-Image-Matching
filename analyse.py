import torch
import cv2
from dataset.m3fd import M3FD_Dataset
from dataset.processing import TransTProcessing
from dataset import sampler
from dataset.loader import LTRLoader
from utils.config import cfg
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageDraw,Image
from model.transt_model import transt_resnet50
import os
import argparse
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import matplotlib
matplotlib.use('Agg')

parser=argparse.ArgumentParser(description='analyse configuration')
parser.add_argument('--gpu_id',type=str,help='gpu id',default='0')
parser.add_argument('--image_source',action="store_true", default=False,help='whether to use images which are same type')
parser.add_argument('--model_id',type=int,default=0,help='0:low-sep-high-sharing; 1:share,2:sep')
parser.add_argument('--backbone_id',type=int,default=0,help='0:resnet;1:mobilenetv3;2:yolov5_backbone')
parser.add_argument('--fusion_network_id',type=int,default=1,help='0:ECA-CFA;1:ECA-search_CFA')
parser.add_argument('--multi_scale',action='store_true',default=False,help='whether to use multi_scale')
parser.add_argument('--pretrained_path',type=str,default='',help='model path')
args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_dir='./features_vis'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# get special layer output
class LayerActivations:
    features=None
    def __init__(self,layer):
        self.hook=layer.register_forward_hook(self.hook_fn)
    
    def hook_fn(self,module,input,output):
        self.features=output.cpu()
    
    def remove(self):
        self.hook.remove()

def vis(act):
    feature=act.features # (16*16)*bs*256 or (32*32)*bs*256
    if feature.shape[0]==256:
        feature=feature.permute(1,0,2)[0] #(16*16)*256
        feature=feature.reshape(16,16,256).permute(2,0,1)#256*16*16
    else:
        feature=feature.permute(1,0,2)[0] #(32*32)*256
        feature=feature.reshape(32,32,256).permute(2,0,1)#256*32*32

    fig = plt.figure(figsize=(50, 50))
    cm = plt.cm.get_cmap('gray')
    for i in range(16):
        ax = fig.add_subplot(4,4, i+1, xticks=[], yticks=[])
        #ax.imshow(feature[i].detach().numpy(), cmap=cm)
        ax.imshow(feature[i].detach().numpy()/np.max(feature[i].detach().numpy()), alpha=0.7, cmap='rainbow')
    fig.tight_layout()
    return fig
    #plt.show()

def main():

    m3fd_test=M3FD_Dataset(cfg.DATASET.M3FD.m3fd_test_dir)
    center_jitter_factor = {'search': 0, 'template': 0}
    scale_jitter_factor = {'search': 0, 'template': 0}
    data_processing_train =TransTProcessing(search_area_factor=cfg.TRAIN.search_area_factor,
                                                      template_area_factor = cfg.TRAIN.template_area_factor,
                                                      search_sz=cfg.TRAIN.search_sz,
                                                      temp_sz=cfg.TRAIN.temp_sz,
                                                      center_jitter_factor=center_jitter_factor,
                                                      scale_jitter_factor=scale_jitter_factor,
                                                      mode='sequence')
    dataset_test = sampler.TransTSampler([m3fd_test], [1],
                                samples_per_epoch=1000*cfg.TRAIN.batch_size, max_gap=100, processing=data_processing_train,args=args)
    loader_test = LTRLoader('test', dataset_test, training=False, batch_size=1, num_workers=0,shuffle=True)
    
    # Create network and actor
    model = transt_resnet50(args)
    checkpiont=torch.load(args.pretrained_path,map_location='cpu')
    model.load_state_dict(checkpiont['net'])
    model.to(device)
    model.eval()

    dec=model.featurefusion_network.decoder
    act = LayerActivations(dec)
    data = next(iter(loader_test)) 
    tpl=data['template_images'].to(device)
    sch=data['search_images'].to(device)
    with torch.no_grad():
        output= model(sch,tpl)
    torch.cuda.empty_cache()
    act.remove()
    f=vis(act)
    f.savefig('./features_vis/'+'decoder.png')
    f.clear()

    sort_idx=output['pred_logits'][0].detach().cpu().numpy()
    sort_idx=sort_idx[:,0].argsort()[::-1]
    max_idx=sort_idx[0]
    pred_bbox=output['pred_boxes'][0].detach().cpu().numpy()*256
    pred_bbox=pred_bbox[max_idx]
    sch_anno=data['search_anno'].cpu().numpy()
    sch=data['search_images'].permute(0,2,3,1).numpy()
    cv2.imwrite('./features_vis/search.jpg',sch[0]*255)
    t=data['template_images'].permute(0,2,3,1).numpy()
    cv2.imwrite('./features_vis/template.jpg',t[0]*255)
    p=Image.fromarray(np.uint8(sch[0]*255))
    draw=ImageDraw.Draw(p)
    x1,y1,x3,y3=sch_anno[0][0],sch_anno[0][1],sch_anno[0][0]+sch_anno[0][2],sch_anno[0][1]+sch_anno[0][3]
    draw.line([(x1, y1), (x3, y1), (x3, y3), (x1, y3), (x1, y1)],width=3,fill='green') 
    x1,y1,x3,y3=pred_bbox[0]-pred_bbox[2]//2,pred_bbox[1]-pred_bbox[3]//2,pred_bbox[0]+pred_bbox[2]//2,pred_bbox[1]+pred_bbox[3]//2
    draw.line([(x1, y1), (x3, y1), (x3, y3), (x1, y3), (x1, y1)],width=3,fill='red')
    save_path='./features_vis/result.jpg'
    p.save(save_path)
    num=5  if args.fusion_network_id else 4
    for i in range(num):
        ECA1_tpl=model.featurefusion_network.encoder.layers[i].norm11
        act1 = LayerActivations(ECA1_tpl)
        ECA1_sch=model.featurefusion_network.encoder.layers[i].norm21
        act2 = LayerActivations(ECA1_sch)
        if not args.fusion_network_id:
          CFA1_tpl=model.featurefusion_network.encoder.layers[i].norm13
          act3 = LayerActivations(CFA1_tpl)
        CFA1_sch=model.featurefusion_network.encoder.layers[i].norm23
        act4 = LayerActivations(CFA1_sch)

        data = next(iter(loader_test)) 
        tpl=data['template_images'].to(device)
        sch=data['search_images'].to(device)
        with torch.no_grad():
            output= model(sch,tpl)
        torch.cuda.empty_cache()
        act1.remove()
        act2.remove()
        if not args.fusion_network_id:
          act3.remove()
        act4.remove()
        f=vis(act1)
        f.savefig('./features_vis/'+'ECA%s_tpl.png'%(i+1))
        f.clear()  # release memory
        f=vis(act2)
        f.savefig('./features_vis/'+'ECA%s_sch.png'%(i+1))
        f.clear() 
        if not args.fusion_network_id:
          f=vis(act3)
          f.savefig('./features_vis/'+'CFA%s_tpl.png'%(i+1))
          f.clear()  
        f=vis(act4)
        f.savefig('./features_vis/'+'CFA%s_sch.png'%(i+1))
        f.clear() 


if __name__=='__main__':
    main()