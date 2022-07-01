import cv2
import torch
import random
import argparse
import os
import numpy as np
from dataset.coco import COCO_Dataset
from dataset.m3fd import  M3FD_Dataset
from dataset.processing import TransTProcessing
from dataset import sampler
import dataset.transform as tfm
from dataset.loader import LTRLoader
from loss.transt_loss import transt_loss
from trainer.actor import TranstActor
from trainer.ltr_trainer import LTRTrainer
from utils.config import cfg
import logging 
from model.transt_model import transt_resnet50

parser=argparse.ArgumentParser(description='training configuration')
parser.add_argument('--pretrained_path',type=str,help='the path of pretrained model')
parser.add_argument('--seed',default=10909,type=int,help='random seed')
parser.add_argument('--optimizer',type=str,help='AdamW or SGD')
parser.add_argument('--lr',type=float,help='learning rate')
parser.add_argument('--log',type=str,help='log name')
parser.add_argument('--gpu_id',type=str,help='gpu id')

parser.add_argument('--image_source',action="store_true", default=False,help='whether to use images which are same type')
parser.add_argument('--model_id',type=int,default=0,help='0:low-sep-high-sharing; 1:share,2:sep')
parser.add_argument('--backbone_id',type=int,default=0,help='0:resnet;1:mobilenetv3;2:yolov5_backbone')
parser.add_argument('--fusion_network_id',type=int,default=0,help='0:ECA-CFA;1:ECA-search_CFA')
parser.add_argument('--multi_scale',action='store_true',default=False,help='whether to use multi_scale')
args=parser.parse_args()


cfg.TRAIN.log=args.log
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

logger=logging.getLogger('%s'%args.log)
logger.setLevel(logging.INFO)
sh=logging.StreamHandler()
sh.setLevel(logging.INFO)
fh=logging.FileHandler(filename='./%s.log'%args.log,mode='w')
fh.setLevel(logging.INFO)
#logger.addHandler(sh)
logger.addHandler(fh)

logger.info('--------------------------------------------------------------------')
logger.info('--------------------------------------------------------------------')

logger.info(args)

def set_seed(seed=args.seed): ##set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    set_seed()
    # Train datasets
    m3fd_train=M3FD_Dataset()

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=cfg.TRAIN.normalize_mean, std=cfg.TRAIN.normalize_std))


    center_jitter_factor = {'search': 3, 'template': 0}
    scale_jitter_factor = {'search': 0.25, 'template': 0}
    # Data processing to do on the training pairs
    data_processing_train =TransTProcessing(search_area_factor=cfg.TRAIN.search_area_factor,
                                                      template_area_factor = cfg.TRAIN.template_area_factor,
                                                      search_sz=cfg.TRAIN.search_sz,
                                                      temp_sz=cfg.TRAIN.temp_sz,
                                                      center_jitter_factor=center_jitter_factor,
                                                      scale_jitter_factor=scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)


    # The sampler for training
    dataset_train = sampler.TransTSampler([m3fd_train], [1],samples_per_epoch=1000*cfg.TRAIN.batch_size, max_gap=1, processing=data_processing_train,args=args)
    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.batch_size, num_workers=cfg.TRAIN.num_workers,shuffle=True)
    
    # Create network and actor
    model = transt_resnet50(args)
    print(model)
    if args.pretrained_path:
        checkpoint=torch.load(args.pretrained_path,map_location='cpu')
        model.load_state_dict(checkpoint['net'])
    # loss
    objective = transt_loss()
    actor = TranstActor(net=model, objective=objective)
    # Optimizer
    n_parameters = sum(p.numel() for p in model.parameters())
    n_parameters1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    p='number of params:{},requires_grad=True:{}'.format(n_parameters,n_parameters1)
    logger.info(p)
    
    param_dicts=filter(lambda p:p.requires_grad,model.parameters())
    """
    param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": args.lr*0.1,
    },]
    """
    if  args.optimizer=='SGD':              
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr)
    elif args.optimizer=='AdamW':
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,weight_decay=1e-4)
        
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train], optimizer, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(5, load_latest=True, fail_safe=False)

if __name__=='__main__':
    main()