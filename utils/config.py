from yacs.config import CfgNode as CN
_C=CN()
cfg=_C

#--------------------------------------------------#
#    training options
#--------------------------------------------------#
_C.TRAIN=CN()

_C.TRAIN.checkpoint_dir= './checkpoint/'  # the directory of checkpoint

_C.TRAIN.use_gpu=True

_C.TRAIN.batch_size=2
_C.TRAIN.num_workers=0
_C.TRAIN.print_interval=10
_C.TRAIN.normalize_mean = [0.485, 0.456, 0.406]
_C.TRAIN.normalize_std=[0.229,0.224,0.225]

#_C.TRAIN.normalize_mean = [0.0, 0.0, 0.0]
#_C.TRAIN.normalize_std=[1.0,1.0,1.0]

_C.TRAIN.search_area_factor = 4.0
_C.TRAIN.template_area_factor = 2
_C.TRAIN.search_feature_sz = 32
_C.TRAIN.template_feature_sz = 16
_C.TRAIN.search_sz = _C.TRAIN.search_feature_sz * 8
_C.TRAIN.temp_sz = _C.TRAIN.template_feature_sz * 8

_C.TRAIN.log=''
_C.TRAIN.thresh=0.5

#--------------------------------------------------#
#    dataset options
#--------------------------------------------------#
_C.DATASET=CN(new_allowed=True)

_C.DATASET.COCO=CN()
_C.DATASET.COCO.coco_dir='./COCO2017/'

_C.DATASET.M3FD=CN()
_C.DATASET.M3FD.m3fd_dir='./Data/m3fd/train/'
_C.DATASET.M3FD.m3fd_test_dir='./Data/m3fd/test/'
#--------------------------------------------------#
#    transt options
#--------------------------------------------------#
_C.TRANST=CN()
_C.TRANST.position_embedding = 'sine'
_C.TRANST.hidden_dim = 256
_C.TRANST.dropout = 0.1
_C.TRANST.nheads = 8
_C.TRANST.dim_feedforward = 2048
_C.TRANST.featurefusion_layers = 4