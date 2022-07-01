import os

#  异源：low-sep-high-sharing + resnet50 + fusion_network(ECA-CFA)

cmd = 'python train.py  --pretrained_path "./checkpoint/different_source/low-sep-high-sharing/resnet/Transt_0_0_0.pth.tar" --model_id 0 --backbone_id 0 --fusion_network_id 0  --lr 1e-4  --optimizer "AdamW" --seed 10722 --log "train"  --gpu_id "0" '
os.system(cmd)



#  异源：low-sep-high-sharing + mobilenetv3 + fusion_network(ECA-CFA)
"""
cmd = 'python train.py  --pretrained_path "./checkpoint/different_source/low-sep-high-sharing/mobilenetv3/Transt_0_1_0.pth.tar" --model_id 0 --backbone_id 1 --fusion_network_id 0  --lr 1e-4  --optimizer "AdamW" --seed 10722 --log "train"  --gpu_id "3" '
os.system(cmd)
"""


#  异源：low-sep-high-sharing + yolov5 + fusion_network(ECA-CFA)
"""
cmd = 'python train.py  --pretrained_path "./checkpoint/different_source/low-sep-high-sharing/yolov5/Transt_0_2_0.pth.tar" --model_id 0 --backbone_id 2 --fusion_network_id 0  --lr 1e-4  --optimizer "AdamW" --seed 10722 --log "train"  --gpu_id "3" '
os.system(cmd)
"""


#  异源：sharing + resnet50 + fusion_network(ECA-CFA)
"""
cmd = 'python train.py  --pretrained_path "./checkpoint/different_source/sharing/resnet/Transt_1_0_0.pth.tar" --model_id 1 --backbone_id 0 --fusion_network_id 0  --lr 1e-4  --optimizer "AdamW" --seed 10722 --log "train"  --gpu_id "3" '
os.system(cmd)
"""


#  异源：sep + resnet50 + fusion_network(ECA-CFA)
"""
cmd = 'python train.py  --pretrained_path "./checkpoint/different_source/sep/resnet/Transt_2_0_0.pth.tar" --model_id 2 --backbone_id 0 --fusion_network_id 0  --lr 1e-4  --optimizer "AdamW" --seed 10722 --log "train"  --gpu_id "3" '
os.system(cmd)
"""


#  异源：low-sep-high-sharing + resnet50 (mulit_scale) + fusion_network(ECA-CFA)
"""
cmd = 'python train.py  --pretrained_path "./checkpoint/different_source/low-sep-high-sharing/resnet/multi/Transt_0_0_0_m.pth.tar" --multi_scale --model_id 0 --backbone_id 0 --fusion_network_id 0  --lr 1e-4  --optimizer "AdamW" --seed 10722 --log "train"  --gpu_id "3" '
os.system(cmd)
"""


#  异源：low-sep-high-sharing + mobilenetv3 (mulit_scale) + fusion_network(ECA-CFA)
"""
cmd = 'python train.py  --pretrained_path "./checkpoint/different_source/low-sep-high-sharing/mobilenetv3/multi/Transt_0_1_0_m.pth.tar" --multi_scale --model_id 0 --backbone_id 1 --fusion_network_id 0  --lr 1e-4  --optimizer "AdamW" --seed 10722 --log "train"  --gpu_id "3" '
os.system(cmd)
"""


#  异源：low-sep-high-sharing + yolov5 (mulit_scale) + fusion_network(ECA-CFA)
"""
cmd = 'python train.py  --pretrained_path "./checkpoint/different_source/low-sep-high-sharing/yolov5/multi/Transt_0_2_0_m.pth.tar" --multi_scale --model_id 0 --backbone_id 2 --fusion_network_id 0  --lr 1e-4  --optimizer "AdamW" --seed 10722 --log "train"  --gpu_id "3" '
os.system(cmd)
"""


#  异源：low-sep-high-sharing + resnet50 + fusion_network(part--ECA-CFA)
"""
cmd = 'python train.py  --pretrained_path "./checkpoint/different_source/low-sep-high-sharing/resnet/part_attention/Transt_0_0_1.pth.tar"  --model_id 0 --backbone_id 0 --fusion_network_id 1  --lr 1e-4  --optimizer "AdamW" --seed 10722 --log "train"  --gpu_id "3" '
os.system(cmd)
"""


#  同源：sharing + resnet50 + fusion_network(ECA-CFA)
"""
cmd = 'python train.py  --pretrained_path "./checkpoint/same_source/sharing/resnet/Transt_0_0_0_s.pth.tar" --image_source --model_id 1 --backbone_id 0 --fusion_network_id 0  --lr 1e-4  --optimizer "AdamW" --seed 10722 --log "train"  --gpu_id "3" '
os.system(cmd)
"""


#  同源：sharing + mobilenetv3(multi) + fusion_network(ECA-CFA)
"""
cmd = 'python train.py  --pretrained_path "./checkpoint/same_source/sharing/mobilenetv3/multi/Transt_0_1_0_m_s.pth.tar" --image_source --multi_scale --model_id 1 --backbone_id 1 --fusion_network_id 0  --lr 1e-4  --optimizer "AdamW" --seed 10722 --log "train"  --gpu_id "3" '
os.system(cmd)
"""


#  同源：sharing + yolov5(multi) + fusion_network(ECA-CFA)
"""
cmd = 'python train.py  --pretrained_path "./checkpoint/same_source/sharing/yolov5/multi/Transt_0_2_0_m_s.pth.tar" --image_source --multi_scale --model_id 1 --backbone_id 2 --fusion_network_id 0  --lr 1e-4  --optimizer "AdamW" --seed 10722 --log "train"  --gpu_id "3" '
os.system(cmd)
"""
