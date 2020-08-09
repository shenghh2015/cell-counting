## June 18
# python live_dead.py --net_type FPN --backbone efficientnetb3 --pre_train True --batch_size 8 --dim 320 --epoch 600 --lr 5e-4 --gpu 0

# python live_dead.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 5 --dim 512 --epoch 200 --lr 5e-4 --train 900 --bk_weight 0.7 --gpu 0
# python live_dead.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 5 --dim 512 --epoch 300 --lr 5e-4 --dataset live_dead_1664 --train 900 --rot 0 --gpu 0
# python live_dead.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 5 --dim 416 --epoch 100 --lr 5e-4 --dataset live_dead --down_factor 2 --train 900 --rot 45 --gpu 0

# python live_dead.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 5 --dim 512 --epoch 200 --lr 5e-4 --dataset live_dead --down_factor 1 --train 900 --gpu 0 --loss focal+dice
# python life_cycle_flu.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 1 --dim 1024 --epoch 60 --lr 5e-4 --train 1100 --filtered True --gpu 0 --loss mse --act_fun relu --channels fl1
# python cell_cycle.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 1 --dim 1024 --down_factor 1 --epoch 100 --dataset cell_cycle_1984_v2 --lr 5e-4 --train 1100 --gpu 3 --loss focal+dice
# python life_cycle_flu.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 1 --dim 1024 --epoch 100 --lr 5e-4 --train 1100 --filtered False --gpu 0 --loss mse --act_fun relu --channels fl1 --dataset cell_cycle_1984_v2 --flu_scale 1.0
# python cell_cycle.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 1 --dim 1024 --down_factor 1 --ext True --epoch 100 --dataset cell_cycle_1984_v2 --lr 5e-4 --train 1100 --gpu 0 --loss focal+dice
#python phase_flu.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 2 --dim 800 --epoch 60 --lr 5e-4 --train 1100 --filtered True --gpu 0 --loss mse --act_fun sigmoid --channels fl1 --dataset cell_cycle_1984 --ext False
# python deeply_train.py --net_type DUNet --backbone efficientnetb3 --pre_train True --batch_size 4 --dim 512 --epoch 1 --lr 5e-4 --dataset live_dead --train 900 --gpu 1 --loss focal+dice
# python single_train.py --net_type FPN --backbone efficientnetb2 --pre_train True --batch_size 4 --dim 512 --epoch 1 --lr 5e-4 --dataset live_dead --train 900 --gpu 0 --loss focal+dice
# python single_train.py --net_type BiFPN --backbone efficientnetb0 --pre_train True --batch_size 4 --dim 512 --epoch 1 --lr 5e-4 --dataset live_dead --train 900 --gpu 0 --loss focal+dice
# python single_train.py --net_type BiFPN --backbone efficientnetb0 --pre_train True --batch_size 4 --dim 512 --epoch 1 --lr 5e-4 --dataset live_dead --train 900 --gpu 0 --loss focal+dice
python single_train.py --net_type Nestnet --backbone efficientnetb3 --pre_train True --batch_size 2 --dim 512 --epoch 60 --lr 5e-4 --dataset live_dead --train 900 --gpu 0 --loss focal+dice