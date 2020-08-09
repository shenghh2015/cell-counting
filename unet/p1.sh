# python cell_cycle.py --net_type FPN --backbone efficientnetb2 --pre_train True --batch_size 8 --dim 320 --epoch 600 --lr 5e-4 --gpu 1
# python live_dead.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 5 --dim 512 --train 100 --epoch 5 --lr 5e-4 --gpu 1

# python cell_cycle.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 5 --dim 512 --epoch 200 --lr 5e-4 --train 1100 --bk_weight 0.7 --gpu 1
# python cell_cycle.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 4 --dim 512 --epoch 300 --lr 1e-4 --train 1100 --gpu 1
#python live_dead.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 2 --epoch 300 --lr 5e-4 --dataset live_dead_1664 --train 900 --dim 800 --rot 0 --gpu 1
#python cell_cycle.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 4 --dim 320 --down_factor 2 --epoch 2 --dataset cell_cycle2 --lr 5e-4 --train 1100 --rot 0 --gpu 1

# python life_cycle_flu.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 1 --dim 1024 --epoch 60 --lr 5e-4 --train 1100 --filtered True --gpu 1 --loss mse --act_fun relu --channels combined --dataset cell_cycle_1984_v2
# python life_cycle_flu.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 1 --dim 1024 --epoch 100 --lr 5e-4 --train 1100 --filtered False --gpu 1 --loss mse --act_fun sigmoid --channels combined --dataset cell_cycle_1984_v2 --ext True
# python phase_flu.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 2 --dim 800 --epoch 60 --lr 5e-4 --train 1100 --filtered True --gpu 1 --loss mse --act_fun relu --channels fl2 --dataset cell_cycle_1984_v2 --ext False
# python biFPN_train.py --net_type BiFPN --backbone efficientnetb0 --pre_train True --batch_size 4 --dim 512 --epoch 100 --lr 5e-4 --dataset live_dead --train 900 --gpu 1 --loss focal+dice
#python single_train.py --net_type PSPNet --backbone efficientnetb3 --pre_train True --batch_size 4 --dim 528 --epoch 1 --lr 5e-4 --dataset live_dead --train 900 --gpu 1 --loss focal+dice
python single_train.py --net_type PSPNet --backbone efficientnetb3 --pre_train True --batch_size 1 --dim 1056 --epoch 1 --lr 5e-4 --dataset cell_cycle_1984_v2 --train 1100 --gpu 1 --loss focal+dice
