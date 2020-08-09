# parser.add_argument("--gpu", type=str, default = '0')
# parser.add_argument("--epoch", type=int, default = 300)
# parser.add_argument("--batch_size", type=int, default = 2)
# parser.add_argument("--backbone", type=str, default = 'resnet34')
# parser.add_argument("--net_type", type=str, default = 'Unet')  #Unet, Linknet, PSPNet, FPN
# parser.add_argument("--pre_train", type=str2bool, default = True)

python cell_cycle.py --net_type Unet --backbone efficientnetb4 --pre_train True --batch_size 5 --epoch 200 --lr 5e-4 --gpu 0
python cell_cycle.py --net_type Linknet --backbone efficientnetb4 --pre_train True --batch_size 5 --epoch 200 --lr 5e-4 --gpu 0
python cell_cycle.py --net_type PSPNet --backbone efficientnetb4 --pre_train True --batch_size 5 --epoch 200 --lr 5e-4 --gpu 2
python cell_cycle.py --net_type FPN --backbone efficientnetb4 --pre_train True --batch_size 5 --epoch 200 --lr 5e-4 --gpu 0

python live_dead.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 6 --epoch 200 --lr 5e-4 --gpu 4

python live_dead.py --net_type FPN --backbone efficientnetb3 --pre_train True --batch_size 4 --epoch 200 --lr 5e-4 --gpu 0
python live_dead.py --net_type PSPNet --backbone efficientnetb3 --pre_train True --batch_size 4 --epoch 200 --lr 5e-4 --gpu 2
python live_dead.py --net_type Linknet --backbone efficientnetb3 --pre_train True --batch_size 4 --epoch 200 --lr 5e-4 --gpu 3

python cell_cycle.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 4 --down True --epoch 200 --lr 5e-4 --gpu 1
python cell_cycle.py --net_type FPN --backbone efficientnetb3 --pre_train True --batch_size 4 --down True --epoch 200 --lr 5e-4 --gpu 2
python cell_cycle.py --net_type Linknet --backbone efficientnetb3 --pre_train True --batch_size 4 --down True --epoch 200 --lr 5e-4 --gpu 3
python cell_cycle.py --net_type PSPNet --backbone efficientnetb3 --pre_train True --batch_size 4 --down True --epoch 200 --lr 5e-4 --gpu 4

python cell_cycle.py --net_type Unet --backbone efficientnetb4 --pre_train True --batch_size 3 --down True --epoch 300 --lr 5e-4 --gpu 4
python live_dead.py --net_type Unet --backbone efficientnetb4 --pre_train True --batch_size 3 --epoch 300 --lr 5e-4 --gpu 6

#### June 28 morning
python cell_cycle.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 6 --down True --epoch 300 --lr 5e-4 --gpu 1
python cell_cycle.py --net_type FPN --backbone efficientnetb3 --pre_train True --batch_size 6 --down True --epoch 300 --lr 5e-4 --gpu 2
python cell_cycle.py --net_type Linknet --backbone efficientnetb3 --pre_train True --batch_size 6 --down True --epoch 300 --lr 5e-4 --gpu 3

python cell_cycle.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 6 --down True --epoch 300 --lr 1e-3 --gpu 7
python live_dead.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 6 --epoch 300 --lr 1e-3 --gpu 1

### June 28 night-June 29
python live_dead.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 6 --epoch 400 --lr 1e-4 --gpu 0
python live_dead.py --net_type Unet --backbone resnet34 --pre_train True --batch_size 6 --epoch 300 --lr 5e-4 --gpu 1
python live_dead.py --net_type Unet --backbone resnet50 --pre_train True --batch_size 6 --epoch 300 --lr 5e-4 --gpu 2
python live_dead.py --net_type Unet --backbone resnet101 --pre_train True --batch_size 6 --epoch 300 --lr 5e-4 --gpu 3

python cell_cycle.py --net_type FPN --backbone efficientnetb3 --pre_train True --batch_size 6 --down True --epoch 400 --lr 1e-4 --gpu 6
python cell_cycle.py --net_type FPN --backbone resnet34 --pre_train True --batch_size 6 --down True --epoch 300 --lr 5e-4 --gpu 7

python cell_cycle.py --net_type FPN --backbone resnet50 --pre_train True --batch_size 6 --down True --epoch 300 --lr 5e-4 --gpu 1
python cell_cycle.py --net_type FPN --backbone resnet101 --pre_train True --batch_size 6 --down True --epoch 300 --lr 5e-4 --gpu 8

python live_dead.py --net_type Unet --backbone efficientnetb1 --pre_train True --batch_size 6 --epoch 400 --lr 5e-4 --docker True --gpu 0

