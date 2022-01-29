# python train.py --gpu 0 --net 'FCRN_A' --dataset 'bacterial' --batch 32 --epochs 500 --lr 1e-4 --rf 0.9
# python train.py --gpu 0 --net 'C_FCRN' --dataset 'bacterial' --batch 32 --epochs 500 --lr 1e-4 --rf 0.9
python train.py --gpu 0 --net 'FCRN_A' --dataset 'BMC' --batch 16 --dim 512 --epochs 2000 --lr 1e-4 --rf 0.9
python train.py --gpu 0 --net 'C_FCRN' --dataset 'BMC' --batch 12 --dim 512 --epochs 2000 --lr 1e-4 --rf 0.9