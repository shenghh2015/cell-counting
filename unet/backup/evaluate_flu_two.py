import os
import cv2
from skimage import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import segmentation_models_v1 as sm
from segmentation_models_v1 import Unet, Linknet, PSPNet, FPN

from helper_function import plot_history_flu
from helper_function import precision, recall, f1_score, calculate_psnr, calculate_pearsonr
from helper_function import plot_flu_prediction

sm.set_framework('tf.keras')
import glob
from natsort import natsorted

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_root_folder = '/data/models/'
#model_root_folder = '/data/models/report_results_phase_flu/'

model_fl1 = 'cellcycle_flu-net-Unet-bone-efficientnetb3-pre-True-epoch-60-batch-2-lr-0.0005-dim-1024-train-1100-rot-0-set-1984-fted-True-loss-mse-act-relu-ch-fl1'
model_fl2 = 'cellcycle_flu-net-Unet-bone-efficientnetb3-pre-True-epoch-60-batch-1-lr-0.0005-dim-1024-train-1100-rot-0-set-1984_v2-fted-True-loss-mse-act-relu-ch-fl2-flu_scale-255.0'
model_list = [model_fl1, model_fl2]
model_name = model_list[1]
model_folder = model_root_folder+model_name

## parse model name
splits = model_name.split('-')
dataset = 'cell_cycle_1984'
val_dim = 1984

flu_scale = 1.0
for v in range(len(splits)):
	if splits[v] == 'net':
		net_arch = splits[v+1]
	elif splits[v] == 'bone':
		backbone = splits[v+1]
	elif splits[v] == 'fted':
		flu_header = 'ff' if splits[v+1].lower() == 'true' else 'f'
	elif splits[v] == 'ch':
		flu_ch = splits[v+1]
	elif splits[v] == 'flu_scale':
		flu_scale = float(splits[v+1])
	elif splits[v] == 'act':
		act_fun = splits[v+1]
		
DATA_DIR = '/data/datasets/{}'.format(dataset) 
x_train_dir = os.path.join(DATA_DIR, 'train_images')
y_train_dir = os.path.join(DATA_DIR, 'train_{}masks'.format(flu_header))

x_valid_dir = os.path.join(DATA_DIR, 'val_images')
y_valid_dir = os.path.join(DATA_DIR, 'val_{}masks'.format(flu_header))

x_test_dir = os.path.join(DATA_DIR, 'test_images')
y_test_dir = os.path.join(DATA_DIR, 'test_{}masks'.format(flu_header))

# load the data without preprocessing
image_dir = os.path.join(DATA_DIR, 'test_images')
map_dir = os.path.join(DATA_DIR, 'test_{}masks'.format(flu_header))		

image_fns = os.listdir(image_dir)
images = []; gt_maps = []
for img_fn in image_fns:
	image = io.imread(image_dir+'/{}'.format(img_fn)); images.append(image)
	gt_map =  io.imread(map_dir+'/{}'.format(img_fn)); gt_maps.append(gt_map)
images = np.stack(images); gt_maps = np.stack(gt_maps) # an array of image and ground truth label maps
gt_maps = gt_maps[:,:,:,:-1]/255.

# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            channels = 'fl1',
            classes=None,
            nb_data=None,
            augmentation=None, 
            preprocessing=None,
    ):
        #self.ids = os.listdir(images_dir)
        id_list = os.listdir(images_dir)
        if nb_data ==None:
            self.ids = id_list
        else:
            self.ids = id_list[:int(min(nb_data,len(id_list)))]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        print(len(self.images_fps)); print(len(self.masks_fps))        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.channels = chennels
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):

        # read data
#         image = cv2.imread(self.images_fps[i])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(self.masks_fps[i], cv2.COLOR_BGR2RGB)
        image = io.imread(self.images_fps[i])
        mask = io.imread(self.masks_fps[i])
        mask = mask/255.
        
	    # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

	    # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.channels == 'fl1':
            output_mask = mask[:,:,:1]
        elif self.channels == 'fl2':
            output_mask = mask[:,:,1:-1]
        else:
            output_mask = mask[:,:,:-1]
        return image, output_mask
        
    def __len__(self):
        return len(self.ids)
    
class Dataloder(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return (batch[0], batch[1])
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

import albumentations as A

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation(dim, rot = 0):
    train_transform = [
        A.PadIfNeeded(min_height=dim, min_width=dim, always_apply=True, border_mode=0),
        A.RandomCrop(height=dim, width=dim, always_apply=True),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(dim = 992):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(dim, dim)
#         A.PadIfNeeded(384, 480)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

# network
best_weight = model_folder+'/best_model.h5'
CLASSES = []
preprocess_input = sm.get_preprocessing(backbone)

#create model
# CLASSES = ['live', 'inter', 'dead']
n_classes = 2 if flu_ch == 'combined' else 1
activation = act_fun
net_func = globals()[net_arch]
model = net_func(backbone, classes=n_classes, activation=activation)

#load best weights
model.load_weights(best_weight)

# define optomizer
# optim = tf.keras.optimizers.Adam(0.001)
# 
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
# dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 0.5]))
# focal_loss = sm.losses.CategoricalFocalLoss()
# total_loss = dice_loss + (1 * focal_loss)
# 
# metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
# 
# compile keras model with defined optimozer, loss and metrics
# model.compile(optim, total_loss, metrics)

# evaluate model
subsets = ['train', 'val', 'test']
subset = subsets[2]

if subset == 'val':
	x_test_dir = x_valid_dir; y_test_dir = y_valid_dir
elif subset == 'train':
	x_test_dir = x_train_dir; y_test_dir = y_train_dir

test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    channels = flu_ch,
    classes=CLASSES, 
    augmentation=get_validation_augmentation(val_dim),
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
## evaluate the performance
# calculate the pixel-level classification performance
pr_masks = model.predict(test_dataloader)
# back to scale [0,1]
pr_masks = pr_masks/flu_scale
gt_masks = []
for i in range(len(test_dataset)):
    _, gt_mask = test_dataset[i];gt_masks.append(gt_mask)
gt_masks = np.stack(gt_masks)
# verify the loaded data
print('Load difference: {:.6f}'.format(np.mean(np.abs(gt_masks-gt_maps))))

# save prediction examples
plot_fig_file = model_folder+'/pred_examples.png'; nb_images = 4
plot_flu_prediction(plot_fig_file, images, gt_maps, pr_masks, nb_images, rand_seed = 6)

# calculate PSNR
f1_mPSNR, f1_psnr_scores = calculate_psnr(gt_masks[:,:,:,0], pr_masks[:,:,:,0])
f2_mPSNR, f2_psnr_scores = calculate_psnr(gt_masks[:,:,:,1], pr_masks[:,:,:,1])
mPSNR, f_psnr_scores = calculate_psnr(gt_masks, pr_masks)
print('PSNR: fluo1 {:.4f}, fluo2 {:.4f}, combined {:.4f}'.format(f1_mPSNR, f2_mPSNR, mPSNR))

# calculate Pearson correlation coefficient
f1_mPear, f1_pear_scores = calculate_pearsonr(gt_masks[:,:,:,0], pr_masks[:,:,:,0])
f2_mPear, f2_pear_scores = calculate_pearsonr(gt_masks[:,:,:,1], pr_masks[:,:,:,1])
f_mPear, f_pear_scores = calculate_pearsonr(gt_masks, pr_masks)
print('Pearsonr: fluo1 {:.4f}, fluo2 {:.4f}, combined {:.4f}'.format(f1_mPear, f2_mPear, f_mPear))

with open(model_folder+'/metric_summary.txt','w+') as f:
	# save PSNR over fluorescent 1 and fluorescent 2
	f.write('PSNR: fluo1 {:.4f}, fluo2 {:.4f}, combined {:.4f}\n'.format(f1_mPSNR, f2_mPSNR, mPSNR))
	f.write('Pearsonr: fluo1 {:.4f}, fluo2 {:.4f}, combined {:.4f}\n'.format(f1_mPear, f2_mPear, f_mPear))

# save hisogram of psnr and coefficient
file_name = model_folder+'/hist_psnr_rho.png'
plot_psnr_histogram(file_name, f1_psnr_scores, f2_psnr_scores, f1_pear_scores, f2_pear_scores)