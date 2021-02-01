import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import segmentation_models_v1 as sm
from segmentation_models_v1 import Unet, Linknet, PSPNet, FPN, DUNet

from helper_function import precision, recall, f1_score, iou_calculate
from sklearn.metrics import confusion_matrix

sm.set_framework('tf.keras')
import glob
from natsort import natsorted

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_root_folder = '/data/models/'
# model_root_folder = '/data/models/'

# model_name = 'deeply-net-DUNet-bone-efficientnetb3-pre-True-epoch-200-batch-10-lr-0.0005-dim-512-train-900-rot-0-set-live_dead-loss-focal+dice'
model_name = 'deeply-net-DUNet-bone-efficientnetb3-pre-True-epoch-200-batch-4-lr-0.0005-dim-800-train-900-rot-0-set-cell_cycle_1984_v2-loss-focal+dice'
model_folder = model_root_folder+model_name

## parse model name
splits = model_name.split('-')

for v in range(len(splits)):
	if splits[v]=='set':
		dataset = splits[v+1]
	elif splits[v] == 'net':
		net_arch = splits[v+1]
	elif splits[v] == 'bone':
		backbone = splits[v+1]

DATA_DIR = '/data/datasets/{}'.format(dataset)
x_train_dir = os.path.join(DATA_DIR, 'train_images')
y_train_dir = os.path.join(DATA_DIR, 'train_masks')

x_valid_dir = os.path.join(DATA_DIR, 'val_images')
y_valid_dir = os.path.join(DATA_DIR, 'val_masks')

x_test_dir = os.path.join(DATA_DIR, 'test_images')
y_test_dir = os.path.join(DATA_DIR, 'test_masks')

if dataset == 'live_dead':
	x_train_dir +='2'; x_valid_dir+= '2'; x_test_dir+='2'

val_dim = 832 if 'live_dead' in dataset else 1984

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
    
    CLASSES = ['bk', 'live', 'inter', 'dead']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = natsorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
#         print(np.unique(mask))
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
#         print(self.class_values)
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
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
        map_batch = batch[1]
        map_batch_list = [map_batch]
        for i in range(4):
            map_batch_list.append(map_batch[:,::2,::2,:])
            map_batch = map_batch[:,::2,::2,:]
        map_batch_list.reverse()
#         batch_tuple = (batch[0],)
        map_tuple = ()
        for i in range(5):
            map_tuple = map_tuple+(map_batch_list[i],)
        return (batch[0], map_tuple)
    
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

def get_validation_augmentation(dim = 832):
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
CLASSES = ['live', 'inter', 'dead']
n_classes = len(CLASSES) + 1
activation = 'softmax'
net_func = globals()[net_arch]
model = net_func(backbone, classes=n_classes, activation=activation)

#load best weights
model.load_weights(best_weight)
## save model
model.save(model_folder+'/ready_model.h5')

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
    classes=CLASSES, 
    augmentation=get_validation_augmentation(val_dim),
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

# calculate the pixel-level classification performance
pr_masks = model.predict(test_dataloader)[-1]; 
pr_maps = np.argmax(pr_masks,axis=-1)

gt_masks = []
for i in range(len(test_dataset)):
    _, gt_mask = test_dataset[i];gt_masks.append(gt_mask)
gt_masks = np.stack(gt_masks);gt_maps = np.argmax(gt_masks,axis=-1)

## IoU and dice coefficient
iou_classes, mIoU, dice_classes, mDice = iou_calculate(gt_masks, pr_masks)
print('iou_classes: {:.4f},{:.4f},{:.4f},{:.4f}; mIoU: {:.4f}'.format(iou_classes[-1],iou_classes[0],iou_classes[1],iou_classes[2], mIoU))
print('dice_classes: {:.4f},{:.4f},{:.4f},{:.4f}; mDice: {:.4f}'.format(dice_classes[-1],dice_classes[0],dice_classes[1],dice_classes[2], mDice))

y_true=gt_maps.flatten(); y_pred = pr_maps.flatten()
cf_mat = confusion_matrix(y_true, y_pred)
cf_mat_reord = np.zeros(cf_mat.shape)
cf_mat_reord[1:,1:]=cf_mat[:3,:3];cf_mat_reord[0,1:]=cf_mat[3,0:3]; cf_mat_reord[1:,0]=cf_mat[0:3,3]
cf_mat_reord[0,0] = cf_mat[3,3]
print('Confusion matrix:')
print(cf_mat_reord)
prec_scores = []; recall_scores = []; f1_scores = []; iou_scores=[]
for i in range(cf_mat.shape[0]):
    prec_scores.append(precision(i,cf_mat_reord))
    recall_scores.append(recall(i,cf_mat_reord))
    f1_scores.append(f1_score(i,cf_mat_reord))
print('Precision: {}, mean {}'.format(np.round(prec_scores, 4),round(np.mean(prec_scores),4)))
print('Recall: {}, mean {}'.format(np.round(recall_scores, 4),round(np.mean(recall_scores),4)))
print('f1 score: {}, mean {}'.format(np.round(f1_scores, 4),round(np.mean(f1_scores),4)))
