import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import segmentation_models as sm
from segmentation_models import Unet, Linknet, PSPNet, FPN
sm.set_framework('tf.keras')
import glob

def generate_folder(folder):
	if not os.path.exists(folder):
		os.system('mkdir -p {}'.format(folder))

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_root_folder ='./models'; 
model_name = 'livedead-net-Unet-bone-efficientnetb3-pre-True-epoch-200-batch-6-lr-0.0005'; 
DATA_DIR = './data/live_dead'; result_folder = './results/{}'.format(os.basename(DATA_DIR))
generate_folder(result_folder);print(result_folder)
CLASSES = ['live', 'inter', 'dead']
x_train_dir = os.path.join(DATA_DIR, 'train_images')
y_train_dir = os.path.join(DATA_DIR, 'train_masks')

x_valid_dir = os.path.join(DATA_DIR, 'val_images')
y_valid_dir = os.path.join(DATA_DIR, 'val_masks')

x_test_dir = os.path.join(DATA_DIR, 'test_images')
y_test_dir = os.path.join(DATA_DIR, 'test_masks')

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
        self.ids = os.listdir(images_dir)
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

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(832, 832)
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

## prediction with the best model
model_folder = os.path.join(model_root_folder,model_name)
best_weight = model_folder+'/best_model.h5'
if os.path.exists(best_weight):
    ## parse the folder name
    splits = model_name.split('-')
    for v in range(len(splits)):
        if splits[v] == 'net':
            net_type = splits[v+1]
        elif splits[v] == 'bone':
            backbone = splits[v+1]
    print('network: {}, backbone: {}'.format(net_type, backbone))
    # dataset settings
    preprocess_input = sm.get_preprocessing(backbone)

    # load test data
    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        classes=CLASSES, 
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )
    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

    #create model
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    net_func = globals()[net_type]
    model = net_func(backbone, classes=n_classes, activation=activation)

    #load best weights
    model.load_weights(model_folder+'/best_model.h5')

    # define optomizer
    optim = tf.keras.optimizers.Adam(0.001)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 0.5]))
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    scores = model.evaluate_generator(test_dataloader)
    print("Loss: {:.5}".format(scores[0]))

    for metric, value in zip(metrics, scores[1:]):
        print("mean {}: {:.5}".format(metric.__name__, value))

## evaluate the model on testing data
images = []; gt_masks = []
for i in range(len(test_dataset)):
    image, gt_mask = test_dataset[i]
    images.append(image); gt_masks.append(gt_mask)
images = np.stack(images); gt_masks = np.stack(gt_masks)
pr_masks = model.predict(test_dataloader)

## calculate IoU and Dice
def iou_calculate(y_true, y_pred):
    # one hot encoding of predictions
    num_classes = y_pred.shape[-1]
    y_pred = np.array([np.argmax(y_pred, axis=-1)==i for i in range(num_classes)]).transpose(1,2,3,0)
    print(y_pred.shape)
    axes = (1,2) # W,H axes of each image
    intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
    # intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    union = np.sum(np.logical_or(y_pred, y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    # union = mask_sum  - intersection

    smooth = .00001
    iou_per_image_class = (intersection + smooth) / (union + smooth)
    dice_per_image_class = (2 * intersection + smooth)/(mask_sum + smooth)
    
    mean_iou_over_images = np.mean(iou_per_image_class, axis = 0)
    mean_iou_over_images_class = np.mean(mean_iou_over_images)
    dice_class = np.mean(dice_per_image_class, axis = 0)
    mean_dice = np.mean(dice_per_image_class)

    return mean_iou_over_images_class, mean_iou_over_images, mean_dice, dice_class

result_file = result_folder+'/'+model_name+'/IoU_Dice.txt'
iou_score, iou_cls, dice_score, dice_class = iou_calculate(gt_masks, pr_masks)

with open(result_file, 'w+') as f:
	# write iou
	f.write('IoU:{}{}{}{}{}',format())

print(iou_cls)
print(iou_score)
print(dice_class)
print(dice_score)
