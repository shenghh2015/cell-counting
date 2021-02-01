import os
import cv2
from skimage import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import segmentation_models_v1 as sm
# from segmentation_models_v1 import Unet, Linknet, PSPNet, FPN, DUNet, BiFPN, Nestnet
sm.set_framework('tf.keras')
from regnet_model import Reg_Net

from helper_function import plot_regress_history, plot_reg_prediction
from helper_function import calculate_psnr, calculate_pearsonr

def str2bool(value):
    return value.lower() == 'true'

def generate_folder(folder_name):
	if not os.path.exists(folder_name):
		os.system('mkdir -p {}'.format(folder_name))

parser = argparse.ArgumentParser()
parser.add_argument("--docker", type=str2bool, default = False)
parser.add_argument("--gpu", type=str, default = '0')
parser.add_argument("--net_type", type=str, default = 'Reg_Net')  #Unet, Linknet, PSPNet, FPN
# parser.add_argument("--backbone", type=str, default = 'efficientnetb3')
parser.add_argument("--epoch", type=int, default = 2)
parser.add_argument("--dim", type=int, default = 256)
parser.add_argument("--batch_size", type=int, default = 2)
parser.add_argument("--dataset", type=str, default = 'bacterial')
# parser.add_argument("--rot", type=float, default = 0)
parser.add_argument("--lr", type=float, default = 1e-4)
# parser.add_argument("--pre_train", type=str2bool, default = True)
parser.add_argument("--cross", type=int, default = 1)
parser.add_argument("--loss", type=str, default = 'mse')
args = parser.parse_args()
print(args)

scale_factor = 3.

model_name = 'regnet-epoch-{}-batch-{}-lr-{}-dim-{}-set-{}-loss-{}-cross-{}'.format(args.epoch,\
             args.batch_size, args.lr, args.dim, args.dataset, args.loss, args.cross)
print(model_name)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.dataset == 'bacterial':
	val_dim = 256; img_dim = 256
elif args.dataset == 'bone_marrow':
	val_dim = 608; img_dim = 600
elif args.dataset == 'colorectal':
	val_dim = 512; img_dim = 500
elif args.dataset == 'hESC':
	val_dim = 512; img_dim = 512

DATA_DIR = '/data/datasets/regnet/{}'.format(args.dataset) if args.docker else './datasets/regnet/{}'.format(args.dataset)
DATA_DIR = DATA_DIR+'/cross-{}'.format(args.cross)
x_train_dir = os.path.join(DATA_DIR, 'train', 'images')
y_train_dir = os.path.join(DATA_DIR, 'train', 'masks')

x_valid_dir = os.path.join(DATA_DIR, 'val', 'images')
y_valid_dir = os.path.join(DATA_DIR, 'val', 'masks')


print(x_train_dir); print(x_valid_dir)
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
    
    CLASSES = ['bk', 'cell']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            scale_factor = 3., 
            classes=None,
            nb_data=None,
            augmentation=None, 
            preprocessing=None,
    ):
        id_list = os.listdir(images_dir)
        if nb_data ==None:
            self.ids = id_list
        else:
            self.ids = id_list[:int(min(nb_data,len(id_list)))]
        #self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        #print(self.images_fps[:4]); print(self.masks_fps[:4])
        print(len(self.images_fps)); print(len(self.masks_fps))
        # convert str names to class values on masks
#         self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = io.imread(self.images_fps[i])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = io.imread(self.masks_fps[i])*scale_factor/255.
#         print(np.unique(mask))
        # extract certain classes from mask (e.g. cars)
#         masks = [(mask == v) for v in self.class_values]
#         print(self.class_values)
#         mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
#         if mask.shape[-1] != 1:
#             background = 1 - mask.sum(axis=-1, keepdims=True)
#             mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, np.expand_dims(mask, axis =-1)
        
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
#         map_batch = batch[1]
#         map_batch_list = [map_batch]
#         for i in range(4):
#             map_batch_list.append(map_batch[:,::2,::2,:])
#             map_batch = map_batch[:,::2,::2,:]
#         map_batch_list.reverse()
#         map_tuple = ()
#         for i in range(5):
#             map_tuple = map_tuple+(map_batch_list[i],)
        return (batch[0], batch[1])
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

import albumentations as A


# define heavy augmentations
def get_training_augmentation(dim = 256, rot_limit = 20):
    train_transform = [

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        A.ShiftScaleRotate(scale_limit=0, rotate_limit=rot_limit, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=dim, min_width=dim, always_apply=True, border_mode=0),
        A.RandomCrop(height=dim, width=dim, always_apply=True),

#         A.IAAAdditiveGaussianNoise(p=0.2),
#         A.IAAPerspective(p=0.5),

#         A.OneOf(
#             [
#                 A.CLAHE(p=1),
#                 A.RandomBrightness(p=1),
#                 A.RandomGamma(p=1),
#             ],
#             p=0.9,
#         ),

#         A.OneOf(
#             [
#                 A.IAASharpen(p=1),
#                 A.Blur(blur_limit=3, p=1),
#                 A.MotionBlur(blur_limit=3, p=1),
#             ],
#             p=0.9,
#         ),

#         A.OneOf(
#             [
#                 A.RandomContrast(p=1),
#                 A.HueSaturationValue(p=1),
#             ],
#             p=0.9,
#         ),
#         A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(dim = 256):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(dim, dim),
        A.RandomCrop(height=dim, width=dim, always_apply=True)
#         A.PadIfNeeded(384, 480)
    ]
    return A.Compose(test_transform)

# def get_preprocessing(preprocessing_fn):
#     """Construct preprocessing transform
#     
#     Args:
#         preprocessing_fn (callbale): data normalization function 
#             (can be specific for each pretrained neural network)
#     Return:
#         transform: albumentations.Compose
#     
#     """
#     
#     _transform = [
#         A.Lambda(image=preprocessing_fn),
#     ]
#     return A.Compose(_transform)


# BACKBONE = 'efficientnetb3'
BATCH_SIZE = args.batch_size
CLASSES = None
LR = args.lr
EPOCHS = args.epoch

# preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1 #if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
# activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
net_func = globals()[args.net_type]

# encoder_weights='imagenet' if args.pre_train else None
model = net_func(None, None, color_type = 3)

# model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

# define optomizer
optim = tf.keras.optimizers.Adam(LR)

# class_weights = [1,1,1,0.5]
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
# if args.loss =='focal+dice':
# 	dice_loss = sm.losses.DiceLoss(class_weights=np.array(class_weights))
# 	focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
# 	total_loss = dice_loss + (1 * focal_loss)
# elif args.loss =='dice':
# 	total_loss = sm.losses.DiceLoss(class_weights=np.array(class_weights))
# elif args.loss =='jaccard':
# 	total_loss = sm.losses.JaccardLoss(class_weights=np.array(class_weights))
# elif args.loss =='focal+jaccard':
# 	dice_loss = sm.losses.JaccardLoss(class_weights=np.array(class_weights))
# 	focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
# 	total_loss = dice_loss + (1 * focal_loss)
# elif args.loss =='focal+jaccard+dice':
# 	dice_loss = sm.losses.JaccardLoss(class_weights=np.array(class_weights))
# 	jaccard_loss = sm.losses.JaccardLoss(class_weights=np.array(class_weights))
# 	focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
# 	total_loss = dice_loss + jaccard_loss+ (1 * focal_loss)
# total_loss = sm.losses.BinaryFocalLoss()+ sm.losses.BinaryCELoss()
total_loss = sm.losses.wMSELoss(lamda = 1, beta =0.2)
# total_loss = tf.keras.losses.MSE
# 	total_loss = dice_loss + (1 * focal_loss)
# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

# metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
metrics = [sm.metrics.MSE(), sm.metrics.PSNR(max_val = scale_factor)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optimizer=optim, loss=total_loss, metrics = metrics)

# Dataset for train images
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    scale_factor = scale_factor,
    classes=CLASSES,
    augmentation=get_training_augmentation(args.dim, 20),
    preprocessing=None,
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    scale_factor = scale_factor,
    classes=CLASSES, 
    augmentation=get_validation_augmentation(val_dim),
    preprocessing=None,
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

print(train_dataloader[0][0].shape)
# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, args.dim, args.dim, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, args.dim, args.dim, n_classes)

model_folder = '/data/models/regnet/{}'.format(model_name) if args.docker else './models/regnet/{}'.format(model_name)
generate_folder(model_folder)


# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(model_folder+'/best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(),
]

# train model
history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)

# save the training information
plot_regress_history(model_folder+'/train_history.png',history)

# evaluate model
test_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    scale_factor = scale_factor,
    classes = CLASSES,
    augmentation=get_validation_augmentation(val_dim),
    preprocessing=None
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
#model = net_func(BACKBONE, encoder_weights=encoder_weights, input_shape = (test_dim, test_dim, 3), classes=n_classes, activation=activation)
# model.compile(optimizer=optim, loss=total_loss, metrics = metrics)

# load best weights
model.load_weights(model_folder+'/best_model.h5')
scores = model.evaluate_generator(test_dataloader)
print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))

# calculate the pixel-level classification performance
pr_masks = model.predict(test_dataloader);pr_masks= pr_masks.squeeze()
images = []; gt_masks = []
for i in range(len(test_dataset)):
    image, gt_mask = test_dataset[i]
    images.append(image); gt_masks.append(gt_mask)
images = np.stack(images)
gt_masks = np.stack(gt_masks).squeeze() #gt_maps = gt_masks.squeeze()

# scale back
gt_masks = gt_masks/scale_factor; pr_masks = pr_masks/scale_factor

# crop 
if args.dataset == 'bone_marrow' or args.dataset == 'colorectal':
	offset1, offset2 = int((val_dim-img_dim)/2), val_dim-int((val_dim-img_dim)/2)
	gt_masks=gt_masks[:,offset1:offset2,offset1:offset2]
	pr_masks=pr_masks[:,offset1:offset2,offset1:offset2]
	images=images[:,offset1:offset2,offset1:offset2]
	print('output: {}'.format(pr_masks.shape))

# save prediction examples
plot_fig_file = model_folder+'/pred_examples.png'; nb_images = 4
plot_reg_prediction(plot_fig_file, images, gt_masks, pr_masks, nb_images, rand_seed = 6)

# calculate PSNR
mPSNR, psnr_scores = calculate_psnr(gt_masks, pr_masks, max_val = 1.0)
print('PSNR: {:.4f}'.format(mPSNR))

with open(model_folder+'/metric_psnr.txt','w+') as f:
	# save PSNR over fluorescent 1 and fluorescent 2
	f.write('PSNR: {:.4f}'.format(mPSNR))