import os
import numpy as np
import cv2
import scipy.ndimage as ndimage
import tensorflow as tf
import albumentations as A
import argparse

from models import *
from helper_function import plot_history

def gen_dir(folder):
    if not os.path.exists(folder):
        os.system('mkdir -p {}'.format(folder))

def save_metrics(file_name, metrics, scores):
    with open(file_name, 'w+') as f:
        for i in range(len(metrics)):
            f.write('{}:{:3f}\n'.format(metrics[i], scores[i]))

data_root = os.path.abspath('../')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type = str, default= '0')
parser.add_argument('--net', type = str, default= 'FCRN_A')
parser.add_argument('--dataset', type = str, default= 'bacterial')
parser.add_argument('--epochs', type = int, default = 500)
parser.add_argument('--batch', type = int, default= 32)
parser.add_argument('--dim', type = int, default= 256)
parser.add_argument('--lr', type = float, default = 1e-4)
parser.add_argument('--rf', type = float, default = 0.9)
parser.add_argument('--w', type = int, default = 1)

args = parser.parse_args()

dataset = args.dataset
net = args.net
batch_size = args.batch
epochs = args.epochs
lr = args.lr
rf =  args.rf             # learning rate reducing factor            
gpu = args.gpu
dim = args.dim
weight = args.w
deeply = True if net == 'C_FCRN_Aux' else False

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

if dataset == 'bacterial':
    val_dim = 256
    sigma = 3
elif dataset == 'BMC':
    val_dim = 608
    sigma = 5
elif dataset == 'CCC':
    val_dim = 512
    sigma = 3
elif dataset == 'hESC':
    val_dim = 512
    sigma = 3

model_name = 'net-{}-set-{}-bt-{}-ep-{}-lr-{}-sig-{}-dim-{}-plt-w-{}'.format(net, dataset, batch_size, epochs, lr, sigma, dim, weight)
dataset_dir = data_root + '/datasets/{}'.format(dataset)
model_dir = data_root + '/models/{}/{}'.format(dataset, model_name)
gen_dir(dataset_dir)
gen_dir(model_dir)

img_dir = dataset_dir + '/images/'
dot_dir = dataset_dir + '/dots/'

# prepare the training and validation set
sample_names = os.listdir(dataset_dir + '/images')
np.random.seed(0)
np.random.shuffle(sample_names)
n_train = int(len(sample_names) * 2./ 3)
train_samples = sample_names[:n_train]
valid_samples = sample_names[n_train:]
print('Train: {}, valid {}'.format(len(train_samples), len(valid_samples)))

class Dataset:
    """ datasets for model training and validation
    
    Args:
        img_dir (str): path to images folder
        dot_dir (str): path to dot annotation
        augmentation (albumentations.Compose): data transfromation pipeline
    """
    
    def __init__(
            self, 
            img_dir,
            dot_dir,
            sample_names,
            augmentation=None,
    ):  
        self.images_fps = [os.path.join(img_dir, sn) for sn in sample_names]
        self.dot_fps = [os.path.join(dot_dir, sn) for sn in sample_names]
        self.ids = self.images_fps
        print('Load files: image {}, dot files: {}'.format(len(self.images_fps),len(self.dot_fps)))
        self.augmentation = augmentation
    
    def __getitem__(self, i):
        
        # load images and dot maps
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_COLOR)
        dot_map = (cv2.imread(self.dot_fps[i], cv2.IMREAD_GRAYSCALE) > 0) * 1.

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=dot_map)
            image, dot_map = sample['image'], sample['mask']

        # generate density map
        den_map = ndimage.gaussian_filter(dot_map, sigma=(3, 3), order=0) * 100

        # apply preprocessing
        # if self.preprocessing:
        #     sample = self.preprocessing(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']

        return image, np.expand_dims(den_map, axis = -1)
        
    def __len__(self):
        return len(self.ids)

class Dataloder(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False, deeply = False):
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
        
        den_map = batch[1]
        den_tuple = den_map
        if deeply:
            den_tuple = [den_map]
            for i in range(3):
                cur_map = den_tuple[-1].copy()
                nex_map = cur_map[:, 0::2, 0::2, :] + cur_map[:, 1::2, 0::2, :] + cur_map[:, 0::2, 1::2, :] + cur_map[:, 1::2, 1::2, :]
                den_tuple.append(nex_map)
            den_tuple.reverse()

        return (batch[0], den_tuple)
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation(dim):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0., rotate_limit=30, shift_limit=0.1, p=1, border_mode=0),
        A.PadIfNeeded(min_height=dim, min_width=dim, always_apply=True, border_mode=0),
        A.RandomCrop(height=dim, width=dim, always_apply=True),]
    return A.Compose(train_transform)

def get_validation_augmentation(dim = 256):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(dim, dim)
    ]
    return A.Compose(test_transform)

model = globals()[net](input_shape = (None, None, 3))
optim = tf.keras.optimizers.Adam(lr)
loss = tf.keras.losses.MSE
from tensorflow.keras import backend as K

def mce(y_true, y_pred):
    x = tf.reduce_sum(y_true, axis = [1, 2, 3])
    y = tf.reduce_sum(y_pred, axis = [1, 2, 3])
    return tf.reduce_mean(tf.math.abs(x - y)) / 100.
metrics = [mce]

train_dataset = Dataset(img_dir = img_dir, 
                        dot_dir = dot_dir, 
                        sample_names = train_samples, 
                        augmentation= get_training_augmentation(dim = dim)
                        )
valid_dataset = Dataset(img_dir = img_dir, 
                        dot_dir = dot_dir, 
                        sample_names = valid_samples, 
                        augmentation= get_training_augmentation(dim = val_dim)
                        )
print(train_dataset[0][0].shape, train_dataset[0][1].shape, 'count: {:.2f}'.format(train_dataset[0][1].sum()))

train_dataloader = Dataloder(train_dataset, batch_size = batch_size, shuffle = True, deeply = deeply)
valid_dataloader = Dataloder(valid_dataset, batch_size = 1, shuffle = False, deeply = deeply)

if deeply:
    print(train_dataloader[0][0].shape, train_dataloader[0][1][0].shape, 'count: {:.2f}'.format(train_dataloader[0][1][0][1, :, :, 0].sum()))
else:
    print(train_dataloader[0][0].shape, train_dataloader[0][1].shape, 'count: {:.2f}'.format(train_dataloader[0][1].sum()))

if deeply:
    if rf < 1:
        callbacks = [
                    tf.keras.callbacks.ModelCheckpoint(model_dir+'/best_model-{epoch:03d}.h5', monitor='val_original_mce', save_weights_only=True, save_best_only=True, mode='min'),
                    tf.keras.callbacks.ReduceLROnPlateau(factor = rf),
        ]
    else:
        callbacks = [
                    tf.keras.callbacks.ModelCheckpoint(model_dir+'/best_model-{epoch:03d}.h5', monitor='val_original_mce', save_weights_only=True, save_best_only=True, mode='min'),
        ]
    if weight == 1:
        loss_weights=[1./64, 1/16, 1./4, 1]
    elif weight == 2:
        loss_weights=[1., 1., 1., 1.]
    elif weight == 3:
        loss_weights=[1./128, 1./32, 1./8, 1.]
    model.compile(optim, loss, metrics, loss_weights = loss_weights)
else:
    if rf < 1:
        callbacks = [
                    tf.keras.callbacks.ModelCheckpoint(model_dir+'/best_model-{epoch:03d}.h5', monitor='val_mce', save_weights_only=True, save_best_only=True, mode='min'),
                    tf.keras.callbacks.ReduceLROnPlateau(factor = rf),
        ]
    else:
        callbacks = [
                    tf.keras.callbacks.ModelCheckpoint(model_dir+'/best_model-{epoch:03d}.h5', monitor='val_mce', save_weights_only=True, save_best_only=True, mode='min'),
        ]
    model.compile(optim, loss, metrics)

history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=epochs, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)

t_epochs = 10
if epochs > t_epochs:
    plot_history(model_dir + '/train.png', history, deeply, t_epochs)
# validate the performance
model_names = sorted([mn for mn in os.listdir(model_dir) if mn.endswith('.h5')])
best_model_name = model_names[-1]
model.load_weights(model_dir + '/' + best_model_name)
scores = model.evaluate_generator(valid_dataloader)
print(model.metrics_names)
print(scores)
file_name = model_dir + '/val_mce.txt'
save_metrics(file_name, model.metrics_names, scores)
