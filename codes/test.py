import os
import numpy as np
import cv2
import scipy.ndimage as ndimage
import tensorflow as tf
import albumentations as A
import argparse

from models import *

def gen_dir(folder):
    if not os.path.exists(folder):
        os.system('mkdir -p {}'.format(folder))

def save_metrics(file_name, metrics, scores):
    with open(file_name, 'w+') as f:
        for i in range(len(metrics)):
            f.write('{}:{:3f}\n'.format(metrics[i], scores[i]))

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default= '')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
data_root = os.path.abspath('../')
model_root = data_root + '/models/'

model_name = args.model_name
# model_name = 'net-C_FCRN_Aux-set-bacterial-bt-32-ep-400-lr-0.0001-sig-3-dim-256-plt-w-1'

seps = model_name.split('-')

for i, sp in enumerate(seps):
    if sp == 'net':
        net = seps[i + 1]
    elif sp == 'set':
        dataset = seps[i + 1]

deeply = True if net == 'C_FCRN_Aux' else False

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

dataset_dir = data_root + '/datasets/{}'.format(dataset)
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

def get_validation_augmentation(dim = 256):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(dim, dim)
    ]
    return A.Compose(test_transform)

model = globals()[net](input_shape = (None, None, 3))
optim = tf.keras.optimizers.Adam(0.0001)
loss = tf.keras.losses.MSE
from tensorflow.keras import backend as K

def mce(y_true, y_pred):
    x = tf.reduce_sum(y_true, axis = [1, 2, 3])
    y = tf.reduce_sum(y_pred, axis = [1, 2, 3])
    return tf.reduce_mean(tf.math.abs(x - y)) / 100.
metrics = [mce]

valid_dataset = Dataset(img_dir = img_dir, 
                        dot_dir = dot_dir, 
                        sample_names = valid_samples, 
                        augmentation= get_validation_augmentation(dim = val_dim)
                        )

valid_dataloader = Dataloder(valid_dataset, batch_size = 1, shuffle = False, deeply = deeply)

if deeply:
    print(valid_dataloader[0][0].shape, valid_dataloader[0][1][0].shape, 'count: {:.2f}'.format(valid_dataloader[0][1][0][0, :, :, 0].sum()))
else:
    print(valid_dataloader[0][0].shape, valid_dataloader[0][1].shape, 'count: {:.2f}'.format(valid_dataloader[0][1].sum()))

if deeply:
    loss_weights=[1./64, 1/16, 1./4, 1]
    model.compile(optim, loss, metrics, loss_weights = loss_weights)
else:
    model.compile(optim, loss, metrics)

model_dir = model_root + '/{}/{}'.format(dataset, model_name)
print(model_dir)
# validate the performance
model_names = sorted([mn for mn in os.listdir(model_dir) if mn.endswith('.h5')])
best_model_name = model_names[-1]
model.load_weights(model_dir + '/' + best_model_name)
scores = model.evaluate_generator(valid_dataloader)
print(model.metrics_names)
print(scores)
file_name = model_dir + '/test_mce.txt'
save_metrics(file_name, model.metrics_names, scores)
