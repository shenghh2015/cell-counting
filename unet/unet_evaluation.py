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
import albumentations as A
from unet_model import U_Net

from helper_function import generate_folder
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
docker = False
# scale_factor = 3.0
model_dir = './models/unet'
model_names = os.listdir(model_dir)
for model_name in model_names:
	print('\n')
	# model_name = model_names[0]
	#model_name = 'unet-epoch-5000-batch-64-lr-0.0001-dim-128-set-bacterial-loss-bce-cross-1'
	splits = model_name.split('-')
	for v in range(len(splits)):
		if splits[v] == 'set':
			dataset = splits[v+1]
		elif splits[v] == 'cross':
			cross = int(splits[v+1])
	print(model_name); print('dataset:{}'.format(dataset)); print('cross:{}'.format(cross))
	if dataset == 'bacterial':
		val_dim = 256; img_dim = 256
	elif dataset == 'bone_marrow':
		val_dim = 608; img_dim = 600
	elif dataset == 'colorectal':
		val_dim = 512; img_dim = 500
	elif dataset == 'hESC':
		val_dim = 512; img_dim = 512

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

	def get_validation_augmentation(dim = 256):
		"""Add paddings to make image shape divisible by 32"""
		test_transform = [
			A.PadIfNeeded(dim, dim),
			A.RandomCrop(height=dim, width=dim, always_apply=True)
		]
		return A.Compose(test_transform)

	CLASSES = ['cell']
	n_classes = 1 
	#create model
	net_type = 'U_Net'
	net_func = globals()[net_type]
	model = net_func(None, None, color_type = 3, num_class =n_classes)
	model_folder = '/data/models/unet/{}'.format(model_name) if docker else './models/unet/{}'.format(model_name)
	model_file = model_folder+'/best_model.h5'
	model.load_weights(model_file)
	print('Load model: {}'.format(model_file))

	DATA_DIR = '/data/datasets/unet/{}'.format(dataset) if docker else './datasets/unet/{}'.format(dataset)
	DATA_DIR = DATA_DIR+'/cross-{}'.format(cross)

	subsets = ['train', 'val']
	for subset in subsets:
		# subset = subsets[0]
		x_valid_dir = os.path.join(DATA_DIR, subset, 'images')
		y_valid_dir = os.path.join(DATA_DIR, subset, 'masks')
		pred_valid_dir = os.path.join(DATA_DIR, subset, 'pr_masks')
		pred_valid_time_dir = os.path.join(DATA_DIR, subset, 'pr_times') # average second/image

		## loading datasets and prediction
		test_dataset = Dataset(
			x_valid_dir, 
			y_valid_dir, 
			classes = CLASSES,
			augmentation=get_validation_augmentation(val_dim),
			preprocessing=None
		)
		test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
		start_time = time.time()
		pr_masks = model.predict(test_dataloader);pr_masks= pr_masks[:,:,:,0].squeeze()
		end_time = time.time()
		print('Data scales: min {:.4f}, max {:.4f}'.format(pr_masks.min(), pr_masks.max()))
		# crop and save prediction maps
		if dataset == 'bone_marrow' or dataset == 'colorectal':
			offset1, offset2 = int((val_dim-img_dim)/2), val_dim-int((val_dim-img_dim)/2)
			pr_masks=pr_masks[:,offset1:offset2,offset1:offset2]
		print('output: {}'.format(pr_masks.shape))

		# save data as numpy data
		generate_folder(pred_valid_dir)
		generate_folder(pred_valid_time_dir)
		for index in range(len(test_dataset)):
			# save data
			npy_file_name = pred_valid_dir+'/{}.npy'.format(test_dataset.ids[index].split('.')[0])
			if index%(len(test_dataset)-1) == 0:
				print('Save prediction to:{}'.format(npy_file_name))
			np.save(npy_file_name, pr_masks[index])

		# load the saved data to check the integrity
		pred_masks = []
		for index in range(len(test_dataset)):
			npy_file_name = pred_valid_dir+'/{}.npy'.format(test_dataset.ids[index].split('.')[0])
			pred_masks.append(np.load(npy_file_name))
		pred_masks = np.stack(pred_masks); print('Loaded scale: min {:.4f}, max {:.4f}'.format(pred_masks.min(),pred_masks.max()))
		print('Save and load difference: {:.4f}'.format(np.abs(pred_masks-pr_masks).mean()))
		
		# save inference time
		dur_time = (end_time - start_time)/len(test_dataset)
		np.save(pred_valid_time_dir+'/ave_time_per_image.npy', dur_time)
		print('Average inference time: {:.4f}'.format(dur_time))