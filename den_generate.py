# script to generate density maps for MBM cell and synthetic bench mark data
# by shenghua
# date: April 27, 2018

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import misc
import scipy.stats as st
from scipy import signal
import glob
import natsort
import csv
from utils.file_load_save import *
# from utils.results_save import *
import re
import scipy.ndimage as ndimage

# a function that generates a normal 2D gaussian at (kernlen/2, kernlen/2) with std as standard deviation
def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d = gkern2d/np.sum(gkern2d)
    return gkern2d

# a function that generates a normal 2D gaussian at location (128,128) based on the basis gaussian
def gaussian_specify(img_dim, gauss_basis, location = (128, 128)):
	z_shp = (img_dim,img_dim)
	x0, x1 = 0 - location[0] + img_dim, z_shp[0] - location[0] + img_dim
	y0, y1 = 0 - location[1] + img_dim, z_shp[0] - location[1] + img_dim	
	z = gauss_basis[x0:x1,y0:y1]
	z = z/np.sum(z)
	return z

## load and process benchmark dataset
data_root_folder = os.path.expanduser('~/bio_cell_data')
dataset_folder = 'cell_count_benchmark'
sub_folder = 'cells'
# sub_folder = 'BM_dataset_MICCAI2015'
file_folder = os.path.join(data_root_folder, dataset_folder,sub_folder)

cell_files = glob.glob(file_folder + '/*cell.png')
annot_files = glob.glob(file_folder + '/*dots.png')

# cell_files = glob.glob(file_folder + '/source/*.png')
# annot_files = glob.glob(file_folder + '/annotations/*dots.png')

# load the cell images and annotated data
annot_list = []
image_list = []
for i, file in enumerate(annot_files):
	annot_list.append(misc.imread(annot_files[i]))
	image_list.append(misc.imread(cell_files[i]))

# density map generation
img_dim = annot_list[0].shape[0]
sigma = 1
gauss_basis = gkern(img_dim*2, sigma)
den_list = []
for j in range(len(annot_files)):
	den_map = np.zeros((img_dim, img_dim))
	annot = 100.0*(annot_list[j][:,:,0]>0)
# 	if len(annot.shape) > 2:
# 		annot = annot_list[j][:,:,0]
# 	xs, ys = np.nonzero(annot)
# 	for i in range(len(xs)):
# 		idx_X = xs[i]
# 		idx_Y = ys[i]
# 		k = gaussian_specify(img_dim, gauss_basis, location = (idx_X, idx_Y))
# 		den_map = den_map + k
	den_map = ndimage.gaussian_filter(annot, sigma=(3, 3), order=0)
	den_list.append(den_map)

# map the annotation to the cell images
def remarkRGBImgs(image, location = (256,256), value = 0, channel = 0):
	idx_X = int(location[0])
	idx_Y = int(location[1])
	image[idx_X, idx_Y-1:idx_Y+2,channel] = value
	image[idx_X-1:idx_X+2, idx_Y,channel] = value
	return image

rgbArr = np.array(image_list)
rgbImages = np.copy(rgbArr)
for j in range(len(annot_files)):
	annot = annot_list[j][:,:,0]
	xs, ys = np.nonzero(annot)
	for i in range(len(xs)):
		idx_X = xs[i]
		idx_Y = ys[i]
		rgbArr[j,:,:] = remarkRGBImgs(rgbArr[j,:,:],(idx_X,idx_Y), 200, 1)


# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

## save the processed data to the traindata folder
file_label_list = []
train_data_folder = os.path.join(data_root_folder,'train_data')
generate_folder(train_data_folder)
train_cell_data_folder = os.path.join(train_data_folder,dataset_folder)
generate_folder(train_cell_data_folder)
train_data_target_folder = os.path.join(train_cell_data_folder,sub_folder)
generate_folder(train_data_target_folder)
for i in range(len(rgbImages)):
	file_name = os.path.split(cell_files[i])[1]
	if sub_folder == 'cells':
		file_label = re.split(r'[\_,\.]', file_name)[-2]
	else:
		file_label = '{}_{}'.format(re.split(r'[\_,\.]', file_name)[-3],re.split(r'[\_,\.]', file_name)[-2])
	train_file_name = os.path.join(train_data_target_folder, '{}.pkl'.format(file_label))
	file_label_list.append(file_label)
	save_pickles(train_file_name, rgbImages[i], rgbArr[i], den_list[i], keys = ['cell', 'annot', 'den'])

fig2 = plt.figure()
for k in range(len(image_list)):
	fig2.clf()
	ax = fig2.add_subplot(1,3,1)
	bx = fig2.add_subplot(1,3,2)
	cx = fig2.add_subplot(1,3,3)
	ax.imshow(image_list[k])
	bx.imshow(rgbArr[k])
	cx.imshow(den_list[k])
	cx.set_xlabel('cell count:{}'.format(round(np.sum(den_list[k]))))
	plt.pause(0.5)
