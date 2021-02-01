import os
import numpy as np
from skimage import io

def plot_prediction(file_name, images, titles, counts = []):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	font_size = 28
	rows, cols, size = 1, len(images), 5
	fig = Figure(tight_layout=True,figsize=(size*cols, (size+0.5)*rows)); ax = fig.subplots(rows,cols)
	ax[0].imshow(images[0]); ax[1].imshow(images[1]);ax[2].imshow(images[2])
	ax[0].set_title(titles[0],fontsize=font_size,fontweight='bold'); ax[1].set_title(titles[1],fontsize=font_size,fontweight='bold')
	ax[2].set_title(titles[2],fontsize=font_size,fontweight='bold')
	ax[0].set_xticks([]);ax[1].set_xticks([]);ax[2].set_xticks([])
	ax[0].set_yticks([]);ax[1].set_yticks([]);ax[2].set_yticks([])
	if len(counts) >0:
# 		ax[0].set_xlabel('Ground truth count:{}'.format(counts[0]),fontsize=font_size-2)
		ax[1].set_xlabel('Count:{:.2f}'.format(counts[1]), fontsize=font_size-2,fontweight='bold')
		ax[2].set_xlabel('Count:{}'.format(counts[2]), fontsize=font_size-2,fontweight='bold')
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)

def gray2rgb(image):
	image_ = image.copy()
	image_ = 255.*(image_-image_.min())/(image_.max()-image_.min())
	image_rgb = np.stack([image_,image_,image_], axis = -1).astype(np.uint8)
	return image_rgb

# dataset = 'colorectal'
# cross = 2
# cross2 = 5
# image_id = '009.png'
# image_id2 = '016.png'
# counts = [712, 137, 796]  ## hard-coded

dataset = 'hESC'
cross = 2
cross2 = 2
image_id = '004.png'
image_id2 = '004.png'
counts = [1100, 137, 755]  ## hard-coded

dataset_root_dir = '/home/sh38/cell_counting/datasets'
result_root_dir = '/home/sh38/cell_counting/results'
## Prediction examples for Mask-RCNN, U-Net and StructRegNet
image_dir = dataset_root_dir+'/regnet/{}/cross-{}/val/images'.format(dataset, cross)
dcfrn_dir = result_root_dir+'/dcfcrn/{}/cross-{}/val'.format(dataset, cross2)
regnet_dir = result_root_dir+'/regnet/{}/cross-{}/val'.format(dataset, cross)

image = io.imread(image_dir+'/{}'.format(image_id))
den_map = np.load(dcfrn_dir+'/{}'.format(image_id2.replace('.png','.npy')))
den_map_rgb = gray2rgb(den_map)
counts[1] = den_map.sum() 
reg_map = io.imread(regnet_dir+'/pr_{}'.format(image_id))

image_list = [image, den_map_rgb, reg_map]
title_list = ['Image', 'C-Aux(Density)', 'RegNet(Detection)']

plot_prediction(result_root_dir+'/density-detection-{}-cross-{}-{}.png'.format(dataset, cross, image_id.split('.')[0]), image_list, title_list, counts)

