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
	ax[0].set_title(titles[0],fontsize=font_size,fontweight='bold')
	ax[1].set_title(titles[1],fontsize=font_size,fontweight='bold')
	ax[2].set_title(titles[2],fontsize=font_size,fontweight='bold')
	ax[0].set_xticks([]);ax[1].set_xticks([]);ax[2].set_xticks([])
	ax[0].set_yticks([]);ax[1].set_yticks([]);ax[2].set_yticks([])
	if len(counts) >0:
		ax[0].set_xlabel('Count:{:.2f}'.format(counts[0]),fontsize=font_size-2,fontweight='bold')
		ax[1].set_xlabel('Count:{:.2f}'.format(counts[1]), fontsize=font_size-2,fontweight='bold')
		ax[2].set_xlabel('Count:{}'.format(round(counts[2])), fontsize=font_size-2,fontweight='bold')
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)

def gray2rgb(image):
	image_ = image.copy()
	image_ = 255.*(image_-image_.min())/(image_.max()-image_.min())
	image_rgb = np.stack([image_,image_,image_], axis = -1).astype(np.uint8)
	return image_rgb

dataset = 'bacterial'
cross = 2
image_id = '017.png'
counts = [208, 233, 277]  ## hard-coded

# dataset = 'bone_marrow'
# cross = 1
# image_id = '000.png'
# counts = [53, 137, 104]  ## hard-coded

# dataset_root_dir = '/home/sh38/cell_counting/datasets'
result_root_dir = '/home/sh38/cell_counting/results'
## Prediction examples for Mask-RCNN, U-Net and StructRegNet
mrcnn_dir = result_root_dir+'/mrcnn/{}/cross-{}/'.format(dataset, cross)
unet_dir = result_root_dir+'/unet/{}/cross-{}/val'.format(dataset, cross)
regnet_dir = result_root_dir+'/regnet/{}/cross-{}/val'.format(dataset, cross)

m_image = io.imread(mrcnn_dir+'/{}'.format(image_id))
u_image = io.imread(unet_dir+'/{}'.format(image_id))
r_image = io.imread(regnet_dir+'/{}'.format(image_id))

image_list = [m_image, u_image, r_image]
title_list = ['Mask R-CNN', 'U-Net', 'StructRegNet']

plot_prediction(result_root_dir+'/detection-{}-cross-{}-{}.png'.format(dataset, cross, image_id.split('.')[0]), image_list, title_list, counts)

