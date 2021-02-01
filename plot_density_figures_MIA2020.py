import os
import numpy as np
from skimage import io

def plot_prediction(file_name, images, titles, counts = []):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	font_size = 28
	rows, cols, size = 1, len(images), 5
	color_map = 'Blues'
	fig = Figure(tight_layout=True,figsize=(size*cols, (size+0.5)*rows)); ax = fig.subplots(rows,cols)
	ax[0].imshow(images[0]); ax[1].imshow(images[1], cmap =color_map);ax[2].imshow(images[2], cmap =color_map); ax[3].imshow(images[3], cmap =color_map); ax[4].imshow(images[4], cmap =color_map)
	ax[0].set_title(titles[0],fontsize=font_size, fontweight='bold'); ax[1].set_title(titles[1],fontsize=font_size, fontweight='bold'); ax[2].set_title(titles[2],fontsize=font_size, fontweight='bold')
	ax[3].set_title(titles[3],fontsize=font_size, fontweight='bold'); ax[4].set_title(titles[4],fontsize=font_size, fontweight='bold')
	ax[0].tick_params(axis='both', which='major', labelsize=font_size-8)
	ax[1].set_xticks([]);ax[2].set_xticks([]);ax[3].set_xticks([]);ax[4].set_xticks([])
	ax[1].set_yticks([]);ax[2].set_yticks([]);ax[3].set_yticks([]);ax[4].set_yticks([])
	if len(counts) >0:
# 		ax[0].set_xlabel('Ground truth count:{}'.format(counts[0]),fontsize=font_size-2)
		ax[1].set_xlabel('Count:{:.2f}'.format(counts[1]), fontsize=font_size-2, fontweight='bold')
		ax[2].set_xlabel('Count:{:.2f}'.format(counts[2]), fontsize=font_size-2, fontweight='bold')
		ax[3].set_xlabel('Count:{:.2f}'.format(counts[3]), fontsize=font_size-2, fontweight='bold')
		ax[4].set_xlabel('Count:{}'.format(round(counts[4])), fontsize=font_size-2, fontweight='bold')
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)

def gray2rgb(image):
	image_ = image.copy()
	image_ = 255.*(image_-image_.min())/(image_.max()-image_.min())
	image_rgb = np.stack([image_,image_,image_], axis = -1).astype(np.uint8)
	return image_rgb

# dataset = 'bacterial'
# cross = 3
# image_id = '018.npy'

# dataset = 'bone_marrow'
# cross = 3
# image_id = '002.npy'

# dataset = 'colorectal'
# cross = 1
# image_id = '007.npy'

# dataset = 'hESC'
# cross = 3
# image_id = '001.npy'

dataset_set = ['bacterial', 'bone_marrow', 'colorectal', 'hESC']

for dataset in dataset_set:
	if dataset == 'bacterial':
		cross = 3
		image_id = '018.npy'
	if dataset == 'bone_marrow':
		cross = 3
		image_id = '002.npy'
	if dataset == 'colorectal':
		cross = 1
		image_id = '007.npy'
	if dataset == 'hESC':
		cross = 3
		image_id = '001.npy'

	dataset_root_dir = '/home/sh38/cell_counting/datasets'
	result_root_dir = '/home/sh38/cell_counting/results'
	## Prediction examples for Mask-RCNN, U-Net and StructRegNet
	image_dir = result_root_dir+'/dcfcrn/{}/cross-{}/val'.format(dataset, cross)
	dcfcrn_dir = result_root_dir+'/dcfcrn/{}/cross-{}/val'.format(dataset, cross)
	fcrn_dir = result_root_dir+'/fcrn/{}/cross-{}/val'.format(dataset, cross)
	ception_dir = result_root_dir+'/ception/{}/cross-{}/val'.format(dataset, cross)
	gt_dir = result_root_dir+'/dcfcrn/{}/cross-{}/val'.format(dataset, cross)

	image = np.load(image_dir+'/img_{}'.format(image_id))
	#print(image.max(), image.min())
	image = np.uint8(255.*(image-image.min())/(image.max()-image.min()))
	fcrn_map = np.load(fcrn_dir+'/{}'.format(image_id))
	ception_map = np.load(ception_dir+'/{}'.format(image_id))
	dcfcrn_map = np.load(dcfcrn_dir+'/{}'.format(image_id))
	gt_map = np.load(dcfcrn_dir+'/gt_{}'.format(image_id))
	# den_map_rgb = gray2rgb(den_map)
	counts = [0, fcrn_map.sum(), ception_map.sum(), dcfcrn_map.sum(), gt_map.sum()]

	rgb_vis = False

	if rgb_vis:
		fcrn_map = gray2rgb(fcrn_map)
		ception_map = gray2rgb(ception_map)
		dcfcrn_map = gray2rgb(dcfcrn_map)
		gt_map = gray2rgb(gt_map)

	image_list = [image, fcrn_map, ception_map, dcfcrn_map, gt_map]
	title_list = ['Image', 'FCRN', 'Count-Ception', 'C-FCRN+Aux', 'Ground truth']

	plot_prediction(result_root_dir+'/density-{}-cross-{}-{}.png'.format(dataset, cross, image_id.split('.')[0]), image_list, title_list, counts)

