import os
import numpy as np
from skimage import io
from helper_function import generate_folder

docker = True
# dataset = 'live_dead'
dataset = 'cell_cycle2'
dataset_dir = '/data/datasets/{}'.format(dataset) if docker else ''
down_factor = 2

down_dataset_dir = '/data/datasets/{}/down_x{}'.format(dataset, down_factor) if docker else ''
generate_folder(down_dataset_dir)

subsets = ['train', 'test', 'val']

for subset in subsets:
# 	subset = subsets[0]
	print('>>>> processing subset {}'.format(subset))
	if dataset == 'live_dead':
		image_folder = dataset_dir +'/{}_images2'.format(subset)
	else:
		image_folder = dataset_dir +'/{}_images'.format(subset)

	mask_folder = dataset_dir +'/{}_masks'.format(subset)

	## generate subset folders
	down_image_folder = os.path.join(down_dataset_dir,'{}_images'.format(subset)); generate_folder(down_image_folder)
	down_mask_folder = os.path.join(down_dataset_dir,'{}_masks'.format(subset)); generate_folder(down_mask_folder)

	image_names = os.listdir(image_folder)
	for i in range(len(image_names)):
		if i%100 ==0:
			print('>>>> {}:{}-th image'.format(subset,i))
		img_name = image_names[i]
		image_file = image_folder+'/'+img_name; mask_file = mask_folder+'/'+img_name
		image = io.imread(image_file); mask = io.imread(mask_file)  # read image and mask
		down_image_file = down_image_folder+'/'+img_name; down_mask_file = down_mask_folder+'/'+img_name
		io.imsave(down_image_file, image[::2,::2,:]);io.imsave(down_mask_file, mask[::2,::2])