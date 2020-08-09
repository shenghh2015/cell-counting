import matplotlib.pyplot as plt
from data_load import *
from utils.data_processing import *
import helper_functions as hf
import matplotlib
import numpy as np
import os

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

# save the fluorescent embryonic cell images and density maps
root_folder = os.path.expanduser('~/dl-cells/dlct-framework/image_sets')
data_types = ['real_fluorescent', 'synthetic', 'MBM', 'CRCHistoPhenotypes']
version_list = [27, 24, 25, 26]
# version_list = [47, 44, 45, 46]

data_folder = os.path.join(root_folder,'real_fluorescent', 'images')
density_folder = os.path.join(root_folder,'real_fluorescent', 'images')

for i, data_type in enumerate(data_types):
	data_folder = os.path.join(root_folder,data_type, 'images')
	density_folder = os.path.join(root_folder,data_type, 'denisties')
	count_map_folder = os.path.join(root_folder,data_type, 'count_map')
	generate_folder(data_folder)
	generate_folder(density_folder)
	generate_folder(count_map_folder)
	# load the data
	version_idx = version_list[i]
	X_train, X_test, Y_train, Y_test = load_train_data(val_version = version_idx)
	X = np.concatenate([X_train, X_test], axis = 0)
	Y = np.concatenate([Y_train, Y_test], axis = 0)
	if 47 in version_list:
		X_pad, Y_map = hf.data_trans(X, Y)
	for j in range(X.shape[0]):
		file_name = os.path.join(data_folder,'img_{}.png'.format(j))
		den_name = os.path.join(density_folder,'den_{}.png'.format(j))
		if data_type == 'real_fluorescent':
# 			image = X[j,:,:,0]
			image = transfer2RGB_v2(X[j,:,:,0])
			print(image.shape)
		else:
			image = X[j,:]
		density = Y[j,:,:]
		matplotlib.image.imsave(file_name, image)
		matplotlib.image.imsave(den_name, density)
		if 47 in version_list:
			pad_image_file_name = os.path.join(count_map_folder,'img_{}.png'.format(j))
			count_map_file_name = os.path.join(count_map_folder,'count_{}.png'.format(j))
			if data_type == 'real_fluorescent':
				image_pad = X_pad[j,:,:,0]
			else:
				image_pad = X_pad[j,:]
			count_map = Y_map[j,:]
			matplotlib.image.imsave(pad_image_file_name, image_pad)
			matplotlib.image.imsave(count_map_file_name, count_map)

# plt.ion()
# fig = plt.figure()


