from helper_functions import *
from data_load import *

import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc as misc
import scipy.ndimage as ndimage

## load the training data
dat_version = 45
X_train,X_val,Y_train,Y_val = load_train_data(val_version=dat_version, cross_val = 0, normal_proc = False)
_, _, Y_dens, _ = load_train_data(val_version = 25,cross_val = 0, normal_proc = False)

## select a sample to show, which includes the figure, the annotation dots and the genereated density maps
img_idx = 0
image = X_train[img_idx,:]
dot_map = Y_train[img_idx,:]
den_map = Y_dens[img_idx,:]
# cropping to a image with 128x128 spatial size
img_obj = Image.fromarray(image)
dot_map_obj = Image.fromarray(dot_map)
den_map_obj = Image.fromarray(den_map)
# size_tuple = (127,127,255,255)  # the diagonal positions for the cropped region
shift_indx = 0
img_size = 256
size_tuple = (shift_indx,shift_indx,shift_indx+img_size,shift_indx+img_size)  # the diagonal positions for the cropped region
crop_img_obj =  img_obj.crop(size_tuple)
crop_dot_map_obj = dot_map_obj.crop(size_tuple)
crop_den_map_obj = den_map_obj.crop(size_tuple)

crop_img_obj.save("image_for_paper/image.png")
## 
# save the dot map
crop_dot_map = np.array(crop_dot_map_obj)
crop_dot_map = ndimage.binary_dilation(crop_dot_map, iterations = 2)*1.0
crop_dot_map = np.expand_dims(crop_dot_map, axis=2)
shp = crop_dot_map.shape
crop_dot_map = np.concatenate([np.zeros(shp),crop_dot_map,np.zeros(shp)], axis = 2)
annot_img = crop_dot_map*255 + np.array(crop_img_obj) * (1-crop_dot_map)
crop_dot_map = np.uint8((crop_dot_map*255))
# annot_img = (annot_img >255)*255 + (annot_img <= 255)*annot_img
annot_img = np.uint8(annot_img)
# for i in range(annot_img.shape[2]):
# 	annot_img[:,:,i] = np.min(annot_img[:,:,i], 255)
crop_dot_map_obj = Image.fromarray(crop_dot_map)
annot_img_obj = Image.fromarray(annot_img)
crop_dot_map_obj.save("image_for_paper/dots.png")
annot_img_obj.save("image_for_paper/annot_img.png")

# save the density map
crop_den_map = np.array(crop_den_map_obj)
crop_den_map = np.expand_dims(crop_den_map, axis=2)
shp = crop_den_map.shape
crop_den_map = np.concatenate([crop_den_map, crop_den_map, crop_den_map], axis = 2)
crop_den_map = (crop_den_map - np.min(crop_den_map))/(np.max(crop_den_map)-np.min(crop_den_map))* 255
crop_den_map = np.uint8(crop_den_map)
crop_den_obj = Image.fromarray(crop_den_map)
crop_den_obj.save("image_for_paper/density.png")

## generate the low resolution density map
## visulation of the density maps
fig = plt.figure()
plt.clf()
plt.ion()
ax = fig.add_subplot(131)
bx = fig.add_subplot(132)
cx = fig.add_subplot(133)
# ax.imshow(crop_img_obj)
ax.imshow(annot_img_obj)
bx.imshow(crop_dot_map_obj)
cx.imshow(crop_den_map_obj)
