import glob
import matplotlib.pyplot as plt
import os
import numpy as np

from utils.file_load_save import *

result_folder = os.path.expanduser('~/dl-cells/dlct-framework/results')
model_folder = 'date-4.29-FCRN_A-mse_ct_err-v-24'
estimte_folder = os.path.join(result_folder, model_folder, '*.pkl')
file_names = glob.glob(estimte_folder)
est_list = []
den_list = []
img_list = []
for i, f in enumerate(file_names):
	file_name = file_names[i]
	est, den, img = read_pickles(file_name, keys = ['est_den', 'gt_den', 'ori_img'])
	est_list.append(est)
	den_list.append(den)
	img_list.append(img)
	
fig = plt.figure()
plt.ion()
fig.clf()
for i in range(len(est_list)):
	ax = fig.add_subplot(1,3,1)
	bx = fig.add_subplot(1,3,2)
	cx = fig.add_subplot(1,3,3)
	ax.imshow(np.squeeze(img_list[i]))
	bx.imshow(np.squeeze(den_list[i]))
	cx.imshow(np.squeeze(est_list[i]))
	bx.set_xlabel('cell count:{0:0.3f}'.format(np.sum(den_list[i])))
	cx.set_xlabel('cell count:{0:0.3f}'.format(np.sum(est_list[i])))
	plt.pause(0.5)


## draft to show figures 02-26-2019
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
from data_load import *
import skimage.measure as measure
# import scipy.misc as misc
from skimage.transform import rescale
Xtr, Xtst, Ytr, Ytst = load_train_data(val_version=27, cross_val = 0, normal_proc = False)
plt.ion()
fig = plt.figure()
fig.clf()
ax = fig.add_subplot(221)
bx = fig.add_subplot(222)
cx = fig.add_subplot(223)
dx = fig.add_subplot(224)
ax.imshow(Ytr[0,:,:])
Y = Ytr[0:2,:,:]
# Y1 = skimage.measure.block_reduce(Y, (8,8), np.mean)
# bx.imshow(Y1)
# Y1_ = Y - misc.imresize(Y1, 8.0)
# cx.imshow(Y1_)
# Y2 = skimage.measure.block_reduce(Y1_, (4,4), np.mean)
# Y2_ = Y1_ - misc.imresize(Y2, 4.0)
# Y3 = skimage.measure.block_reduce(Y2_, (2,2), np.mean)
# Y3_ = Y2_ - misc.imresize(Y3, 2.0)
# Y4 = Y3_
# Yc = 64*misc.imresize(Y1, 8.0) + 16**misc.imresize(Y2, 4.0) + 4*misc.imresize(Y3, 2.0) + Y4

Y4 = Y - rescale(measure.block_reduce(Y, (2,2), np.mean), 2.0)
Y3 = measure.block_reduce(Y, (2,2), np.mean) - rescale(measure.block_reduce(Y, (4,4), np.mean), 2.0)
Y2 = measure.block_reduce(Y, (4,4), np.mean) - rescale(measure.block_reduce(Y, (8,8), np.mean), 2.0)
Y1 = measure.block_reduce(Y, (8,8), np.mean)

Yc = rescale(Y1,8) + rescale(Y2, 4.0) + rescale(Y3, 2.0) + Y4

Y4 = Y - rescale(measure.block_reduce(Y, (2,2), np.mean), 2.0)
Y3 = measure.block_reduce(Y, (2,2), np.mean) - rescale(measure.block_reduce(Y, (4,4), np.mean), 2.0)
Y2 = measure.block_reduce(Y, (4,4), np.mean) - rescale(measure.block_reduce(Y, (8,8), np.mean), 2.0)
Y1 = measure.block_reduce(Y, (8,8), np.mean) 

