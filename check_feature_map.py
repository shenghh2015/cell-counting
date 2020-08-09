from keras.models import load_model
import keras.backend as K
import keras.losses
from keras.regularizers import l2
from keras.utils import plot_model

import os
import glob
import os
import time
import numpy as np
import pickle
from natsort import natsorted

from models import *
from data_load import *
from utils.file_load_save import *
from utils.metrics import *
from utils.plot_function import *
import helper_functions as hf
from utils.loss_fns import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
keras.losses.mse_err = mse_err
keras.losses.mse_ct_err = mse_ct_err
models_root_folder = os.path.expanduser('~/dl-cells/dlct-framework/models')
# fnc_str = 'buildModel_FCRN_A'
# loss_fn = 'mse'
# date = '5.12'
fnc_str = 'buildModel_MCNN_U'
loss_fn = 'mse'
date = '5.13'
data_version = 25
cross_nb = 0
model_folders_ptrn = 'date-{}-{}-mse*v-{}*norm*'.format(date,fnc_str,data_version)
final_model_folders_ptrn = os.path.join(models_root_folder, model_folders_ptrn)
model_folders = glob.glob(final_model_folders_ptrn)

model_folder = model_folders[-1]
import glob
fcns = globals()['{}'.format(fnc_str)]
model_file_list = glob.glob(model_folder +'/weights*.h5')
# 	print(model_file_list)
model_file_list = natsorted(model_file_list)
acc_list = []
mae_list = []
std_list = []
input_shape = (608,608)
# 	input_shape = (256,256)
model = fcns(input_shape)
model_file = model_file_list[-1]
model.load_weights(model_file, by_name = True)
# load data with data version and cross validation number
X_train,X_val,Y_train,Y_val = load_train_data(val_version = data_version, cross_val = cross_nb, normal_proc = True)
shp = X_val.shape
if shp[1:3] == (600,600):
	X_zap = np.zeros((shp[0],)+input_shape+(3,))
	X_zap[:,4:-4,4:-4,:] = X_val
	X_val = X_zap
if len(X_val.shape)<3:
	X_val = X_val.reshape(X_val.shape+(1,))
time1 = time.time()
x_len = X_val.shape[0]

# plot_model(model, to_file='model.png')

plt.ion()
fig = plt.figure()

# for buildModel_MCNN_U
model_feat1 = Model(input=model.input, output=model.get_layer('activation_12').output)
model_feat2 = Model(input=model.input, output=model.get_layer('activation_11').output)
model_feat3 = Model(input=model.input, output=model.get_layer('activation_10').output)
model_feat4 = Model(input=model.input, output=model.get_layer('activation_3').output)
model_feat5 = Model(input=model.input, output=model.get_layer('activation_6').output)
model_feat6 = Model(input=model.input, output=model.get_layer('activation_9').output)
model_feat7 = Model(input=model.input, output=model.get_layer('activation_13').output)

# for buildModel_FCRN_A
model_feat1 = Model(input=model.input, output=model.get_layer('activation_1').output)
model_feat2 = Model(input=model.input, output=model.get_layer('activation_2').output)
model_feat3 = Model(input=model.input, output=model.get_layer('activation_3').output)
model_feat4 = Model(input=model.input, output=model.get_layer('activation_4').output)
model_feat5 = Model(input=model.input, output=model.get_layer('activation_5').output)
model_feat6 = Model(input=model.input, output=model.get_layer('activation_6').output)
model_feat7 = Model(input=model.input, output=model.get_layer('activation_7').output)

fig.clf()
k = 0

# X_val = X_train
# Y_val = Y_train
shp = X_val.shape
if shp[1:3] == (600,600):
	X_zap = np.zeros((shp[0],)+input_shape+(3,))
	X_zap[:,4:-4,4:-4,:] = X_val
	X_val = X_zap
if len(X_val.shape)<3:
	X_val = X_val.reshape(X_val.shape+(1,))
time1 = time.time()
x_len = X_val.shape[0]
den = model.predict(X_val[k:k+1,:])/100
den1 = model_feat1.predict(X_val[k:k+1,:])/100
den2 = model_feat2.predict(X_val[k:k+1,:])/100
den3 = model_feat3.predict(X_val[k:k+1,:])/100
den4 = model_feat4.predict(X_val[k:k+1,:])/100
den5 = model_feat6.predict(X_val[k:k+1,:])/100
den6 = model_feat6.predict(X_val[k:k+1,:])/100
lres_features = model_feat7.predict(X_val[k:k+1,:])/100

ax = fig.add_subplot(3,3,1)
bx = fig.add_subplot(3,3,2)
cx = fig.add_subplot(3,3,3)
dx = fig.add_subplot(3,3,4)
ex = fig.add_subplot(3,3,5)
fx = fig.add_subplot(3,3,6)
gx = fig.add_subplot(3,3,7)
hx = fig.add_subplot(3,3,8)
ix = fig.add_subplot(3,3,9)

## features at different layers
image = X_val[k,:]
image = (image- np.min(image))/(np.max(image)-np.min(image))
ax.imshow(image)
bx.imshow(Y_val[k,:])
cx.imshow(np.squeeze(den))
dx.imshow(np.squeeze(den1[0,:,:,0]))
ex.imshow(np.squeeze(den2[0,:,:,0]))
fx.imshow(np.squeeze(den3[0,:,:,0]))
gx.imshow(np.squeeze(den4[0,:,:,0]))
hx.imshow(np.squeeze(den5[0,:,:,0]))
ix.imshow(np.squeeze(den6[0,:,:,0]))

## features at a specific layer
ax.imshow(image)
bx.imshow(Y_val[k,:])
cx.imshow(np.squeeze(den))
dx.imshow(np.squeeze(den4[0,:,:,0]))
ex.imshow(np.squeeze(den4[0,:,:,1]))
fx.imshow(np.squeeze(den4[0,:,:,2]))
gx.imshow(np.squeeze(den4[0,:,:,3]))
hx.imshow(np.squeeze(den4[0,:,:,4]))
ix.imshow(np.squeeze(den4[0,:,:,5]))

## 512 low resolution feature maps
ax.imshow(lres_features[0,:,:,0])
bx.imshow(lres_features[0,:,:,10])
cx.imshow(lres_features[0,:,:,20])
dx.imshow(lres_features[0,:,:,30])
ex.imshow(lres_features[0,:,:,40])
fx.imshow(lres_features[0,:,:,50])
gx.imshow(lres_features[0,:,:,60])
hx.imshow(lres_features[0,:,:,70])
ix.imshow(lres_features[0,:,:,80])

# cex = ex.imshow(np.squeeze(den4[0,:,:,0]))
# fig.colorbar(cex)
# fx.imshow(np.squeeze(den5[0,:,:,0]))

## load the model
# load YAML and create model
from keras.models import model_from_yaml
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

k =0
from helper_functions import *
den = model.predict(X_val[k:k+3,:])/100
image_arr = np.squeeze(X_val[:3,:])
den_arr = np.squeeze(Y_val[:3,:])
est_arr = np.squeeze(den)
save_val_results('results',(8,8),image_arr,den_arr, est_arr, 0)



