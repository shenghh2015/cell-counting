## this is the 0.1 version of the deep learning framework: cell counting
## this should be the entries for the neural network training
## by shenghua 04-27-2018

from keras.optimizers import SGD
import tensorflow as tf
import keras.backend as K

import os
import helper_functions as hf
import time
from models import *
from data_load import *
from utils.loss_fns import *

def model_train(model, val_version = 21, cross_val =0, nb_epoch_per_record =2, nb_epochs =1000, batch_size = 120, input_shape = (128,128), normal_proc = True, lr_max = None):
	# here is where we really load the data
	X_train,X_val,Y_train,Y_val = load_train_data(val_version = val_version, cross_val =cross_val, normal_proc = normal_proc)
	Y_train = Y_train*100
	Y_val = Y_val*100
	# here is where we really train the model
	hf.train_model_multi_task(model, X_train,X_val,Y_train,Y_val, nb_epochs=nb_epochs, nb_epoch_per_record=nb_epoch_per_record, input_shape=input_shape, batch_size = batch_size, lr_max = max_lr)
#	hf.train_model(model, X_train,X_val,Y_train,Y_val, nb_epochs=nb_epochs, nb_epoch_per_record=nb_epoch_per_record, input_shape=input_shape, batch_size = batch_size, lr_max = max_lr)

def get_session(gpu_fraction=1.0):
#     '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 	config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
# 	session = tf.Session(config=config)
# 	K.set_session(session)
	
	fraction = 1.0
	K.set_session(get_session(gpu_fraction=fraction))   ## use part of the GPU to run more tasks

	## model folder is where we store our trained model
# 	model_folder = os.path.expanduser('~/dl-cells/dlct-framework/models')

	model_folder = os.path.expanduser('../models')

	## parameters we set to train the model
	nb_epochs = 200000
	nb_epoch_per_record = 1
# 	input_shape = (96,96)
	input_shape = (128,128)
	batch_size = 100
# 	activation = 'relu'
	lr=0.0001
# 	lr=0.005
	dropout = None
	max_lr = None
	trial = 0
# 	init = 'normal'
# 	init = 'glorot_uniform'
# 	date = '5.22'			  # date when we train the model
	date = '10.1'			  # date when we train the model
#	net_arch ='FCRN_A'		  # the baseline method
#	net_arch ='URN'		      # the network that we create in models.py
# 	net_arch ='U_Net_FCRN_A'
# 	net_arch = 'MCNN'		 
# 	net_arch = 'MCNN_U'
# 	net_arch = 'imp_MCNN_U2'
# 	net_arch = 'MCNN_U_x4'
# 	net_arch = 'buildModel_FCRN_A'
#	net_arch = 'buildModel_FCRN_A_v2'
# 	net_arch = 'buildModel_U_net'
#	net_arch = 'buildMultiModel_FCRN_A_residual'
	net_arch = 'buildMultiModel_U_net'
# 	net_arch = 'buildMultiModel_U_net_res'
# 	net_arch = 'buildModel_InCep'
# 	net_arch = 'buildModel_InCep_v2'
# 	net_arch = 'buildMultiModel_FCRN_A'
# 	net_arch = 'buildMultiModel_InCep'
# 	net_arch = 'buildMultiModel_InCep_v2'
# 	net_arch = 'buildMultiModel_InCep_v3'
# 	net_arch = 'buildModel_MCNN_U'
	fcns = globals()[net_arch]
	val_version = 24	# benchmark 0: synthetic data
#	val_version = 25	# benchmark 1: BM data
#	val_version = 26	# benchmark 2: H&E data
#	val_version = 27	# 
	loss_fn = 'mse'
# 	loss_fn = 'mae'
# 	loss_fn = 'mean_squared_error'
# 	loss_fn = 'mse_ct_err'
	cross_val = 0
	normal_proc = True
# 	ratio = 0.1
# 	ratio = 0.0005
# 	loss_fcn = globals()[loss_fn]
	K.clear_session()
	model = fcns(input_shape, lr = lr, dropout= dropout, loss_fcn = loss_fn)
# 	model = fcns(input_shape)
# 	sgd = SGD(lr=lr, decay=1e-4, momentum=0.9, nesterov=True) # define the optimizer
# 	model.compile(loss='mean_squared_error', optimizer=sgd)   # compile the model with loss and optimizer
# 	model.compile(loss=loss_fcn, optimizer=sgd)   # compile the model with loss and optimizer
	name_str = 'date-{}-{}-{}-v-{}-cross-{}-batch-{}-drop-{}-lr-{}-nb_epochs-{}-norm-{}-t-{}'.format(date,
			net_arch,loss_fn, val_version, cross_val, batch_size,dropout,lr, nb_epoch_per_record, normal_proc, trial)  # path to store the trained model
	model_name = os.path.join(model_folder,name_str) 	# the complete path
	model.name = model_name
	print(model_name)
	time1 = time.time()
	model_train(model, val_version, cross_val, nb_epoch_per_record, nb_epochs, batch_size, input_shape, normal_proc, lr_max = max_lr)
	time2 = time.time()
	print('Training time:{}'.format(time2-time1))
# 	for cross_val in cross_vals:
# 		K.clear_session()
# 		model = fcns()
# 		sgd = SGD(lr=lr, decay=1e-4, momentum=0.9, nesterov=True) # define the optimizer
# 		model.compile(loss='mean_squared_error', optimizer=sgd)   # compile the model with loss and optimizer
# 		name_str = 'date-{}-{}-v-{}-cross-{}-batch-{}-lr-{}'.format(date,net_arch,val_version,cross_val, batch_size,lr)  # path to store the trained model
# 		model_name = os.path.join(model_folder,name_str) 	# the complete path
# 		model.name = model_name
# 		print(model_name)
# 		model_train(model, val_version, cross_val, nb_epoch_per_record, nb_epochs, batch_size, input_shape)