## this is the 0.1 version of the deep learning framework: cell counting
## this should be the entries for the neural network training
## by shenghua 05-26-2018

import tensorflow as tf
import keras.backend as K

import os
import helper_functions as hf
from models import *
from models_comp import *
from data_load import *
from utils.loss_fns import *

def model_train(model, val_version = 21, cross_val =0, nb_epoch_per_record =2, nb_epochs =1000, batch_size = 120, input_shape = (128,128), normal_proc = True):
	# here is where we really load the data
	X_train,X_val,Y_train,Y_val = load_train_data(val_version = val_version, cross_val =cross_val, normal_proc = normal_proc)
# 	Y_train = Y_train*100
# 	Y_val = Y_val*100
	# here is where we really train the model
#	hf.train_model_multi_task(model, X_train,X_val,Y_train,Y_val, nb_epochs=nb_epochs, nb_epoch_per_record=nb_epoch_per_record, input_shape=input_shape, batch_size = batch_size, lr_max = max_lr)
	hf.train_model(model, X_train,X_val,Y_train,Y_val, nb_epochs=nb_epochs, nb_epoch_per_record=nb_epoch_per_record, input_shape=input_shape, batch_size = batch_size, method = 'count_ception')

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "6"
	config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
	session = tf.Session(config=config)
	K.set_session(session)

	## model folder is where we store our trained model
	model_folder = os.path.expanduser('~/dl-cells/dlct-framework/models')

	## parameters we set to train the model
	nb_epochs = 400
	nb_epoch_per_record = 1
# 	input_shape = (96,96)
# 	input_shape = (128,128)
	input_shape = (128,128)
	batch_size = 20
# 	activation = 'relu'
	lr=0.00001
	dropout = None
	date = '7.9'
#	date = '6.27'			  # date when we train the model
	net_arch = 'buildModel_Count_ception'
	fcns = globals()[net_arch]
#	val_version = 44	# benchmark 0: synthetic data
#	val_version = 45	# benchmark 1: synthetic data
	val_version = 46	# benchmark 2: synthetic data
# 	val_version = 47	# 
	loss_fn = 'mae'
# 	loss_fn = 'mse'
	cross_val = 5
	normal_proc = True
	K.clear_session()
	model = fcns(input_shape, lr = lr, dropout= dropout, loss_fcn = loss_fn)
	name_str = 'date-{}-{}-{}-v-{}-cross-{}-batch-{}-drop-{}-lr-{}-nb_epochs-{}-norm-{}'.format(date,
			net_arch,loss_fn,val_version,cross_val,batch_size,dropout,lr, nb_epoch_per_record, normal_proc)  # path to store the trained model
	model_name = os.path.join(model_folder,name_str) 	# the complete path
	model.name = model_name
	print(model_name)
	model_train(model, val_version, cross_val, nb_epoch_per_record, nb_epochs, batch_size, input_shape, normal_proc)
