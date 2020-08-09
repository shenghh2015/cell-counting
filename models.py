from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, merge, Input, Cropping2D, concatenate, Add
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D,GlobalAveragePooling2D, Conv2D,Lambda, UpSampling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU, ELU  
from keras.regularizers import l2
from keras.optimizers import SGD
import keras.backend as K
import keras

from utils.loss_fns import *

def URN(input_shape = (128,128)):
	nb_filters=32
	kernel_size=(3,3)
	pool_size=(2,2)
	concat_axis = 3

	inputs = Input(shape=input_shape+(3,))
	conv1 = Conv2D(nb_filters, kernel_size, padding='same')(inputs)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

	conv2 = Conv2D(nb_filters, kernel_size, padding='same')(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

	conv3 = Conv2D(nb_filters*2, kernel_size, padding='same')(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
	pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

	conv4 = Conv2D(nb_filters*4, kernel_size, padding='same')(pool3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)
	pool4 = MaxPooling2D(pool_size=pool_size)(conv4)

	conv5 = Conv2D(nb_filters*8, kernel_size, padding='same')(pool4)
	conv5 = BatchNormalization()(conv5)
	conv5 = Activation('relu')(conv5)

# 	up_conv5 = UpSampling2D(size=pool_size)(conv5)
	de_conv5 = Conv2DTranspose(nb_filters*4, kernel_size, strides=(2, 2), padding='same')(conv5)
	de_conv5 = concatenate([de_conv5, conv4], axis = concat_axis)
	conv6 = Conv2D(nb_filters*4, kernel_size, strides=(1, 1), padding='same')(de_conv5)
	conv6 = BatchNormalization()(conv6)
	conv6 = Activation('relu')(conv6)

	de_conv6 = Conv2DTranspose(nb_filters*2, kernel_size, strides=(2, 2), padding='same')(conv6)
	de_conv6 = concatenate([de_conv6, conv3], axis = concat_axis)
	conv7 = Conv2D(nb_filters*2, kernel_size, strides=(1, 1), padding='same')(de_conv6)
	conv7 = BatchNormalization()(conv7)
	conv7 = Activation('relu')(conv7)

	de_conv7 = Conv2DTranspose(nb_filters, kernel_size, strides=(2, 2), padding='same')(conv7)
	de_conv7 = concatenate([de_conv7, conv2], axis = concat_axis)
	conv8 = Conv2D(nb_filters, kernel_size, strides=(1, 1), padding='same')(de_conv7)
	conv8 = BatchNormalization()(conv8)
	conv8 = Activation('relu')(conv8)

	de_conv8 = Conv2DTranspose(nb_filters, kernel_size, strides=(2, 2), padding='same')(conv8)
	de_conv8 = concatenate([de_conv8, conv1], axis = concat_axis)
	conv9 = Conv2D(nb_filters, kernel_size, strides=(1, 1), padding='same')(de_conv8)
	conv9 = BatchNormalization()(conv9)
	conv9 = Activation('relu')(conv9)

	fcn_output = Conv2D(1, kernel_size, padding='same')(conv9)
	fcn_output = BatchNormalization()(fcn_output)
	fcn_output = Activation('relu')(fcn_output)

	model = Model(inputs, fcn_output)
    
	return model

# the defect for this implementation: checkerboard artifact
def FCRN_A(input_shape = (96,96)):
	nb_filters=32
	kernel_size=(3,3)
	pool_size=(2,2)
	concat_axis = 3

	inputs = Input(shape=input_shape+(3,))
	conv1 = Conv2D(nb_filters, kernel_size, padding='same')(inputs)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

	conv2 = Conv2D(nb_filters*2, kernel_size, padding='same')(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

	conv3 = Conv2D(nb_filters*4, kernel_size, padding='same')(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
	pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

	conv4 = Conv2D(nb_filters*16, kernel_size, padding='same')(pool3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)

	de_conv5 = Conv2DTranspose(nb_filters*4, kernel_size, strides=(2, 2), padding='same')(conv4)
	de_conv5 = BatchNormalization()(de_conv5)
	de_conv5 = Activation('relu')(de_conv5)

	de_conv6 = Conv2DTranspose(nb_filters*2, kernel_size, strides=(2, 2), padding='same')(de_conv5)
	de_conv6 = BatchNormalization()(de_conv6)
	de_conv6 = Activation('relu')(de_conv6)

	de_conv7 = Conv2DTranspose(nb_filters, kernel_size, strides=(2, 2), padding='same')(de_conv6)
	de_conv7 = BatchNormalization()(de_conv7)
	de_conv7 = Activation('relu')(de_conv7)

	fcn_output = Conv2D(1, kernel_size, padding='same')(de_conv7)
	fcn_output = BatchNormalization()(fcn_output)
	fcn_output = Activation('relu')(fcn_output)

	model = Model(inputs, fcn_output)

	return model

# this implementation reduce the checkerboard artifact
weight_decay = 1e-5
K.set_image_dim_ordering('tf')

def _conv_bn_relu(nb_filter, kernel_size):
	def f(input):
		conv_a = Conv2D(nb_filter, kernel_size, padding='same', 
			kernel_initializer='orthogonal', kernel_regularizer = l2(weight_decay),
			bias_regularizer=l2(weight_decay))(input)
		norm_a = BatchNormalization()(conv_a)
		act_a = Activation(activation = 'relu')(norm_a)
		return act_a
	return f

def FCRN_A_base(input, dropout = None):
	nb_filter = 32
	kernel_size = (3,3)
# 	input = Input(shape=input_shape+(3,))
	block1 = _conv_bn_relu(nb_filter, kernel_size)(input)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	# =========================================================================
	block2 = _conv_bn_relu(nb_filter*2, kernel_size)(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	# =========================================================================
	block3 = _conv_bn_relu(nb_filter*4, kernel_size)(pool2)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	# =========================================================================
	block4 = _conv_bn_relu(nb_filter*16, kernel_size)(pool3)
	if dropout:
		block4 = Dropout(dropout)(block4)
	# =========================================================================
	up5 = UpSampling2D(size=(2, 2))(block4)
	block5 = _conv_bn_relu(nb_filter*4, kernel_size)(up5)
	# =========================================================================
	up6 = UpSampling2D(size=(2, 2))(block5)
	block6 = _conv_bn_relu(nb_filter*2, kernel_size)(up6)
	# =========================================================================
	up7 = UpSampling2D(size=(2, 2))(block6)
	block7 = _conv_bn_relu(nb_filter, kernel_size)(up7)
	return block7

def _conv_bn_relux2(nb_filter, kernel_size):
	def f(input):
		conv_a = Conv2D(nb_filter, kernel_size, padding='same', 
			kernel_initializer='orthogonal', kernel_regularizer = l2(weight_decay),
			bias_regularizer=l2(weight_decay))(input)
		norm_a = BatchNormalization()(conv_a)
		act_a = Activation(activation = 'relu')(norm_a)
		conv_b = Conv2D(nb_filter, kernel_size, padding='same', 
			kernel_initializer='orthogonal', kernel_regularizer = l2(weight_decay),
			bias_regularizer=l2(weight_decay))(act_a)
		norm_b = BatchNormalization()(conv_b)
		act_b = Activation(activation = 'relu')(norm_b)
		return act_b
	return f

def _res_conv_bn_relu(nb_filter, kernel_size):
	def f(input):
		conv_ = _conv_bn_relux2(nb_filter, kernel_size)(input)
# 		conv_a = Conv2D(nb_filter, kernel_size, padding='same', 
# 			kernel_initializer='orthogonal', kernel_regularizer = l2(weight_decay),
# 			bias_regularizer=l2(weight_decay))(input)
# 		norm_a = BatchNormalization()(conv_a)
# 		act_a = Activation(activation = 'relu')(norm_a)
# 		conv_b = Conv2D(nb_filter, kernel_size, padding='same', 
# 			kernel_initializer='orthogonal', kernel_regularizer = l2(weight_decay),
# 			bias_regularizer=l2(weight_decay))(act_a)
# 		norm_b = BatchNormalization()(conv_b)
# 		act_b = Activation(activation = 'relu')(norm_b)
		add_ = keras.layers.Add()([input, conv_])
		return add_
	return f

def FCRN_A_base_v2(input, dropout = None):
	nb_filter = 32
	kernel_size = (3,3)
# 	input = Input(shape=input_shape+(3,))
	block1 = _conv_bn_relux2(nb_filter, kernel_size)(input)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	# =========================================================================
	block2 = _conv_bn_relux2(nb_filter*2, kernel_size)(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	# =========================================================================
	block3 = _conv_bn_relux2(nb_filter*4, kernel_size)(pool2)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	# =========================================================================
	block4 = _conv_bn_relux2(nb_filter*16, kernel_size)(pool3)
	if dropout:
		block4 = Dropout(dropout)(block4)
	# =========================================================================
	up5 = UpSampling2D(size=(2, 2))(block4)
	block5 = _conv_bn_relux2(nb_filter*4, kernel_size)(up5)
	# =========================================================================
	up6 = UpSampling2D(size=(2, 2))(block5)
	block6 = _conv_bn_relux2(nb_filter*2, kernel_size)(up6)
	# =========================================================================
	up7 = UpSampling2D(size=(2, 2))(block6)
	block7 = _conv_bn_relux2(nb_filter, kernel_size)(up7)
	return block7

def U_net_base(input, dropout = None):
	nb_filter = 32
	kernel_size = (3,3)
# 	input = Input(shape=input_shape+(3,))
	block1 = _conv_bn_relu(nb_filter, kernel_size)(input)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	# =========================================================================
	block2 = _conv_bn_relu(nb_filter*2, kernel_size)(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	# =========================================================================
	block3 = _conv_bn_relu(nb_filter*4, kernel_size)(pool2)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	# =========================================================================
	block4 = _conv_bn_relu(nb_filter*16, kernel_size)(pool3)
	if dropout:
		block4 = Dropout(dropout)(block4)
	# =========================================================================
	up5 = concatenate([UpSampling2D(size=(2, 2))(block4), block3], axis = -1)
	block5 = _conv_bn_relu(nb_filter*4, kernel_size)(up5)
	# =========================================================================
	up6 = concatenate([UpSampling2D(size=(2, 2))(block5), block2], axis = -1)
	block6 = _conv_bn_relu(nb_filter*2, kernel_size)(up6)
	# =========================================================================
	up7 = concatenate([UpSampling2D(size=(2, 2))(block6), block1], axis = -1)
	block7 = _conv_bn_relu(nb_filter, kernel_size)(up7)
	return block7

def FCRN_A_Multi_base(input, dropout = None):
	nb_filter = 32
	kernel_size = (3,3)
	block1 = _conv_bn_relu(nb_filter, kernel_size)(input)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	# =========================================================================
	block2 = _conv_bn_relu(nb_filter*2, kernel_size)(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	# =========================================================================
	block3 = _conv_bn_relu(nb_filter*4, kernel_size)(pool2)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	# =========================================================================
	block4 = _conv_bn_relu(nb_filter*16, kernel_size)(pool3)
	if dropout:
		block4 = Dropout(dropout)(block4)
	# =========================================================================
	up5 = UpSampling2D(size=(2, 2))(block4)
	block5 = _conv_bn_relu(nb_filter*4, kernel_size)(up5)
	# =========================================================================
	up6 = UpSampling2D(size=(2, 2))(block5)
	block6 = _conv_bn_relu(nb_filter*2, kernel_size)(up6)
	# =========================================================================
	up7 = UpSampling2D(size=(2, 2))(block6)
	block7 = _conv_bn_relu(nb_filter, kernel_size)(up7)
	# =========================================================================
	out_block1 = _conv_bn_relu(nb_filter, kernel_size)(block4)
	# =========================================================================
	out_block2 = _conv_bn_relu(nb_filter, kernel_size)(block5)
	# =========================================================================
	out_block3 = _conv_bn_relu(nb_filter, kernel_size)(block6)
	return out_block1, out_block2, out_block3, block7

def U_net_Multi_base(input, dropout = None):
	nb_filter = 32
	kernel_size = (3,3)
	block1 = _conv_bn_relu(nb_filter, kernel_size)(input)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	# =========================================================================
	block2 = _conv_bn_relu(nb_filter*2, kernel_size)(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	# =========================================================================
	block3 = _conv_bn_relu(nb_filter*4, kernel_size)(pool2)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	# =========================================================================
	block4 = _conv_bn_relu(nb_filter*16, kernel_size)(pool3)
	if dropout:
		block4 = Dropout(dropout)(block4)
	# =========================================================================
	up5 = concatenate([UpSampling2D(size=(2, 2))(block4), block3], axis = -1)
	block5 = _conv_bn_relu(nb_filter*4, kernel_size)(up5)
	# =========================================================================
	up6 = concatenate([UpSampling2D(size=(2, 2))(block5), block2], axis = -1)
	block6 = _conv_bn_relu(nb_filter*2, kernel_size)(up6)
	# =========================================================================
	up7 = concatenate([UpSampling2D(size=(2, 2))(block6), block1], axis = -1)
	block7 = _conv_bn_relu(nb_filter, kernel_size)(up7)
	# =========================================================================
	out_block1 = _conv_bn_relu(nb_filter, kernel_size)(block4)
	# =========================================================================
	out_block2 = _conv_bn_relu(nb_filter, kernel_size)(block5)
	# =========================================================================
	out_block3 = _conv_bn_relu(nb_filter, kernel_size)(block6)
	return out_block1, out_block2, out_block3, block7

def buildMultiModel_U_net(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse', activation = 'linear', loss_weights=[1./64,1/16, 1./4, 1]):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	out_block1, out_block2, out_block3, block7 = U_net_Multi_base(input_, dropout)
	# =========================================================================
	density_pred_1 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = activation, name = 'red8')(out_block1)	
	density_pred_2 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = activation, name = 'red4')(out_block2)
	density_pred_3 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = activation, name = 'red2')(out_block3)
	density_pred_4 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = activation, name = 'original')(block7)
	# =========================================================================
	model = Model (input = input_, output = [density_pred_1, density_pred_2, density_pred_3, density_pred_4])
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse', loss_weights=loss_weights)
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae', loss_weights=loss_weights)
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn, loss_weights=loss_weights)
	return model

def FCRN_A_Multi_residual_base_v2(input, dropout = None):
	nb_filter = 32
	kernel_size = (3,3)
	block1 = _conv_bn_relu(nb_filter, kernel_size)(input)
	block1 = _res_conv_bn_relu(nb_filter, kernel_size)(block1)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	# =========================================================================
	block2 = _conv_bn_relu(nb_filter*2, kernel_size)(pool1)
	block2 = _res_conv_bn_relu(nb_filter*2, kernel_size)(block2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	# =========================================================================
	block3 = _conv_bn_relu(nb_filter*4, kernel_size)(pool2)
	block3 = _res_conv_bn_relu(nb_filter*4, kernel_size)(block3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	# =========================================================================
	block4 = _conv_bn_relu(nb_filter*16, kernel_size)(pool3)
	if dropout:
		block4 = Dropout(dropout)(block4)
	# =========================================================================
	up5 = UpSampling2D(size=(2, 2))(block4)
	block5 = _conv_bn_relu(nb_filter*4, kernel_size)(up5)
	block5 = _res_conv_bn_relu(nb_filter*4, kernel_size)(block5)
	# =========================================================================
	up6 = UpSampling2D(size=(2, 2))(block5)
	block6 = _conv_bn_relu(nb_filter*2, kernel_size)(up6)
	block6 = _res_conv_bn_relu(nb_filter*2, kernel_size)(block6)
	# =========================================================================
	up7 = UpSampling2D(size=(2, 2))(block6)
	block7 = _conv_bn_relu(nb_filter, kernel_size)(up7)
	block7 = _res_conv_bn_relu(nb_filter, kernel_size)(block7)
	# =========================================================================
	out_block1 = _conv_bn_relu(nb_filter, kernel_size)(block4)
	# =========================================================================
	out_block2 = _conv_bn_relu(nb_filter, kernel_size)(block5)
	# =========================================================================
	out_block3 = _conv_bn_relu(nb_filter, kernel_size)(block6)
	return out_block1, out_block2, out_block3, block7

def buildMultiModel_FCRN_A_residual(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse', loss_weights=[1./64,1/16, 1./4, 1]):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	out_block1, out_block2, out_block3, block7 = FCRN_A_Multi_residual_base_v2(input_, dropout)
	# =========================================================================
	density_pred_1 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red8')(out_block1)	
	density_pred_2 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red4')(out_block2)
	density_pred_3 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red2')(out_block3)
	density_pred_4 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'original')(block7)
	# =========================================================================
	model = Model (input = input_, output = [density_pred_1, density_pred_2, density_pred_3, density_pred_4])
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse', loss_weights=loss_weights)
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae', loss_weights=loss_weights)
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn, loss_weights=loss_weights)
	return model

def U_net_Multi_res_base(input, dropout = None):
	nb_filter = 32
	kernel_size = (3,3)
	block1 = _conv_bn_relu(nb_filter, kernel_size)(input)
	block1 = _res_conv_bn_relu(nb_filter, kernel_size)(block1)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	# =========================================================================
	block2 = _conv_bn_relu(nb_filter*2, kernel_size)(pool1)
	block2 = _res_conv_bn_relu(nb_filter*2, kernel_size)(block2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	# =========================================================================
	block3 = _conv_bn_relu(nb_filter*4, kernel_size)(pool2)
	block3 = _res_conv_bn_relu(nb_filter*4, kernel_size)(block3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	# =========================================================================
	block4 = _conv_bn_relu(nb_filter*16, kernel_size)(pool3)
	if dropout:
		block4 = Dropout(dropout)(block4)
	# =========================================================================
	up5 = concatenate([UpSampling2D(size=(2, 2))(block4), block3], axis = -1)
	block5 = _conv_bn_relu(nb_filter*4, kernel_size)(up5)
	block5 = _res_conv_bn_relu(nb_filter*4, kernel_size)(block5)
	# =========================================================================
	up6 = concatenate([UpSampling2D(size=(2, 2))(block5), block2], axis = -1)
	block6 = _conv_bn_relu(nb_filter*2, kernel_size)(up6)
	block6 = _res_conv_bn_relu(nb_filter*2, kernel_size)(block6)
	# =========================================================================
	up7 = concatenate([UpSampling2D(size=(2, 2))(block6), block1], axis = -1)
	block7 = _conv_bn_relu(nb_filter, kernel_size)(up7)
	block7 = _res_conv_bn_relu(nb_filter, kernel_size)(block7)
	# =========================================================================
	out_block1 = _conv_bn_relu(nb_filter, kernel_size)(block4)
	# =========================================================================
	out_block2 = _conv_bn_relu(nb_filter, kernel_size)(block5)
	# =========================================================================
	out_block3 = _conv_bn_relu(nb_filter, kernel_size)(block6)
	return out_block1, out_block2, out_block3, block7

def buildMultiModel_U_net_res(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse', loss_weights=[1./64,1/16, 1./4, 1]):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	out_block1, out_block2, out_block3, block7 = U_net_Multi_res_base(input_, dropout)
	# =========================================================================
	density_pred_1 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red8')(out_block1)	
	density_pred_2 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red4')(out_block2)
	density_pred_3 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red2')(out_block3)
	density_pred_4 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'original')(block7)
	# =========================================================================
	model = Model (input = input_, output = [density_pred_1, density_pred_2, density_pred_3, density_pred_4])
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse', loss_weights=loss_weights)
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae', loss_weights=loss_weights)
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn, loss_weights=loss_weights)
	return model

def buildMultiModel_FCRN_A(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse', loss_weights=[1./64,1/16, 1./4, 1]):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	out_block1, out_block2, out_block3, block7 = FCRN_A_Multi_base(input_, dropout)
	# =========================================================================
	density_pred_1 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red8')(out_block1)	
	density_pred_2 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red4')(out_block2)
	density_pred_3 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red2')(out_block3)
	density_pred_4 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'original')(block7)
	# =========================================================================
	model = Model (input = input_, output = [density_pred_1, density_pred_2, density_pred_3, density_pred_4])
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse', loss_weights=loss_weights)
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae', loss_weights=loss_weights)
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn, loss_weights=loss_weights)
	return model

def buildModel_FCRN_A(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse'):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	act_ = FCRN_A_base (input_, dropout)
	# =========================================================================
	density_pred =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear')(act_)
	# =========================================================================
	model = Model (input = input_, output = density_pred)
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse')
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae')
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn)
	return model

def buildModel_U_net(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse', activation = 'linear'):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	act_ = U_net_base (input_, dropout)
	# =========================================================================
	density_pred =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = activation)(act_)
	# =========================================================================
	model = Model (input = input_, output = density_pred)
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse')
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae')
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn)
	return model

def buildModel_FCRN_A_v2(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse'):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	act_ = FCRN_A_base_v2 (input_, dropout)
	# =========================================================================
	density_pred =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear')(act_)
	# =========================================================================
	model = Model (input = input_, output = density_pred)
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse')
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae')
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn)
	return model

def _branch_conv_bn_relu(input, kernel_size = (3,3)):
	nb_filter = 32
	block1 = _conv_bn_relu(nb_filter, kernel_size)(input)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	# =========================================================================
	block2 = _conv_bn_relu(nb_filter*2, kernel_size)(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	# =========================================================================
	block3 = _conv_bn_relu(nb_filter*4, kernel_size)(pool2)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	return block1, pool1, block2, pool2, block3, pool3

def MCNN_base(input):
	nb_filter = 32
	kernel_size = (3,3)
	concat_axis = -1
	# branch 1
	b1_block1, b1_pool1, b1_block2, b1_pool2, b1_block3, b1_pool3 = _branch_conv_bn_relu(input,(3,3))
	# branch 2
	b2_block1, b2_pool1, b2_block2, b2_pool2, b2_block3, b2_pool3 = _branch_conv_bn_relu(input,(5,5))
	# branch 3
	b3_block1, b3_pool1, b3_block2, b3_pool2, b3_block3, b3_pool3 = _branch_conv_bn_relu(input,(7,7))
	# =========================================================================	
	# merge feature 3
	merged_block3 = concatenate([b1_block3, b2_block3, b3_block3], axis = concat_axis)
	merged_block3 = _conv_bn_relu(nb_filter*4,(1,1))(merged_block3)
	# merge feature 2
	merged_block2 = concatenate([b1_block2, b2_block2, b3_block2], axis = concat_axis)
	merged_block2 = _conv_bn_relu(nb_filter*4,(1,1))(merged_block2)
	# merge feature 3
	merged_block1 = concatenate([b1_block1, b2_block1, b3_block1], axis = concat_axis)
	merged_block1 = _conv_bn_relu(nb_filter*4,(1,1))(merged_block1)
	# =========================================================================	
	merged_pool3 = MaxPooling2D(pool_size=(2, 2))(merged_block3)
	# =========================================================================
	block4 = _conv_bn_relu(nb_filter*16, (3,3))(merged_pool3)
	up4 = concatenate([UpSampling2D(size=(2, 2))(block4), merged_block3], axis = concat_axis)
	# =========================================================================
	block5 = _conv_bn_relu(nb_filter*4,(3,3))(up4)
	up5 = concatenate([UpSampling2D(size=(2, 2))(block5), merged_block2], axis= concat_axis)
	# =========================================================================
	block6 = _conv_bn_relu(nb_filter*2,(3,3))(up5)
	up6 = concatenate([UpSampling2D(size=(2, 2))(block6), merged_block1], axis= concat_axis)
	# =========================================================================
	block7 = _conv_bn_relu(nb_filter, (3,3))(up6)
	return block7

def buildModel_MCNN_U(input_shape, lr = 1e-2, loss_fcn = 'mse'):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	act_ = MCNN_base (input_)
	# =========================================================================
	density_pred =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear')(act_)
	# =========================================================================
	model = Model (input = input_, output = density_pred)
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse')
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn)
	return model

def _incep_conv_bn_relu(nb_filter = 32, kernel_list = [(1,1),(3,3),(5,5)]):
	def f(input):
		_block1 = _conv_bn_relu(nb_filter, kernel_list[0])(input)
		_block2 = _conv_bn_relu(nb_filter, kernel_list[1])(input)
		_block3 = _conv_bn_relu(nb_filter, kernel_list[2])(input)
		_merge = concatenate([_block1, _block2, _block3], axis = -1)
		_output = _conv_bn_relu(nb_filter,(1,1))(_merge)
		return _output
	return f

# def _conv_bn_relu(nb_filter, kernel_size):
# 	def f(input):
# 		conv_a = Conv2D(nb_filter, kernel_size, padding='same', 
# 			kernel_initializer='orthogonal', kernel_regularizer = l2(weight_decay),
# 			bias_regularizer=l2(weight_decay))(input)
# 		norm_a = BatchNormalization()(conv_a)
# 		act_a = Activation(activation = 'relu')(norm_a)
# 		return act_a
# 	return f

def InCep_base(input, dropout = 0.2):
	nb_filter = 32
	kernel_size = (3,3)
	kernel_list = [(1,1),(3,3),(5,5)]
	concat_axis = -1
	# branch 1
	nb_filter = 32

	block1 = _incep_conv_bn_relu(nb_filter, kernel_list)(input)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	# =========================================================================
	block2 = _incep_conv_bn_relu(nb_filter*2, kernel_list)(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	# =========================================================================
	block3 = _incep_conv_bn_relu(nb_filter*4, kernel_list)(pool2)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	# =========================================================================
	block4 = _conv_bn_relu(nb_filter*16, kernel_size)(pool3)
	if dropout:
		block4 = Dropout(dropout)(block4)
	# =========================================================================
	up5 = UpSampling2D(size=(2, 2))(block4)
	block5 = _conv_bn_relu(nb_filter*4, kernel_size)(up5)
	# =========================================================================
	up6 = UpSampling2D(size=(2, 2))(block5)
	block6 = _conv_bn_relu(nb_filter*2, kernel_size)(up6)
	# =========================================================================
	up7 = UpSampling2D(size=(2, 2))(block6)
	block7 = _conv_bn_relu(nb_filter, kernel_size)(up7)
	return block7

def InCep_Multi_base(input, dropout = None):
	nb_filter = 32
	kernel_size = (3,3)
	kernel_list = [(1,1),(3,3),(5,5)]
	concat_axis = -1
	# branch 1
	nb_filter = 32
	block1 = _incep_conv_bn_relu(nb_filter, kernel_list)(input)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	# =========================================================================
	block2 = _incep_conv_bn_relu(nb_filter*2, kernel_list)(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	# =========================================================================
	block3 = _incep_conv_bn_relu(nb_filter*4, kernel_list)(pool2)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	# =========================================================================
	block4 = _conv_bn_relu(nb_filter*16, kernel_size)(pool3)
	if dropout:
		block4 = Dropout(dropout)(block4)
	# =========================================================================
	up5 = UpSampling2D(size=(2, 2))(block4)
	block5 = _conv_bn_relu(nb_filter*4, kernel_size)(up5)
	# =========================================================================
	up6 = UpSampling2D(size=(2, 2))(block5)
	block6 = _conv_bn_relu(nb_filter*2, kernel_size)(up6)
	# =========================================================================
	up7 = UpSampling2D(size=(2, 2))(block6)
	block7 = _conv_bn_relu(nb_filter, kernel_size)(up7)
	# =========================================================================
	out_block1 = _conv_bn_relu(nb_filter, kernel_size)(block4)
	# =========================================================================
	out_block2 = _conv_bn_relu(nb_filter, kernel_size)(block5)
	# =========================================================================
	out_block3 = _conv_bn_relu(nb_filter, kernel_size)(block6)
	return out_block1, out_block2, out_block3, block7

def buildMultiModel_InCep(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse', loss_weights=[1./64,1/16, 1./4, 1]):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	out_block1, out_block2, out_block3, block7 = InCep_Multi_base(input_, dropout)
	# =========================================================================
	density_pred_1 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red8')(out_block1)	
	density_pred_2 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red4')(out_block2)
	density_pred_3 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red2')(out_block3)
	density_pred_4 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'original')(block7)
	# =========================================================================
	model = Model (input = input_, output = [density_pred_1, density_pred_2, density_pred_3, density_pred_4])
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse', loss_weights=loss_weights)
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae', loss_weights=loss_weights)
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn, loss_weights=loss_weights)
	return model

def InCep_Multi_base_v2(input, dropout = None):
	nb_filter = 32
	kernel_size = (3,3)
	kernel_list = [(1,1),(3,3),(5,5)]
	concat_axis = -1
	# branch 1
	nb_filter = 32
	block1 = _incep_conv_bn_relu(nb_filter, kernel_list)(input)
	block1 = _conv_bn_relu(nb_filter,kernel_size)(block1)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	# =========================================================================
	block2 = _incep_conv_bn_relu(nb_filter*2, kernel_list)(pool1)
	block2 = _conv_bn_relu(nb_filter*2,kernel_size)(block2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	# =========================================================================
	block3 = _incep_conv_bn_relu(nb_filter*4, kernel_list)(pool2)
	block3 = _conv_bn_relu(nb_filter*4,kernel_size)(block3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	# =========================================================================
	block4 = _conv_bn_relu(nb_filter*16, kernel_size)(pool3)
	if dropout:
		block4 = Dropout(dropout)(block4)
	# =========================================================================
	up5 = UpSampling2D(size=(2, 2))(block4)
	block5 = _incep_conv_bn_relu(nb_filter*4, kernel_list)(up5)
	block5 = _conv_bn_relu(nb_filter*4, kernel_size)(block5)
	# =========================================================================
	up6 = UpSampling2D(size=(2, 2))(block5)
	block6 = _incep_conv_bn_relu(nb_filter*2, kernel_list)(up6)
	block6 = _conv_bn_relu(nb_filter*2, kernel_size)(block6)
	# =========================================================================
	up7 = UpSampling2D(size=(2, 2))(block6)
	block7 = _incep_conv_bn_relu(nb_filter, kernel_list)(up7)
	block7 = _conv_bn_relu(nb_filter, kernel_size)(block7)
	# =========================================================================
	out_block1 = _conv_bn_relu(nb_filter, kernel_size)(block4)
	# =========================================================================
	out_block2 = _conv_bn_relu(nb_filter, kernel_size)(block5)
	# =========================================================================
	out_block3 = _conv_bn_relu(nb_filter, kernel_size)(block6)
	return out_block1, out_block2, out_block3, block7

def buildMultiModel_InCep_v2(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse', loss_weights=[1./64,1/16, 1./4, 1]):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	out_block1, out_block2, out_block3, block7 = InCep_Multi_base_v2(input_, dropout)
	# =========================================================================
	density_pred_1 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red8')(out_block1)	
	density_pred_2 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red4')(out_block2)
	density_pred_3 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red2')(out_block3)
	density_pred_4 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'original')(block7)
	# =========================================================================
	model = Model (input = input_, output = [density_pred_1, density_pred_2, density_pred_3, density_pred_4])
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse', loss_weights=loss_weights)
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae', loss_weights=loss_weights)
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn, loss_weights=loss_weights)
	return model

def _incep_conv_bn_relu_v3(nb_filter = 32, kernel_list = [(1,1),(3,3),(5,5)]):
	def f(input):
		_block1 = _conv_bn_relu(nb_filter, kernel_list[0])(input)
		_block2 = _conv_bn_relu(nb_filter, kernel_list[1])(input)
# 		_block3 = _conv_bn_relu(nb_filter, kernel_list[2])(input)
# 		_merge = concatenate([_block1, _block2, _block3], axis = -1)
		_merge = concatenate([_block1, _block2], axis = -1)
		_output = _conv_bn_relu(nb_filter,(1,1))(_merge)
		return _output
	return f

def InCep_Multi_base_v3(input, dropout = None):
	nb_filter = 32
	kernel_size = (3,3)
	kernel_list = [(1,1),(3,3),(5,5)]
	concat_axis = -1
	# branch 1
	nb_filter = 32
	block1 = _incep_conv_bn_relu_v3(nb_filter, kernel_list)(input)
	block1 = _conv_bn_relu(nb_filter,kernel_size)(block1)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	# =========================================================================
	block2 = _incep_conv_bn_relu_v3(nb_filter*2, kernel_list)(pool1)
	block2 = _conv_bn_relu(nb_filter*2,kernel_size)(block2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	# =========================================================================
	block3 = _incep_conv_bn_relu_v3(nb_filter*4, kernel_list)(pool2)
	block3 = _conv_bn_relu(nb_filter*4,kernel_size)(block3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	# =========================================================================
	block4 = _conv_bn_relu(nb_filter*16, kernel_size)(pool3)
	if dropout:
		block4 = Dropout(dropout)(block4)
	# =========================================================================
	up5 = UpSampling2D(size=(2, 2))(block4)
	block5 = _incep_conv_bn_relu_v3(nb_filter*4, kernel_list)(up5)
	block5 = _conv_bn_relu(nb_filter*4, kernel_size)(block5)
	# =========================================================================
	up6 = UpSampling2D(size=(2, 2))(block5)
	block6 = _incep_conv_bn_relu_v3(nb_filter*2, kernel_list)(up6)
	block6 = _conv_bn_relu(nb_filter*2, kernel_size)(block6)
	# =========================================================================
	up7 = UpSampling2D(size=(2, 2))(block6)
	block7 = _incep_conv_bn_relu_v3(nb_filter, kernel_list)(up7)
	block7 = _conv_bn_relu(nb_filter, kernel_size)(block7)
	# =========================================================================
	out_block1 = _conv_bn_relu(nb_filter, kernel_size)(block4)
	# =========================================================================
	out_block2 = _conv_bn_relu(nb_filter, kernel_size)(block5)
	# =========================================================================
	out_block3 = _conv_bn_relu(nb_filter, kernel_size)(block6)
	return out_block1, out_block2, out_block3, block7

def buildMultiModel_InCep_v3(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse', loss_weights=[1./64,1/16, 1./4, 1]):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	out_block1, out_block2, out_block3, block7 = InCep_Multi_base_v3(input_, dropout)
	# =========================================================================
	density_pred_1 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red8')(out_block1)	
	density_pred_2 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red4')(out_block2)
	density_pred_3 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'red2')(out_block3)
	density_pred_4 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear', name = 'original')(block7)
	# =========================================================================
	model = Model (input = input_, output = [density_pred_1, density_pred_2, density_pred_3, density_pred_4])
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse', loss_weights=loss_weights)
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae', loss_weights=loss_weights)
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn, loss_weights=loss_weights)
	return model


def buildModel_InCep(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse'):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	act_ = InCep_base(input_, dropout)
	# =========================================================================
	density_pred =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear')(act_)
	# =========================================================================
	model = Model (input = input_, output = density_pred)
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse')
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae')
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn)
	return model

def _incep_block(nb_filter = 32, nb_reduce = [32,32], ratio = [0.25,0.5,0.125,0.125],kernel_list = [(1,1),(3,3),(5,5)]):
	def f(input):
		# 1x1 branch
		_block1 = _conv_bn_relu(int(nb_filter*ratio[0]), kernel_list[0])(input)
		# 3x3 branch
		_block2 = _conv_bn_relu(nb_reduce[0], (1,1))(input)
		_block2 = _conv_bn_relu(int(nb_filter*ratio[1]), kernel_list[1])(_block2)
		# 5x5 branch
		_block3 = _conv_bn_relu(nb_reduce[1], (1,1))(input)
		_block3 = _conv_bn_relu(int(nb_filter*ratio[2]), kernel_list[2])(_block3)
		# maxpool 3x3 branch
		_block4 = MaxPooling2D(pool_size=(3, 3),strides=(1,1), border_mode='same')(input)
		_block4 = _conv_bn_relu(int(nb_filter*ratio[3]),(1,1))(_block4)
		# merge
		_merge = concatenate([_block1, _block2, _block3, _block4], axis = -1)
# 		_output = _conv_bn_relu(nb_filter,(1,1))(_merge)
		return _merge
	return f

# inception base 2
def InCep_base_v2(input, dropout = None):
	nb_filter = 64
	kernel_size = (3,3)
	kernel_list = [(1,1),(3,3),(5,5)]
	concat_axis = -1
	nb_filter = 32
	block1 = _conv_bn_relu(nb_filter, kernel_size)(input)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	# =========================================================================
	block2 = _incep_block(nb_filter*2, nb_reduce = [32,8], kernel_list = kernel_list)(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	# =========================================================================
	block3 = _incep_block(nb_filter*4, nb_reduce = [64,16], kernel_list = kernel_list)(pool2)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	# =========================================================================
	block4 = _incep_block(nb_filter*8, nb_reduce = [128,32], kernel_list =kernel_list)(pool3)
	if dropout:
		block4 = Dropout(dropout)(block4)
	# =========================================================================
	up5 = UpSampling2D(size=(2, 2))(block4)
	block5 = _conv_bn_relu(nb_filter*4, kernel_size)(up5)
	# =========================================================================
	up6 = UpSampling2D(size=(2, 2))(block5)
	block6 = _conv_bn_relu(nb_filter*2, kernel_size)(up6)
	# =========================================================================
	up7 = UpSampling2D(size=(2, 2))(block6)
	block7 = _conv_bn_relu(nb_filter, kernel_size)(up7)
	return block7

# non-uniform
def buildModel_InCep_v2 (input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse'):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	act_ = InCep_base_v2(input_, dropout)
	# =========================================================================
	density_pred =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear')(act_)
	# =========================================================================
	model = Model (input = input_, output = density_pred)
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse')
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae')
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn)
	return model

# proposed method 0
def U_Net_FCRN_A(input_shape = (128,128)):
	nb_filters=32
	kernel_size=(3,3)
	pool_size=(2,2)
	concat_axis = 3
	weight_decay = 1e-5


	inputs = Input(shape=input_shape+(3,))
	conv1 = Conv2D(nb_filters, kernel_size, padding='same',
					kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

	conv2 = Conv2D(nb_filters*2, kernel_size, padding='same',
					kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

	conv3 = Conv2D(nb_filters*4, kernel_size, padding='same',
					kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
	pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

	conv4 = Conv2D(nb_filters*8, kernel_size, padding='same',
					kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)
	pool4 = MaxPooling2D(pool_size=pool_size)(conv4)

	conv_fc = Conv2D(nb_filters*16, kernel_size, padding='same',
					kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool4)
	conv_fc = BatchNormalization()(conv_fc)
	conv_fc = Activation('relu')(conv_fc)

	de_conv5 = Conv2DTranspose(nb_filters*8, kernel_size, strides=(2, 2), padding='same',
					kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(conv_fc)
	de_conv5 = concatenate([de_conv5, conv4], axis = concat_axis)
	de_conv5 = Conv2D(nb_filters*8, kernel_size, strides=(1, 1), padding='same',
					kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv5)
	de_conv5 = BatchNormalization()(de_conv5)
	de_conv5 = Activation('relu')(de_conv5)

	de_conv6 = Conv2DTranspose(nb_filters*4, kernel_size, strides=(2, 2), padding='same',
					kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv5)
	de_conv6 = concatenate([de_conv6, conv3], axis = concat_axis)
	de_conv6 = Conv2D(nb_filters*4, kernel_size, strides=(1, 1), padding='same',
					kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv6)
	de_conv6 = BatchNormalization()(de_conv6)
	de_conv6 = Activation('relu')(de_conv6)

	de_conv7 = Conv2DTranspose(nb_filters*2, kernel_size, strides=(2, 2), padding='same',
					kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv6)
	de_conv7 = concatenate([de_conv7, conv2], axis = concat_axis)
	de_conv7 = Conv2D(nb_filters*2, kernel_size, strides=(1, 1), padding='same',
					kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv7)
	de_conv7 = BatchNormalization()(de_conv7)
	de_conv7 = Activation('relu')(de_conv7)

	de_conv8 = Conv2DTranspose(nb_filters, kernel_size, strides=(2, 2), padding='same',
					kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv7)
	de_conv8 = concatenate([de_conv8, conv1], axis = concat_axis)
	de_conv8 = Conv2D(nb_filters, kernel_size, strides=(1, 1), padding='same',
					kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv8)
	de_conv8 = BatchNormalization()(de_conv8)
	de_conv8 = Activation('relu')(de_conv8)

	fcn_output = Conv2D(1, kernel_size, padding='same',
					kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv8)
	fcn_output = BatchNormalization()(fcn_output)
	fcn_output = Activation('relu')(fcn_output)

	model = Model(inputs, fcn_output)

	return model

# proposed method 1
def MCNN(input_shape = (96,96)):
	nb_filters=32
	kernel_size=(3,3)
	pool_size=(2,2)
	concat_axis = 3

	inputs = Input(shape=input_shape+(3,))
	
	# column1
	conv1 = Conv2D(nb_filters, kernel_size, padding='same')(inputs)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

	conv2 = Conv2D(nb_filters*2, kernel_size, padding='same')(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

	conv3 = Conv2D(nb_filters*4, kernel_size, padding='same')(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
	pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

# 	conv4 = Conv2D(nb_filters*16, kernel_size, padding='same')(pool3)
# 	conv4 = BatchNormalization()(conv4)
# 	conv4 = Activation('relu')(conv4)

	# column2
	kernel_size = (5,5)
	conv1_1 = Conv2D(nb_filters, kernel_size, padding='same')(inputs)
	conv1_1 = BatchNormalization()(conv1_1)
	conv1_1 = Activation('relu')(conv1_1)
	pool1_1 = MaxPooling2D(pool_size=pool_size)(conv1_1)

	conv2_1 = Conv2D(nb_filters*2, kernel_size, padding='same')(pool1_1)
	conv2_1 = BatchNormalization()(conv2_1)
	conv2_1 = Activation('relu')(conv2_1)
	pool2_1 = MaxPooling2D(pool_size=pool_size)(conv2_1)

	conv3_1 = Conv2D(nb_filters*4, kernel_size, padding='same')(pool2_1)
	conv3_1 = BatchNormalization()(conv3_1)
	conv3_1 = Activation('relu')(conv3_1)
	pool3_1 = MaxPooling2D(pool_size=pool_size)(conv3_1)

# 	conv4_1 = Conv2D(nb_filters*16, kernel_size, padding='same')(pool3_1)
# 	conv4_1 = BatchNormalization()(conv4_1)
# 	conv4_1 = Activation('relu')(conv4_1)

	# column3
	kernel_size = (7,7)
	conv1_2 = Conv2D(nb_filters, kernel_size, padding='same')(inputs)
	conv1_2 = BatchNormalization()(conv1_2)
	conv1_2 = Activation('relu')(conv1_2)
	pool1_2 = MaxPooling2D(pool_size=pool_size)(conv1_2)

	conv2_2 = Conv2D(nb_filters*2, kernel_size, padding='same')(pool1_2)
	conv2_2 = BatchNormalization()(conv2_2)
	conv2_2 = Activation('relu')(conv2_2)
	pool2_2 = MaxPooling2D(pool_size=pool_size)(conv2_2)

	conv3_2 = Conv2D(nb_filters*4, kernel_size, padding='same')(pool2_2)
	conv3_2 = BatchNormalization()(conv3_2)
	conv3_2 = Activation('relu')(conv3_2)
	pool3_2 = MaxPooling2D(pool_size=pool_size)(conv3_2)

# 	conv4_2 = Conv2D(nb_filters*16, kernel_size, padding='same')(pool3_2)
# 	conv4_2 = BatchNormalization()(conv4_2)
# 	conv4_2 = Activation('relu')(conv4_2)

	# merge the features
	merge_conv = concatenate([pool3, pool3_1, pool3_2], axis = concat_axis)
	
	kernel_size = (3,3)
	conv4 = Conv2D(nb_filters*16, kernel_size, padding='same')(merge_conv)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)

	de_conv5 = Conv2DTranspose(nb_filters*4, kernel_size, strides=(2, 2), padding='same')(merge_conv)
	de_conv5 = BatchNormalization()(de_conv5)
	de_conv5 = Activation('relu')(de_conv5)

	de_conv6 = Conv2DTranspose(nb_filters*2, kernel_size, strides=(2, 2), padding='same')(de_conv5)
	de_conv6 = BatchNormalization()(de_conv6)
	de_conv6 = Activation('relu')(de_conv6)

	de_conv7 = Conv2DTranspose(nb_filters, kernel_size, strides=(2, 2), padding='same')(de_conv6)
	de_conv7 = BatchNormalization()(de_conv7)
	de_conv7 = Activation('relu')(de_conv7)

	fcn_output = Conv2D(1, kernel_size, padding='same')(de_conv7)
	fcn_output = BatchNormalization()(fcn_output)
	fcn_output = Activation('relu')(fcn_output)

	model = Model(inputs, fcn_output)

	return model

# proposed method 2
def MCNN_U(input_shape = (96,96)):
	nb_filters=32
	kernel_size=(3,3)
	pool_size=(2,2)
	concat_axis = 3
	weight_decay = 1e-5

	inputs = Input(shape=input_shape+(3,))
	
	# column1
	conv1 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

	conv2 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

	conv3 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
	pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

# 	conv4 = Conv2D(nb_filters*16, kernel_size, padding='same')(pool3)
# 	conv4 = BatchNormalization()(conv4)
# 	conv4 = Activation('relu')(conv4)

	# column2
	kernel_size = (5,5)
	conv1_1 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
	conv1_1 = BatchNormalization()(conv1_1)
	conv1_1 = Activation('relu')(conv1_1)
	pool1_1 = MaxPooling2D(pool_size=pool_size)(conv1_1)

	conv2_1 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1_1)
	conv2_1 = BatchNormalization()(conv2_1)
	conv2_1 = Activation('relu')(conv2_1)
	pool2_1 = MaxPooling2D(pool_size=pool_size)(conv2_1)

	conv3_1 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2_1)
	conv3_1 = BatchNormalization()(conv3_1)
	conv3_1 = Activation('relu')(conv3_1)
	pool3_1 = MaxPooling2D(pool_size=pool_size)(conv3_1)

# 	conv4_1 = Conv2D(nb_filters*16, kernel_size, padding='same')(pool3_1)
# 	conv4_1 = BatchNormalization()(conv4_1)
# 	conv4_1 = Activation('relu')(conv4_1)

	# column3
	kernel_size = (7,7)
	conv1_2 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
	conv1_2 = BatchNormalization()(conv1_2)
	conv1_2 = Activation('relu')(conv1_2)
	pool1_2 = MaxPooling2D(pool_size=pool_size)(conv1_2)

	conv2_2 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1_2)
	conv2_2 = BatchNormalization()(conv2_2)
	conv2_2 = Activation('relu')(conv2_2)
	pool2_2 = MaxPooling2D(pool_size=pool_size)(conv2_2)

	conv3_2 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2_2)
	conv3_2 = BatchNormalization()(conv3_2)
	conv3_2 = Activation('relu')(conv3_2)
	pool3_2 = MaxPooling2D(pool_size=pool_size)(conv3_2)

	# merge the features
	merge_conv = concatenate([pool3, pool3_1, pool3_2], axis = concat_axis)
	
	kernel_size = (3,3)
	conv4 = Conv2D(nb_filters*16, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)

	merge_conv1 = concatenate([conv3, conv3_1, conv3_2], axis = concat_axis)
	kernel_size = (3,3)
	conv4_1 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv1)
	conv4_1 = BatchNormalization()(conv4_1)
	conv4_1 = Activation('relu')(conv4_1)

	merge_conv2 = concatenate([conv2, conv2_1, conv2_2], axis = concat_axis)
	kernel_size = (3,3)
	conv4_2 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv2)
	conv4_2 = BatchNormalization()(conv4_2)
	conv4_2 = Activation('relu')(conv4_2)

	merge_conv3 = concatenate([conv1, conv1_1, conv1_2], axis = concat_axis)
	kernel_size = (3,3)
	conv4_3 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv3)
	conv4_3 = BatchNormalization()(conv4_3)
	conv4_3 = Activation('relu')(conv4_3)

	de_conv5 = Conv2DTranspose(nb_filters*4, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv)
	de_conv5 = concatenate([de_conv5, conv4_1], axis = concat_axis)
	de_conv5 = Conv2D(nb_filters*4, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv5)
	de_conv5 = BatchNormalization()(de_conv5)
	de_conv5 = Activation('relu')(de_conv5)

	de_conv6 = Conv2DTranspose(nb_filters*2, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv5)
	de_conv6 = concatenate([de_conv6, conv4_2], axis = concat_axis)
	de_conv6 = Conv2D(nb_filters*2, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv6)
	de_conv6 = BatchNormalization()(de_conv6)
	de_conv6 = Activation('relu')(de_conv6)

	de_conv7 = Conv2DTranspose(nb_filters, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv6)
	de_conv7 = concatenate([de_conv7, conv4_3], axis = concat_axis)
	de_conv7 = Conv2D(nb_filters, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv7)
	de_conv7 = BatchNormalization()(de_conv7)
	de_conv7 = Activation('relu')(de_conv7)

	fcn_output = Conv2D(1, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv7)
	fcn_output = BatchNormalization()(fcn_output)
	fcn_output = Activation('relu')(fcn_output)

	model = Model(inputs, fcn_output)

	return model

# proposed method 2.1
def imp_MCNN_U(input_shape = (96,96)):
	nb_filters=32
	kernel_size=(3,3)
	pool_size=(2,2)
	concat_axis = 3
	weight_decay = 1e-5

	inputs = Input(shape=input_shape+(3,))
	
	# column1
	conv1 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

	conv2 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

	conv3 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
	pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

# 	conv4 = Conv2D(nb_filters*16, kernel_size, padding='same')(pool3)
# 	conv4 = BatchNormalization()(conv4)
# 	conv4 = Activation('relu')(conv4)

	# column2
	kernel_size = (5,5)
	conv1_1 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
	conv1_1 = BatchNormalization()(conv1_1)
	conv1_1 = Activation('relu')(conv1_1)
	pool1_1 = MaxPooling2D(pool_size=pool_size)(conv1_1)

	conv2_1 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1_1)
	conv2_1 = BatchNormalization()(conv2_1)
	conv2_1 = Activation('relu')(conv2_1)
	pool2_1 = MaxPooling2D(pool_size=pool_size)(conv2_1)

	conv3_1 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2_1)
	conv3_1 = BatchNormalization()(conv3_1)
	conv3_1 = Activation('relu')(conv3_1)
	pool3_1 = MaxPooling2D(pool_size=pool_size)(conv3_1)

# 	conv4_1 = Conv2D(nb_filters*16, kernel_size, padding='same')(pool3_1)
# 	conv4_1 = BatchNormalization()(conv4_1)
# 	conv4_1 = Activation('relu')(conv4_1)

	# column3
	kernel_size = (7,7)
	conv1_2 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
	conv1_2 = BatchNormalization()(conv1_2)
	conv1_2 = Activation('relu')(conv1_2)
	pool1_2 = MaxPooling2D(pool_size=pool_size)(conv1_2)

	conv2_2 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1_2)
	conv2_2 = BatchNormalization()(conv2_2)
	conv2_2 = Activation('relu')(conv2_2)
	pool2_2 = MaxPooling2D(pool_size=pool_size)(conv2_2)

	conv3_2 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2_2)
	conv3_2 = BatchNormalization()(conv3_2)
	conv3_2 = Activation('relu')(conv3_2)
	pool3_2 = MaxPooling2D(pool_size=pool_size)(conv3_2)

	# merge the features
	merge_conv = concatenate([pool3, pool3_1, pool3_2], axis = concat_axis)
	
	kernel_size = (3,3)
	conv4 = Conv2D(nb_filters*16, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)

	merge_conv1 = concatenate([conv3, conv3_1, conv3_2], axis = concat_axis)
	kernel_size = (1,1)
	conv4_1 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv1)
	conv4_1 = BatchNormalization()(conv4_1)
	conv4_1 = Activation('relu')(conv4_1)

	merge_conv2 = concatenate([conv2, conv2_1, conv2_2], axis = concat_axis)
	kernel_size = (1,1)
	conv4_2 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv2)
	conv4_2 = BatchNormalization()(conv4_2)
	conv4_2 = Activation('relu')(conv4_2)

	merge_conv3 = concatenate([conv1, conv1_1, conv1_2], axis = concat_axis)
	kernel_size = (1,1)
	conv4_3 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv3)
	conv4_3 = BatchNormalization()(conv4_3)
	conv4_3 = Activation('relu')(conv4_3)

	# upsamping and conv
	kernel_size = (3,3)
	de_conv5 = Conv2DTranspose(nb_filters*4, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(conv4)
	de_conv5 = concatenate([de_conv5, conv4_1], axis = concat_axis)
	de_conv5 = Conv2D(nb_filters*4, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv5)
	de_conv5 = BatchNormalization()(de_conv5)
	de_conv5 = Activation('relu')(de_conv5)

	de_conv6 = Conv2DTranspose(nb_filters*2, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv5)
	de_conv6 = concatenate([de_conv6, conv4_2], axis = concat_axis)
	de_conv6 = Conv2D(nb_filters*2, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv6)
	de_conv6 = BatchNormalization()(de_conv6)
	de_conv6 = Activation('relu')(de_conv6)

	de_conv7 = Conv2DTranspose(nb_filters, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv6)
	de_conv7 = concatenate([de_conv7, conv4_3], axis = concat_axis)
	de_conv7 = Conv2D(nb_filters, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv7)
	de_conv7 = BatchNormalization()(de_conv7)
	de_conv7 = Activation('relu')(de_conv7)

	fcn_output = Conv2D(1, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv7)
	fcn_output = BatchNormalization()(fcn_output)
	fcn_output = Activation('relu')(fcn_output)

	model = Model(inputs, fcn_output)

	return model

# proposed method 2.1
def imp_MCNN_U2(input_shape = (96,96)):
	nb_filters=32
	kernel_size=(3,3)
	pool_size=(2,2)
	concat_axis = 3
	weight_decay = 1e-5

	inputs = Input(shape=input_shape+(3,))
	
	# column1
	conv1 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

	conv2 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

	conv3 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
	pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

# 	conv4 = Conv2D(nb_filters*16, kernel_size, padding='same')(pool3)
# 	conv4 = BatchNormalization()(conv4)
# 	conv4 = Activation('relu')(conv4)

	# column2
	kernel_size = (5,5)
	conv1_1 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
	conv1_1 = BatchNormalization()(conv1_1)
	conv1_1 = Activation('relu')(conv1_1)
	pool1_1 = MaxPooling2D(pool_size=pool_size)(conv1_1)

	conv2_1 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1_1)
	conv2_1 = BatchNormalization()(conv2_1)
	conv2_1 = Activation('relu')(conv2_1)
	pool2_1 = MaxPooling2D(pool_size=pool_size)(conv2_1)

	conv3_1 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2_1)
	conv3_1 = BatchNormalization()(conv3_1)
	conv3_1 = Activation('relu')(conv3_1)
	pool3_1 = MaxPooling2D(pool_size=pool_size)(conv3_1)

# 	conv4_1 = Conv2D(nb_filters*16, kernel_size, padding='same')(pool3_1)
# 	conv4_1 = BatchNormalization()(conv4_1)
# 	conv4_1 = Activation('relu')(conv4_1)

	# column3
# 	kernel_size = (7,7)
# 	conv1_2 = Conv2D(nb_filters, kernel_size, padding='same',
# 		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
# 	conv1_2 = BatchNormalization()(conv1_2)
# 	conv1_2 = Activation('relu')(conv1_2)
# 	pool1_2 = MaxPooling2D(pool_size=pool_size)(conv1_2)
# 
# 	conv2_2 = Conv2D(nb_filters*2, kernel_size, padding='same',
# 		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1_2)
# 	conv2_2 = BatchNormalization()(conv2_2)
# 	conv2_2 = Activation('relu')(conv2_2)
# 	pool2_2 = MaxPooling2D(pool_size=pool_size)(conv2_2)
# 
# 	conv3_2 = Conv2D(nb_filters*4, kernel_size, padding='same',
# 		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2_2)
# 	conv3_2 = BatchNormalization()(conv3_2)
# 	conv3_2 = Activation('relu')(conv3_2)
# 	pool3_2 = MaxPooling2D(pool_size=pool_size)(conv3_2)

	# merge the features
	merge_conv = concatenate([pool3, pool3_1], axis = concat_axis)
	
	kernel_size = (3,3)
	conv4 = Conv2D(nb_filters*16, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)

	merge_conv1 = concatenate([conv3, conv3_1], axis = concat_axis)
	kernel_size = (1,1)
	conv4_1 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv1)
	conv4_1 = BatchNormalization()(conv4_1)
	conv4_1 = Activation('relu')(conv4_1)

	merge_conv2 = concatenate([conv2, conv2_1], axis = concat_axis)
	kernel_size = (1,1)
	conv4_2 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv2)
	conv4_2 = BatchNormalization()(conv4_2)
	conv4_2 = Activation('relu')(conv4_2)

	merge_conv3 = concatenate([conv1, conv1_1], axis = concat_axis)
	kernel_size = (1,1)
	conv4_3 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv3)
	conv4_3 = BatchNormalization()(conv4_3)
	conv4_3 = Activation('relu')(conv4_3)

	# upsamping and conv
	kernel_size = (3,3)
	de_conv5 = Conv2DTranspose(nb_filters*4, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(conv4)
	de_conv5 = concatenate([de_conv5, conv4_1], axis = concat_axis)
	de_conv5 = Conv2D(nb_filters*4, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv5)
	de_conv5 = BatchNormalization()(de_conv5)
	de_conv5 = Activation('relu')(de_conv5)

	de_conv6 = Conv2DTranspose(nb_filters*2, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv5)
	de_conv6 = concatenate([de_conv6, conv4_2], axis = concat_axis)
	de_conv6 = Conv2D(nb_filters*2, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv6)
	de_conv6 = BatchNormalization()(de_conv6)
	de_conv6 = Activation('relu')(de_conv6)

	de_conv7 = Conv2DTranspose(nb_filters, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv6)
	de_conv7 = concatenate([de_conv7, conv4_3], axis = concat_axis)
	de_conv7 = Conv2D(nb_filters, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv7)
	de_conv7 = BatchNormalization()(de_conv7)
	de_conv7 = Activation('relu')(de_conv7)

	fcn_output = Conv2D(1, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv7)
	fcn_output = BatchNormalization()(fcn_output)
	fcn_output = Activation('relu')(fcn_output)

	model = Model(inputs, fcn_output)

	return model


# proposed method 3
def _bot_cnn_bn_relu(nb_filters, input, kernel_size, weight_decay):
	def f(input):
		conv_a = Conv2D(int(nb_filters/4), (1,1), padding='same',
			kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(input)
		norm_a = BatchNormalization()(conv_a)
		act_a = Activation(activation = 'relu')(norm_a)
		conv_b = Conv2D(int(nb_filters/4), kernel_size, padding='same',
			kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(act_a)
		norm_b = BatchNormalization()(conv_b)
		act_b = Activation(activation = 'relu')(norm_b)
		conv_c = Conv2D(nb_filters, (1,1), padding='same',
			kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(act_b)
		norm_c = BatchNormalization()(conv_c)
		act_c = Activation(activation = 'relu')(norm_c)
		return act_c
	return f

def MCNN_Bot_U_x3(input_shape = (96,96)):
	nb_filters=32
	kernel_size=(3,3)
	pool_size=(2,2)
	concat_axis = 3
	weight_decay = 1e-5
	inputs = Input(shape=input_shape+(3,))
	
	# column1
	conv1 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

	conv2 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

	conv3 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
	pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

	# column2
	kernel_size = (5,5)
	conv1_1 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
	conv1_1 = BatchNormalization()(conv1_1)
	conv1_1 = Activation('relu')(conv1_1)
	pool1_1 = MaxPooling2D(pool_size=pool_size)(conv1_1)

	conv2_1 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1_1)
	conv2_1 = BatchNormalization()(conv2_1)
	conv2_1 = Activation('relu')(conv2_1)
	pool2_1 = MaxPooling2D(pool_size=pool_size)(conv2_1)

	conv3_1 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2_1)
	conv3_1 = BatchNormalization()(conv3_1)
	conv3_1 = Activation('relu')(conv3_1)
	pool3_1 = MaxPooling2D(pool_size=pool_size)(conv3_1)

	# column3
	kernel_size = (7,7)
	conv1_2 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
	conv1_2 = BatchNormalization()(conv1_2)
	conv1_2 = Activation('relu')(conv1_2)
	pool1_2 = MaxPooling2D(pool_size=pool_size)(conv1_2)

	conv2_2 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1_2)
	conv2_2 = BatchNormalization()(conv2_2)
	conv2_2 = Activation('relu')(conv2_2)
	pool2_2 = MaxPooling2D(pool_size=pool_size)(conv2_2)

	conv3_2 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2_2)
	conv3_2 = BatchNormalization()(conv3_2)
	conv3_2 = Activation('relu')(conv3_2)
	pool3_2 = MaxPooling2D(pool_size=pool_size)(conv3_2)

	# merge the features
	merge_conv = concatenate([pool3, pool3_1, pool3_2], axis = concat_axis)
	
	kernel_size = (3,3)
	conv4 = Conv2D(nb_filters*16, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)

	merge_conv1 = concatenate([conv3, conv3_1, conv3_2], axis = concat_axis)
	kernel_size = (3,3)
	conv4_1 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv1)
	conv4_1 = BatchNormalization()(conv4_1)
	conv4_1 = Activation('relu')(conv4_1)

	merge_conv2 = concatenate([conv2, conv2_1, conv2_2], axis = concat_axis)
	kernel_size = (3,3)
	conv4_2 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv2)
	conv4_2 = BatchNormalization()(conv4_2)
	conv4_2 = Activation('relu')(conv4_2)

	merge_conv3 = concatenate([conv1, conv1_1, conv1_2], axis = concat_axis)
	kernel_size = (3,3)
	conv4_3 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv3)
	conv4_3 = BatchNormalization()(conv4_3)
	conv4_3 = Activation('relu')(conv4_3)

	de_conv5 = Conv2DTranspose(nb_filters*4, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv)
	de_conv5 = concatenate([de_conv5, conv4_1], axis = concat_axis)
	de_conv5 = Conv2D(nb_filters*4, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv5)
	de_conv5 = BatchNormalization()(de_conv5)
	de_conv5 = Activation('relu')(de_conv5)

	de_conv6 = Conv2DTranspose(nb_filters*2, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv5)
	de_conv6 = concatenate([de_conv6, conv4_2], axis = concat_axis)
	de_conv6 = Conv2D(nb_filters*2, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv6)
	de_conv6 = BatchNormalization()(de_conv6)
	de_conv6 = Activation('relu')(de_conv6)

	de_conv7 = Conv2DTranspose(nb_filters, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv6)
	de_conv7 = concatenate([de_conv7, conv4_3], axis = concat_axis)
	de_conv7 = Conv2D(nb_filters, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv7)
	de_conv7 = BatchNormalization()(de_conv7)
	de_conv7 = Activation('relu')(de_conv7)

	fcn_output = Conv2D(1, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv7)
	fcn_output = BatchNormalization()(fcn_output)
	fcn_output = Activation('relu')(fcn_output)

	model = Model(inputs, fcn_output)

	return model

# proposed method 4
def MCNN_U_x4(input_shape = (96,96)):
	nb_filters=32
	kernel_size=(3,3)
	pool_size=(2,2)
	concat_axis = 3
	weight_decay = 1e-5

	inputs = Input(shape=input_shape+(3,))
	
	# column1
	conv1 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

	conv2 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

	conv3 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
	pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

	conv4 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)
	pool4 = MaxPooling2D(pool_size=pool_size)(conv4)

# 	conv4 = Conv2D(nb_filters*16, kernel_size, padding='same')(pool3)
# 	conv4 = BatchNormalization()(conv4)
# 	conv4 = Activation('relu')(conv4)

	# column2
	kernel_size = (5,5)
	conv1_1 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
	conv1_1 = BatchNormalization()(conv1_1)
	conv1_1 = Activation('relu')(conv1_1)
	pool1_1 = MaxPooling2D(pool_size=pool_size)(conv1_1)

	conv2_1 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1_1)
	conv2_1 = BatchNormalization()(conv2_1)
	conv2_1 = Activation('relu')(conv2_1)
	pool2_1 = MaxPooling2D(pool_size=pool_size)(conv2_1)

	conv3_1 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2_1)
	conv3_1 = BatchNormalization()(conv3_1)
	conv3_1 = Activation('relu')(conv3_1)
	pool3_1 = MaxPooling2D(pool_size=pool_size)(conv3_1)

	conv4_1 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool3_1)
	conv4_1 = BatchNormalization()(conv4_1)
	conv4_1 = Activation('relu')(conv4_1)
	pool4_1 = MaxPooling2D(pool_size=pool_size)(conv4_1)

# 	conv4_1 = Conv2D(nb_filters*16, kernel_size, padding='same')(pool3_1)
# 	conv4_1 = BatchNormalization()(conv4_1)
# 	conv4_1 = Activation('relu')(conv4_1)

	# column3
	kernel_size = (7,7)
	conv1_2 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(inputs)
	conv1_2 = BatchNormalization()(conv1_2)
	conv1_2 = Activation('relu')(conv1_2)
	pool1_2 = MaxPooling2D(pool_size=pool_size)(conv1_2)

	conv2_2 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool1_2)
	conv2_2 = BatchNormalization()(conv2_2)
	conv2_2 = Activation('relu')(conv2_2)
	pool2_2 = MaxPooling2D(pool_size=pool_size)(conv2_2)

	conv3_2 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool2_2)
	conv3_2 = BatchNormalization()(conv3_2)
	conv3_2 = Activation('relu')(conv3_2)
	pool3_2 = MaxPooling2D(pool_size=pool_size)(conv3_2)

	conv4_2 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(pool3_2)
	conv4_2 = BatchNormalization()(conv4_2)
	conv4_2 = Activation('relu')(conv4_2)
	pool4_2 = MaxPooling2D(pool_size=pool_size)(conv4_2)

# 	conv4_2 = Conv2D(nb_filters*16, kernel_size, padding='same')(pool3_2)
# 	conv4_2 = BatchNormalization()(conv4_2)
# 	conv4_2 = Activation('relu')(conv4_2)

	# merge the features
	merge_conv = concatenate([pool4, pool4_1, pool4_2], axis = concat_axis)
	
	kernel_size = (3,3)
	fc = Conv2D(nb_filters*16, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv)
	fc = BatchNormalization()(fc)
	fc = Activation('relu')(fc)

	merge_conv0 = concatenate([conv4, conv4_1, conv4_2], axis = concat_axis)
	kernel_size = (3,3)
	skip_0 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv0)
	skip_0 = BatchNormalization()(skip_0)
	skip_0 = Activation('relu')(skip_0)

	merge_conv1 = concatenate([conv3, conv3_1, conv3_2], axis = concat_axis)
	kernel_size = (3,3)
	skip_1 = Conv2D(nb_filters*4, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv1)
	skip_1 = BatchNormalization()(skip_1)
	skip_1 = Activation('relu')(skip_1)

	merge_conv2 = concatenate([conv2, conv2_1, conv2_2], axis = concat_axis)
	kernel_size = (3,3)
	skip_2 = Conv2D(nb_filters*2, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv2)
	skip_2 = BatchNormalization()(skip_2)
	skip_2 = Activation('relu')(skip_2)

	merge_conv3 = concatenate([conv1, conv1_1, conv1_2], axis = concat_axis)
	kernel_size = (3,3)
	skip_3 = Conv2D(nb_filters, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv3)
	skip_3 = BatchNormalization()(skip_3)
	skip_3 = Activation('relu')(skip_3)

	de_conv5 = Conv2DTranspose(nb_filters*4, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(merge_conv)
	de_conv5 = concatenate([de_conv5, skip_0], axis = concat_axis)
	de_conv5 = Conv2D(nb_filters*4, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv5)
	de_conv5 = BatchNormalization()(de_conv5)
	de_conv5 = Activation('relu')(de_conv5)

	de_conv6 = Conv2DTranspose(nb_filters*4, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv5)
	de_conv6 = concatenate([de_conv6, skip_1], axis = concat_axis)
	de_conv6 = Conv2D(nb_filters*4, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv6)
	de_conv6 = BatchNormalization()(de_conv6)
	de_conv6 = Activation('relu')(de_conv6)

	de_conv7 = Conv2DTranspose(nb_filters*2, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv6)
	de_conv7 = concatenate([de_conv7, skip_2], axis = concat_axis)
	de_conv7 = Conv2D(nb_filters*2, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv7)
	de_conv7 = BatchNormalization()(de_conv7)
	de_conv7 = Activation('relu')(de_conv7)

	de_conv8 = Conv2DTranspose(nb_filters, kernel_size, strides=(2, 2), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv7)
	de_conv8 = concatenate([de_conv8, skip_3], axis = concat_axis)
	de_conv8 = Conv2D(nb_filters, kernel_size, strides=(1, 1), padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv8)
	de_conv8 = BatchNormalization()(de_conv8)
	de_conv8 = Activation('relu')(de_conv8)

	fcn_output = Conv2D(1, kernel_size, padding='same',
		kernel_regularizer = l2(weight_decay),bias_regularizer=l2(weight_decay))(de_conv8)
	fcn_output = BatchNormalization()(fcn_output)
	fcn_output = Activation('relu')(fcn_output)

	model = Model(inputs, fcn_output)

	return model

# baseline method 2
def FCRN_B(input_shape = (96,96)):
	nb_filters=32
	kernel_size=(3,3)
	kernel_size1 = (5,5)
	pool_size=(2,2)
	concat_axis = 3

	inputs = Input(shape=input_shape+(3,))
	conv1 = Conv2D(nb_filters, kernel_size, padding='same')(inputs)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
# 	pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

	conv2 = Conv2D(nb_filters*2, kernel_size, padding='same')(conv1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

	conv3 = Conv2D(nb_filters*4, kernel_size, padding='same')(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
# 	pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

	conv4 = Conv2D(nb_filters*8, kernel_size, padding='same')(conv3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)
	pool4 = MaxPooling2D(pool_size=pool_size)(conv4)

	conv5 = Conv2D(nb_filters*8, kernel_size1, padding='same')(pool4)
	conv5 = BatchNormalization()(conv5)
	conv5 = Activation('relu')(conv5)

	de_conv6 = Conv2DTranspose(nb_filters*8, kernel_size, strides=(2, 2), padding='same')(conv5)
	de_conv6 = BatchNormalization()(de_conv6)
	de_conv6 = Activation('relu')(de_conv6)

	fcn_output = Conv2DTranspose(1, kernel_size1, strides=(2, 2), padding='same')(de_conv6)
	fcn_output = BatchNormalization()(fcn_output)
	fcn_output = Activation('relu')(fcn_output)

	model = Model(inputs, fcn_output)

	return model

# comparison method
