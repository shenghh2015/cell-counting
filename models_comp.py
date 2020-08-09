from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, merge, Input, Cropping2D, concatenate, Add
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D,GlobalAveragePooling2D, Conv2D,Lambda, UpSampling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU, ELU 
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
import keras.backend as K
import keras


# this implementation reduce the checkerboard artifact
# weight_decay = 1e-5
K.set_image_dim_ordering('tf')

def _conv_bn_leaky(nb_filter, kernel_size, pad = 'same'):
	def f(input):
		conv_a = Conv2D(nb_filter, kernel_size, padding= pad,
			kernel_initializer='glorot_uniform')(input)
		norm_a = BatchNormalization()(conv_a)
		act_a = LeakyReLU()(norm_a)
# 		act_a = Activation(activation=LeakyReLU())(norm_a)
		return act_a
	return f

def _concate_conv(nb_filter1, kernel_size1, nb_filter2, kernel_size2):
	def f(input):
		conv_a = _conv_bn_leaky(nb_filter1, kernel_size1, pad = 'same')(input)
		conv_b = _conv_bn_leaky(nb_filter2, kernel_size2, pad = 'same')(input)
		return concatenate([conv_a, conv_b], axis = -1)
	return f

def Count_ception_base(input):
	kernel_size_a = (1,1)
	kernel_size_b = (3,3)
	kernel_size_c = (14,14)
	kernel_size_d = (17,17)
	block1 = _conv_bn_leaky(64, kernel_size_b, 'valid')(input)
	block2 = _concate_conv(16, kernel_size_a, 16, kernel_size_b)(block1)
	block3 = _concate_conv(16, kernel_size_a, 32, kernel_size_b)(block2)
	block4 = _conv_bn_leaky(16, kernel_size_c, 'valid')(block3)
	block5 = _concate_conv(112, kernel_size_a, 48, kernel_size_b)(block4)
	block6 = _concate_conv(40, kernel_size_a, 40, kernel_size_b)(block5)
	block7 = _concate_conv(32, kernel_size_a, 96, kernel_size_b)(block6)
	block8 = _conv_bn_leaky(16, kernel_size_d, 'valid')(block7)
	block9 = _conv_bn_leaky(64, kernel_size_a, 'same')(block8)
	block10 = _conv_bn_leaky(1, kernel_size_a, 'same')(block9)
	return block10

def buildModel_Count_ception(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mae'):
	_shape = (input_shape[0]+62, input_shape[1]+62)
	input_ = Input (shape = _shape+(3,))
	# =========================================================================
	act_ = Count_ception_base (input_)
	# =========================================================================
	model = Model (input = input_, output = act_)
	opt = Adam(lr = lr)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse')
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae')
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn)
	return model

	
# def _conv_bn_relu(nb_filter, kernel_size):
# 	def f(input):
# 		conv_a = Conv2D(nb_filter, kernel_size, padding='same',
# 			kernel_initializer='glorot_uniform', kernel_regularizer = l2(weight_decay),
# 			bias_regularizer=l2(weight_decay))(input)
# 		norm_a = BatchNormalization()(conv_a)
# 		act_a = Activation(activation = 'relu')(norm_a)
# 		return act_a
# 	return f

# def FCRN_A_base(input, dropout = None):
# 	nb_filter = 32
# 	kernel_size = (3,3)
# # 	input = Input(shape=input_shape+(3,))
# 	block1 = _conv_bn_relu(nb_filter, kernel_size)(input)
# 	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
# 	# =========================================================================
# 	block2 = _conv_bn_relu(nb_filter*2, kernel_size)(pool1)
# 	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
# 	# =========================================================================
# 	block3 = _conv_bn_relu(nb_filter*4, kernel_size)(pool2)
# 	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
# 	# =========================================================================
# 	block4 = _conv_bn_relu(nb_filter*16, kernel_size)(pool3)
# 	if dropout:
# 		block4 = Dropout(dropout)(block4)
# 	# =========================================================================
# 	up5 = UpSampling2D(size=(2, 2))(block4)
# 	block5 = _conv_bn_relu(nb_filter*4, kernel_size)(up5)
# 	# =========================================================================
# 	up6 = UpSampling2D(size=(2, 2))(block5)
# 	block6 = _conv_bn_relu(nb_filter*2, kernel_size)(up6)
# 	# =========================================================================
# 	up7 = UpSampling2D(size=(2, 2))(block6)
# 	block7 = _conv_bn_relu(nb_filter, kernel_size)(up7)
# 	return block7

# def buildModel_FCRN_A(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse'):
# 	input_ = Input (shape = input_shape+(3,))
# 	# =========================================================================
# 	act_ = FCRN_A_base (input_, dropout)
# 	# =========================================================================
# 	density_pred =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'linear')(act_)
# 	# =========================================================================
# 	model = Model (input = input_, output = density_pred)
# 	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
# 	if loss_fcn == 'mse':
# 		model.compile(optimizer = opt, loss = 'mse')
# 	elif loss_fcn == 'mae':
# 		model.compile(optimizer = opt, loss = 'mae')
# 	else:
# 		los_fn = globals()[loss_fcn]
# 		model.compile(optimizer = opt, loss = los_fn)
# 	return model