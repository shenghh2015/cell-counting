from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, merge, Input, Cropping2D, concatenate, Add
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D,GlobalAveragePooling2D, Conv2D,Lambda, UpSampling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU, ELU  
from keras.regularizers import l2
from keras.optimizers import SGD
import keras.backend as K
import keras

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

def U_net_prymarid_base(input, dropout = None):
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
	# branch 1
	out_block1 = _conv_bn_relu(nb_filter, kernel_size)(block4)
	# branch 2
	out_block2 = _conv_bn_relu(nb_filter, kernel_size)(block5)
	# branch 3
	out_block3 = _conv_bn_relu(nb_filter, kernel_size)(block6)
	# branch 4
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

def buildPrymaridModel_U_net(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse', activation = 'linear'):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	out_block1, out_block2, out_block3, block7 = U_net_prymarid_base(input_, dropout)
	# =========================================================================
	density_pred_1 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'relu')(out_block1)
	density_1 = UpSampling2D(size=(8, 8))(density_pred_1)
	density_pred_2 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'relu')(out_block2)
	density_2 = UpSampling2D(size=(4, 4))(density_pred_2)
	density_pred_3 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'relu')(out_block3)
	density_3 = UpSampling2D(size=(2, 2))(density_pred_3)
	density_4 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'relu')(block7)
	density_map = Add()([density_1, density_2, density_3, density_4])
	# =========================================================================
# 	model = Model (input = input_, output = [density_pred_1, density_pred_2, density_pred_3, density_pred_4])
	model = Model (input = input_, output = density_map)
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse')
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae')
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn)
	return model

def buildPrymaridModel_U_net_v2(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse', activation = 'linear'):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	out_block1, out_block2, out_block3, block7 = U_net_prymarid_base(input_, dropout)
	# =========================================================================
	density_pred_1 =  Conv2D(32, (3,3), padding='same', kernel_initializer='orthogonal', activation = 'relu')(out_block1)
	density_1 = UpSampling2D(size=(8, 8))(density_pred_1)
	density_pred_2 =  Conv2D(32, (3,3), padding='same', kernel_initializer='orthogonal', activation = 'relu')(out_block2)
	density_2 = UpSampling2D(size=(4, 4))(density_pred_2)
	density_pred_3 =  Conv2D(32, (3,3), padding='same', kernel_initializer='orthogonal', activation = 'relu')(out_block3)
	density_3 = UpSampling2D(size=(2, 2))(density_pred_3)
	density_4 =  Conv2D(32, (3,3), padding='same', kernel_initializer='orthogonal', activation = 'relu')(block7)
	density_features = Add()([density_1, density_2, density_3, density_4])
	density_map = Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'relu')(density_features)
	# =========================================================================
# 	model = Model (input = input_, output = [density_pred_1, density_pred_2, density_pred_3, density_pred_4])
	model = Model (input = input_, output = density_map)
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse')
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae')
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn)
	return model

def buildPrymaridModel_U_net_v3(input_shape, lr = 1e-2, dropout = None, loss_fcn = 'mse', activation = 'linear'):
	input_ = Input (shape = input_shape+(3,))
	# =========================================================================
	out_block1, out_block2, out_block3, block7 = U_net_prymarid_base(input_, dropout)
	# =========================================================================
	density_pred_1 =  Conv2D(32, (3,3), padding='same', kernel_initializer='orthogonal', activation = 'relu')(out_block1)
	density_1 = UpSampling2D(size=(8, 8))(density_pred_1)
	density_pred_2 =  Conv2D(32, (3,3), padding='same', kernel_initializer='orthogonal', activation = 'relu')(out_block2)
	density_2 = UpSampling2D(size=(4, 4))(density_pred_2)
	density_pred_3 =  Conv2D(32, (3,3), padding='same', kernel_initializer='orthogonal', activation = 'relu')(out_block3)
	density_3 = UpSampling2D(size=(2, 2))(density_pred_3)
	density_4 =  Conv2D(32, (3,3), padding='same', kernel_initializer='orthogonal', activation = 'relu')(block7)
	density_features = Add()([density_1, density_2, density_3, density_4])
	density_map = Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = 'relu')(density_features)
	# =========================================================================
# 	model = Model (input = input_, output = [density_pred_1, density_pred_2, density_pred_3, density_pred_4])
	model = Model (input = input_, output = density_map)
	opt = SGD(lr = lr, momentum = 0.9, nesterov = True)
	if loss_fcn == 'mse':
		model.compile(optimizer = opt, loss = 'mse')
	elif loss_fcn == 'mae':
		model.compile(optimizer = opt, loss = 'mae')
	else:
		los_fn = globals()[loss_fcn]
		model.compile(optimizer = opt, loss = los_fn)
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

