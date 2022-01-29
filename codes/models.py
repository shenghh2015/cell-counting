from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, BatchNormalization, Input, Cropping2D, concatenate
from tensorflow.keras.layers import MaxPooling2D, Conv2D, UpSampling2D
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

weight_decay = 1e-5

def _conv_bn_relu(nb_filter, kernel_size):
	def f(input):
		conv_a = Conv2D(nb_filter, kernel_size, padding='same', 
			kernel_initializer='orthogonal', kernel_regularizer = l2(weight_decay),
			bias_regularizer=l2(weight_decay))(input)
		norm_a = BatchNormalization()(conv_a)
		act_a = Activation(activation = 'relu')(norm_a)
		return act_a
	return f

def FCRN_A_base(input):
	nb_filter = 32
	kernel_size = (3,3)
	block1 = _conv_bn_relu(nb_filter, kernel_size)(input)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	block2 = _conv_bn_relu(nb_filter*2, kernel_size)(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	block3 = _conv_bn_relu(nb_filter*4, kernel_size)(pool2)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	block4 = _conv_bn_relu(nb_filter*16, kernel_size)(pool3)
	up5 = UpSampling2D(size=(2, 2))(block4)
	block5 = _conv_bn_relu(nb_filter*4, kernel_size)(up5)
	up6 = UpSampling2D(size=(2, 2))(block5)
	block6 = _conv_bn_relu(nb_filter*2, kernel_size)(up6)
	up7 = UpSampling2D(size=(2, 2))(block6)
	block7 = _conv_bn_relu(nb_filter, kernel_size)(up7)
	return block7

def U_net_base(input):
	nb_filter = 32
	kernel_size = (3,3)
	block1 = _conv_bn_relu(nb_filter, kernel_size)(input)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	block2 = _conv_bn_relu(nb_filter*2, kernel_size)(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	block3 = _conv_bn_relu(nb_filter*4, kernel_size)(pool2)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	block4 = _conv_bn_relu(nb_filter*16, kernel_size)(pool3)
	up5 = concatenate([UpSampling2D(size=(2, 2))(block4), block3], axis = -1)
	block5 = _conv_bn_relu(nb_filter*4, kernel_size)(up5)
	up6 = concatenate([UpSampling2D(size=(2, 2))(block5), block2], axis = -1)
	block6 = _conv_bn_relu(nb_filter*2, kernel_size)(up6)
	up7 = concatenate([UpSampling2D(size=(2, 2))(block6), block1], axis = -1)
	block7 = _conv_bn_relu(nb_filter, kernel_size)(up7)
	return block7

def U_net_Multi_base(input):
	nb_filter = 32
	kernel_size = (3,3)
	block1 = _conv_bn_relu(nb_filter, kernel_size)(input)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	block2 = _conv_bn_relu(nb_filter*2, kernel_size)(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
	block3 = _conv_bn_relu(nb_filter*4, kernel_size)(pool2)
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
	block4 = _conv_bn_relu(nb_filter*16, kernel_size)(pool3)
	up5 = concatenate([UpSampling2D(size=(2, 2))(block4), block3], axis = -1)
	block5 = _conv_bn_relu(nb_filter*4, kernel_size)(up5)
	up6 = concatenate([UpSampling2D(size=(2, 2))(block5), block2], axis = -1)
	block6 = _conv_bn_relu(nb_filter*2, kernel_size)(up6)
	up7 = concatenate([UpSampling2D(size=(2, 2))(block6), block1], axis = -1)
	block7 = _conv_bn_relu(nb_filter, kernel_size)(up7)
	out_block1 = _conv_bn_relu(nb_filter, kernel_size)(block4)
	out_block2 = _conv_bn_relu(nb_filter, kernel_size)(block5)
	out_block3 = _conv_bn_relu(nb_filter, kernel_size)(block6)
	return out_block1, out_block2, out_block3, block7

def FCRN_A(input_shape = (None, None, 3), activation = 'linear'):
	input_ = Input(shape = input_shape)
	act_ = FCRN_A_base(input_)
	density_pred =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = activation)(act_)
	model = Model(inputs = input_, outputs = density_pred)
	return model

def C_FCRN(input_shape = (None, None, 3), activation = 'linear'):
	input_ = Input(shape = input_shape)
	act_ = U_net_base(input_)
	density_pred =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = activation)(act_)
	model = Model (inputs = input_, outputs = density_pred)
	return model

# loss_weights=[1./64,1/16, 1./4, 1]
def C_FCRN_Aux(input_shape = (None, None, 3), activation = 'linear'):
	input_ = Input(shape = input_shape)
	out_block1, out_block2, out_block3, block7 = U_net_Multi_base(input_)
	density_pred_1 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = activation, name = 'red8')(out_block1)	
	density_pred_2 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = activation, name = 'red4')(out_block2)
	density_pred_3 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = activation, name = 'red2')(out_block3)
	density_pred_4 =  Conv2D(1, (1,1), padding='same', kernel_initializer='orthogonal', activation = activation, name = 'original')(block7)
	model = Model (inputs = input_, outputs = [density_pred_1, density_pred_2, density_pred_3, density_pred_4])
	return model