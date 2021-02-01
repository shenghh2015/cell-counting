# import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from tensorflow.keras.layers import ELU, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GaussianDropout

import numpy as np

smooth = 1.
dropout_rate = 0.5
act = "relu"
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
		add_ = tf.keras.layers.Add()([input, conv_])
		return add_
	return f

########################################

"""
Efficent and Regression Net [Xie et.al, 2018]
Total params: 7,759,521
"""
def Reg_Net(img_rows, img_cols, color_type=3):

    nb_filter = [32,64,128,256,256]

    # Handle Dimension Ordering for different backends
    global bn_axis
    bn_axis = 3
    img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    
    conv1_1 = _conv_bn_relu(nb_filter=nb_filter[0], kernel_size=3)(img_input)
    conv1_2 = _res_conv_bn_relu(nb_filter=nb_filter[0], kernel_size=3)(conv1_1)
    conv1_3 = _conv_bn_relu(nb_filter=nb_filter[1], kernel_size=3)(conv1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_3)

#     conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    conv2_1 = _res_conv_bn_relu(nb_filter=nb_filter[1], kernel_size=3)(pool1)
    conv2_2 = _conv_bn_relu(nb_filter=nb_filter[2], kernel_size=3)(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_2)

    conv3_1 = _res_conv_bn_relu(nb_filter=nb_filter[2], kernel_size=3)(pool2)
    conv3_2 = _conv_bn_relu(nb_filter=nb_filter[3], kernel_size=3)(conv3_1)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_2)

    conv4_1 = _res_conv_bn_relu(nb_filter=nb_filter[3], kernel_size=3)(pool3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    conv5_1 = _res_conv_bn_relu(nb_filter=nb_filter[4], kernel_size=3)(pool4)

    up4 = UpSampling2D(size=(2, 2))(conv5_1)
    dconv4_1 = concatenate([up4, conv4_1], name='merge4', axis=bn_axis)
    dconv4_2 = _conv_bn_relu(nb_filter=nb_filter[3], kernel_size=3)(dconv4_1)

    up3 = UpSampling2D(size=(2, 2))(dconv4_2)
    dconv3_1 = concatenate([up3, conv3_1], name='merge3', axis=bn_axis)
    dconv3_2 = _conv_bn_relu(nb_filter=nb_filter[2], kernel_size=3)(dconv3_1)

    up2 = UpSampling2D(size=(2, 2))(dconv3_2)
    dconv2_1 = concatenate([up2, conv2_1], name='merge2', axis=bn_axis)
    dconv2_2 = _conv_bn_relu(nb_filter=nb_filter[1], kernel_size=3)(dconv2_1)

    up1 = UpSampling2D(size=(2, 2))(dconv2_2)
    dconv1_1 = concatenate([up1, conv1_2], name='merge1', axis=bn_axis)
    dconv1_2 = _conv_bn_relu(nb_filter=nb_filter[0], kernel_size=3)(dconv1_1)

    reg_output = Conv2D(1, (1, 1), activation='relu', name='output', kernel_initializer = 'orthogonal', padding='same', kernel_regularizer=l2(1e-5))(dconv1_2)

    model = Model(inputs=img_input, outputs=reg_output)

    return model

# from struct_regress_model import Reg_Net
# model = Reg_Net(None, None, color_type = 3)