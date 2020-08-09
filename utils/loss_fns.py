import keras.backend as K

def mse_err(y_true, y_pred):
	pix_err = K.mean(K.square(y_true-y_pred),axis = [1,2,-1])
	return pix_err

def mse_ct_err(y_true, y_pred):
	ct_true = K.sum(y_true, axis= [1,2,-1])
	ct_pred = K.sum(y_pred, axis= [1,2,-1])
	pix_err = K.sum(K.square(y_true-y_pred),axis = [1,2,-1])
	ct_err = K.square(ct_true-ct_pred)
# 	shp = y_true.get_shape().as_list()
	img_size = K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2]
# 	img_size = 128*128
	return K.mean(pix_err+0.0005*ct_err)/img_size
# 	return K.mean(pix_err)/img_size