from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras
import keras.backend as K
from math import floor
import numpy as np
from scipy import signal as sg
import random
import os
import time
from utils.file_load_save import *

from utils.loss_fns import *

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

# plot and save the file
def plot_loss(model_name,loss,val_loss):
	generate_folder(model_name)
	f_out=model_name+'/loss_epochs.png'
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	start_idx = 4
	if len(loss)>start_idx:
		fig = Figure(figsize=(8,6))
		ax = fig.add_subplot(1,1,1)
		ax.plot(loss[start_idx:],'b-',linewidth=1.3)
		ax.plot(val_loss[start_idx:],'r-',linewidth=1.3)
		ax.set_title('Model Loss')
		ax.set_ylabel('MSE')
		ax.set_xlabel('epochs')
		ax.legend(['train', 'test'], loc='upper left')  
		canvas = FigureCanvasAgg(fig)
		canvas.print_figure(f_out, dpi=80)

def plot_multi_loss(model_name,train_loss_dic,val_loss_dic):
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	generate_folder(model_name)
	f_out=model_name+'/loss_epochs.png'
	start_idx = 4
	if (len(train_loss_dic['loss'])>4):
		fig = Figure(figsize=(14,8))
		ax = fig.add_subplot(2,3,1)
		ax.plot(train_loss_dic['loss'][start_idx:],'b-',linewidth=1.3)
		ax.plot(val_loss_dic['loss'][start_idx:],'r-',linewidth=1.3)
		ax.set_title('Total loss')
		ax.set_ylabel('MSE')
		ax.set_xlabel('epochs')
		ax.legend(['train', 'test'], loc='upper left')
		bx = fig.add_subplot(2,3,2)
		bx.plot(train_loss_dic['ori'][start_idx:],'b-',linewidth=1.3)
		bx.plot(val_loss_dic['ori'][start_idx:],'r-',linewidth=1.3)
		bx.set_title('Red_1 loss')
		bx.set_ylabel('MSE')
		bx.set_xlabel('epochs')
		bx.legend(['train', 'test'], loc='upper left')
		cx = fig.add_subplot(2,3,3)
		cx.plot(train_loss_dic['red2'][start_idx:],'b-',linewidth=1.3)
		cx.plot(val_loss_dic['red2'][start_idx:],'r-',linewidth=1.3)
		cx.set_title('Red_2 loss')
		cx.set_ylabel('MSE')
		cx.set_xlabel('epochs')
		cx.legend(['train', 'test'], loc='upper left')
		dx = fig.add_subplot(2,3,4)
		dx.plot(train_loss_dic['red4'][start_idx:],'b-',linewidth=1.3)
		dx.plot(val_loss_dic['red4'][start_idx:],'r-',linewidth=1.3)
		dx.set_title('Red_4 loss')
		dx.set_ylabel('MSE')
		dx.set_xlabel('epochs')
		dx.legend(['train', 'test'], loc='upper left')
		ex = fig.add_subplot(2,3,5)
		ex.plot(train_loss_dic['red8'][start_idx:],'b-',linewidth=1.3)
		ex.plot(val_loss_dic['red8'][start_idx:],'r-',linewidth=1.3)
		ex.set_title('Red_8 loss')
		ex.set_ylabel('MSE')
		ex.set_xlabel('epochs')
		ex.legend(['train', 'test'], loc='upper left')  
		canvas = FigureCanvasAgg(fig)
		canvas.print_figure(f_out, dpi=80)

def save_train_loss(model_path,loss_tr, loss_te):
	import os
	generate_folder(model_path)
# 	f_loss_tr=model_path+'/loss_tr.txt'
# 	f_loss_te=model_path+'/loss_te.txt'
# 	np.savetxt(f_loss_tr,loss_tr);
# 	np.savetxt(f_loss_te,loss_te);
	pkl_fn = os.path.join(model_path, 'training.pkl')
	save_pickles(pkl_fn,loss_tr,loss_te, [], keys = ['tr_loss', 'te_loss', ' '])

def save_train_multi_loss(model_path,train_loss_dic, val_loss_dic):
	import os
	generate_folder(model_path)
# 	f_loss_tr=model_path+'/loss_tr.txt'
# 	f_loss_te=model_path+'/loss_te.txt'
# 	np.savetxt(f_loss_tr,loss_tr);
# 	np.savetxt(f_loss_te,loss_te);
	pkl_fn = os.path.join(model_path, 'training.pkl')
	save_pickles(pkl_fn,train_loss_dic,val_loss_dic, [], keys = ['tr_loss', 'te_loss', ' '])

def save_model(model,model_name):
	generate_folder(model_name)
	model_path=model_name+'/model.h5'
	model.save(model_path)

def save_model_epoch_idx(model,model_name,epoch_idx):
	generate_folder(model_name)
	# serialize model to YAML
	model_yaml = model.to_yaml()
	model_path = model_name+"/model.yaml"
	if not os.path.exists(model_path):
		with open(model_path, "w") as yaml_file:
			yaml_file.write(model_yaml)
	# serialize weights to HDF5
	model.save_weights(model_name+"/weights_{}.h5".format(epoch_idx))
# 	model_path=model_name+'/model-{}.h5'.format(epoch_idx)
# 	model.save(model_path)

# plot and save the file
# def save_val_results(model_name,image,den, est, indx):
# 	import os
# 	result_folder = os.path.join(model_name, 'val_results')
# 	generate_folder(result_folder)
# 	file_path = os.path.join(result_folder,'val_{}.png'.format(indx))
# 	from matplotlib.backends.backend_agg import FigureCanvasAgg
# 	from matplotlib.figure import Figure
# 	fig = Figure(figsize=(10,3))
# 	fig = Figure()
# 	ax = fig.add_subplot(1,3,1)
# 	ax.imshow(image)
# 	bx = fig.add_subplot(1,3,2)
# 	bx.imshow(den)
# 	cx = fig.add_subplot(1,3,3)
# 	cx.imshow(est)
# 	canvas = FigureCanvasAgg(fig)
# 	canvas.print_figure(file_path, dpi=100)

def plot_subplot(fig, image, den, est, rows, cols, row_idx):
	ax = fig.add_subplot(rows,cols,1+row_idx*cols)
	ax.imshow(image)
	bx = fig.add_subplot(rows,cols,2+row_idx*cols)
	bx.imshow(den)
	bx.set_xlabel('{0:.2f}'.format(np.sum(den)))
	cx = fig.add_subplot(rows,cols,3+row_idx*cols)
	cx.set_xlabel('{0:.2f}'.format(np.sum(est)))
	cx.imshow(est)
	if row_idx == 0:
		ax.set_title('cell image')
		bx.set_title('GT')
		cx.set_title('estimation')
	
def save_val_results(model_name, fig_size, image_arr,den_arr, est_arr, indx):
	import os
	result_folder = os.path.join(model_name, 'val_results')
	generate_folder(result_folder)
	file_path = os.path.join(result_folder,'val_{}.png'.format(indx))
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	if not fig_size:
		fig_size = (9,8)
	fig = Figure(figsize=fig_size)
	for i in range(image_arr.shape[0]):
		image = image_arr[i,:]
		if not np.sum(image) == 0:
			image = (image -np.min(image))/(np.max(image)-np.min(image))
		plot_subplot(fig, image, den_arr[i,:], est_arr[i,:], 3,3,i)
	fig.tight_layout()
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_path, dpi=100)

## given the batch and hImageSize, nb_sampling, generate the patches
def patch_density_gen(batch, hImgSize, nb_sampling=100, test_flag=False):
	import numpy as np
	import random
	imglist = []
	densitylist = []
	shp = batch.shape
	for i in range(batch.shape[0]):
		data = np.copy(batch[i,:])
		index_tuples=np.nonzero(batch[i,:,:,0])
		nb_pixels=index_tuples[0].shape[0]
		if test_flag==True:
			idx_set=range(nb_pixels)
		else:
			idx_set=random.sample(range(nb_pixels),nb_sampling)
		for cidx in idx_set:
			xid=index_tuples[0][cidx]
			yid=index_tuples[1][cidx]
			xlIdx=max(0,xid-hImgSize)
			xrIdx=min(shp[1]-1,xid+hImgSize)
			ylIdx=max(0,yid-hImgSize)
			yrIdx=min(shp[2]-1,yid+hImgSize)
			ax=xlIdx+hImgSize-xid
			bx=xid+hImgSize-xrIdx
			ay=ylIdx+hImgSize-yid
			by=yid+hImgSize-yrIdx
			extr_image=data[xlIdx:xrIdx,ylIdx:yrIdx,:]
			image=np.pad(extr_image,((ax,bx),(ay,by),(0,0)),'constant')
			## normalize the each patch
# 			image = (image-np.mean(image))/np.std(image)
# 			im = image[:,:,0]
# 			im = (im-np.mean(im))/np.std(im)
			if shp[-1]<3:
				imglist.append(image[:,:,0].reshape(image.shape[:2]+(1,)))
				densitylist.append(image[:,:,1].reshape(image.shape[:2]+(1,)))
			else:
				imglist.append(image[:,:,:3])	
				densitylist.append(image[:,:,-1].reshape(image.shape[:2]+(1,)))
# 			imglist.append(np.reshape(im,(image.shape[0],image.shape[1],1)))
# 			densitylist.append(np.reshape(image[:,:,1],(image.shape[0],image.shape[1],1)))
	return imglist, densitylist

def reduced_density_maps(density_arr):
	import skimage.measure
	shp = density_arr.shape
	reduce2_list = []
	reduce4_list = []
	reduce8_list = []
	for i in range(shp[0]):
		den_map = np.squeeze(density_arr[i])
		reduce2 = skimage.measure.block_reduce(den_map, (2,2), np.sum)
		reduce4 = skimage.measure.block_reduce(den_map, (4,4), np.sum)
		reduce8 = skimage.measure.block_reduce(den_map, (8,8), np.sum)
		reduce2_list.append(reduce2)
		reduce4_list.append(reduce4)
		reduce8_list.append(reduce8)
	reduce2_arr = np.array(reduce2_list)
	reduce4_arr = np.array(reduce4_list)
	reduce8_arr = np.array(reduce8_list)
	reduce2_arr = reduce2_arr.reshape(reduce2_arr.shape +(1,))
	reduce4_arr = reduce4_arr.reshape(reduce4_arr.shape +(1,))
	reduce8_arr = reduce8_arr.reshape(reduce8_arr.shape +(1,))
	return reduce2_arr, reduce4_arr, reduce8_arr

# 02-26-2019
def multiscale_ground_truth(density_arr):
	import skimage.measure as measure
	from skimage.transform import rescale
	shp = density_arr.shape
	Y4_list = []
	Y3_list = []
	Y2_list = []
	Y1_list = []
	for i in range(shp[0]):
		den_map = np.squeeze(density_arr[i])
	# 		reduce2 = skimage.measure.block_reduce(den_map, (2,2), np.sum)
	# 		reduce4 = skimage.measure.block_reduce(den_map, (4,4), np.sum)
	# 		reduce8 = skimage.measure.block_reduce(den_map, (8,8), np.sum)
		Y4 = den_map - rescale(measure.block_reduce(den_map, (2,2), np.mean), 2.0)
		Y3 = measure.block_reduce(den_map, (2,2), np.mean) - rescale(measure.block_reduce(den_map, (4,4), np.mean), 2.0)
		Y2 = measure.block_reduce(den_map, (4,4), np.mean) - rescale(measure.block_reduce(den_map, (8,8), np.mean), 2.0)
		Y1 = measure.block_reduce(den_map, (8,8), np.mean)
		Y4_list.append(Y4)
		Y3_list.append(Y3)
		Y2_list.append(Y2)
		Y1_list.append(Y1)
	Y4_arr = np.array(Y4_list)
	Y3_arr = np.array(Y3_list)
	Y2_arr = np.array(Y2_list)
	Y1_arr = np.array(Y1_list)
	Y4_arr = Y4_arr.reshape(Y4_arr.shape +(1,))
	Y3_arr = Y3_arr.reshape(Y3_arr.shape +(1,))
	Y2_arr = Y2_arr.reshape(Y2_arr.shape +(1,))
	Y1_arr = Y1_arr.reshape(Y1_arr.shape +(1,))
	return Y4_arr, Y3_arr, Y2_arr, Y1_arr

## training with control of whether to compute the mean of absolute error
def train_model_multi_task(model, X_train, X_val, Y_train, Y_val, nb_epochs=400, nb_epoch_per_record=1, input_shape=(100,100,1), batch_size =256, is_mae = False, lr_max = None):
	import random
	keras.losses.mse_ct_err = mse_ct_err
	if len(X_train.shape)<3:
		X_train = X_train.reshape(X_train.shape + (1,))
	Y_train = Y_train.reshape(Y_train.shape + (1,))
	Images=np.concatenate([X_train,Y_train,Y_train],axis=3)

	if len(X_train.shape)<3:
		X_val = X_val.reshape(X_val.shape + (1,))
	Y_val = Y_val.reshape(Y_val.shape + (1,))
	val_Images=np.concatenate([X_val,Y_val, Y_val],axis=3)
	
	## train data generator
	train_datagen = ImageDataGenerator(
	horizontal_flip = True,
	vertical_flip =True,
	fill_mode = "constant",
	cval=0,
	zoom_range = 0.0,
	width_shift_range = 0.1,
	height_shift_range= 0.1,
	rotation_range=40)
	train_datagen.fit(Images)
	train_gen=train_datagen.flow(Images,None,batch_size=Images.shape[0])

	## validation data generator
	val_datagen = ImageDataGenerator(
	horizontal_flip = True,
	vertical_flip =True,
	fill_mode = "constant",
	cval=0,
	zoom_range = 0.0,
	width_shift_range = 0.1,
	height_shift_range=0.1,
	rotation_range=40)
	val_datagen.fit(val_Images)
	val_gen=val_datagen.flow(val_Images,None,batch_size=val_Images.shape[0])

	# training
	nb_sampling=100
	nb_val_sampling=100
	epochIdx=0
	train_loss_dic = {'loss':[],'ori':[], 'red2':[], 'red4':[], 'red8':[]}
	val_loss_dic = {'loss':[],'ori':[], 'red2':[], 'red4':[], 'red8':[]}
# 	tr_loss=[]
# 	te_loss=[]
# 	tr_red2_loss=[]
# 	te_red2_loss=[]
# 	tr_red4_loss=[]
# 	te_red4_loss=[]
# 	tr_red8_loss=[]
# 	te_red8_loss=[]
# 	tr_ori_loss=[]
# 	te_ori_loss=[]
	tr_acc =[]
	te_acc =[]
	mae_list=[]
	std_list=[]
	bs_acc =0.0
	bs_mse =float('Inf')
	red1_mse = float('Inf')
	red2_mse = float('Inf')
	red4_mse = float('Inf')
	red8_mse = float('Inf')
# 	print(input_shape[0])
	hImgSize=floor(input_shape[0]/2)
	print(hImgSize)
	time_end = time.time()
	while(epochIdx<nb_epochs):
		epochIdx+=1
		prev_time = time_end
		# update learning rate
		if lr_max:
			if epochIdx<= 50:
				lr = lr_max
			elif epochIdx <= 100:
				lr = lr_max/2
			else:
				lr = lr_max/4
			K.set_value(model.optimizer.lr, lr)
# 		if epochIdx > 50:
# 			K.set_value(model.optimizer.loss,'mse_ct_err')
		batch=train_gen.next()
		imglist, densitylist= patch_density_gen(batch, hImgSize, nb_sampling=nb_sampling)
		trainImages=np.array(imglist)
		imglist=None
		densityMaps = np.array(densitylist)
# 		print(densityMaps.shape)
# 		print(np.max(densityMaps))
# 		red2, red4, red8 = reduced_density_maps(densityMaps)
		densityMaps = densityMaps/100
		Y4tr, Y3tr, Y2tr, Y1tr = multiscale_ground_truth(densityMaps)
# 		print(Y4tr.shape, Y3tr.shape, Y2tr.shape, Y1tr.shape)
		densitylist =None
		hist = model.fit(trainImages,[Y1tr*100, Y2tr*100, Y3tr*100, Y4tr*100],batch_size=batch_size, nb_epoch=nb_epoch_per_record, verbose=1, shuffle=True)
# 		hist=model.fit(trainImages,densityMaps,batch_size=batch_size, nb_epoch=nb_epoch_per_record, verbose=1, shuffle=True)
		trainImages=None
		densityMaps = None
		## evaluation
		val_batch=val_gen.next()
		val_imglist, val_densitylist=patch_density_gen(val_batch, hImgSize, nb_sampling=nb_val_sampling)
		valImages=np.array(val_imglist)
		valDensityMaps = np.array(val_densitylist)/100
# 		val_red2, val_red4, val_red8 = reduced_density_maps(valDensityMaps)
		Y4val, Y3val, Y2val, Y1val = multiscale_ground_truth(valDensityMaps)
		val_imglist = None
		val_densitylist = None
		score=model.evaluate(valImages,[Y1val*100, Y2val*100, Y3val*100, Y4val*100],verbose=1,batch_size=int(batch_size))
# 		tr_loss.append(hist.history['loss'][-1])
# 		tr_loss.append(hist.history['loss'][-1])
# 		te_loss.append(score)
# 		print(score)
# 		print(hist.history['loss'])
# 		print(hist.history)
# 		tr_loss.append()
		train_loss_dic['loss'].append(hist.history['loss'][-1])
		train_loss_dic['ori'].append(hist.history['original_loss'][-1])
		train_loss_dic['red2'].append(hist.history['red2_loss'][-1])
		train_loss_dic['red4'].append(hist.history['red4_loss'][-1])
		train_loss_dic['red8'].append(hist.history['red8_loss'][-1])
		val_loss_dic['loss'].append(score[0])
		val_loss_dic['ori'].append(score[4])
		val_loss_dic['red2'].append(score[3])
		val_loss_dic['red4'].append(score[2])
		val_loss_dic['red8'].append(score[1])
# 		print('\nepoch:{0}=> tr mse:{1:.6f}, val mse:{2:.6f}, lr:{3:.4f}'.format(epochIdx,train_loss_dic['loss'][-1],val_loss_dic['loss'][-1],K.get_value(model.optimizer.lr)))
# 		print('\nepoch '+str(epochIdx)+'-> tr mse:'+str(tr_loss[-1])+', val mse:'+str(te_loss[-1])+'--')
		plot_multi_loss(model.name, train_loss_dic, val_loss_dic)
		save_train_multi_loss(model.name, train_loss_dic, val_loss_dic)
# 		print(score[0])
		if red1_mse>score[4]:
			red1_mse = score[4]
			# save model weights
			save_model_epoch_idx(model,model.name,epochIdx)	
# 		if (bs_mse>score[0])or(red1_mse>score[4])or(red2_mse>score[3])or(red4_mse>score[2]) or (red8_mse>score[1]):
# 			if bs_mse >score[0]:
# 				bs_mse = score[0]
# 			if red1_mse>score[4]:
# 				red1_mse = score[4]
# 			if red2_mse>score[3]:
# 				red2_mse = score[3]
# 			if red4_mse>score[2]:
# 				red4_mse = score[2]
# 			if red8_mse>score[1]:
# 				red8_mse = score[1]
# 			# save model weights
# 			save_model_epoch_idx(model,model.name,epochIdx)
			# save the validation results
			img_idx=random.randint(0,valImages.shape[0]-1)
			img_idx=random.sample(range(valImages.shape[0]),3)
			image_arr = np.squeeze(valImages[img_idx,:])
			den_arr = np.squeeze(valDensityMaps[img_idx,:])/100
			est_arr = model.predict(valImages[img_idx,:])[-1]/100
			save_val_results(model.name,(8,8),image_arr,den_arr, np.squeeze(est_arr), epochIdx)
# 			bs_mse=score[0]
		# make prediction
# 		if is_mae:
# 			mean_error =0
# 			std_error =0
# 			estimate_ls = []
# 			real_count_ls =[]
# 			batch_idx = 0
# 			valImages.shape
# 			while(True):
# 				if ((batch_idx+1)*batch_size<=valDensityMaps.shape[0]):
# 					image_arr = valImages[batch_idx*batch_size:(batch_idx+1)*batch_size,:]
# 					density_arr = valDensityMaps[batch_idx*batch_size:(batch_idx+1)*batch_size,:]
# 				elif ((batch_idx+1)*batch_size>valDensityMaps.shape[0] and batch_idx*batch_size<valDensityMaps.shape[0]):
# 					image_arr = valImages[batch_idx*batch_size:,:]
# 					density_arr = valDensityMaps[batch_idx*batch_size:,:]
# 				else:
# 					break
# 				preds = model.predict(image_arr)
# 				preds = preds.reshape(-1,preds.shape[1],preds.shape[2])
# 				preds = preds/100
# 				density_arr = density_arr/100
# 				pred_counts = np.apply_over_axes(np.sum,preds,[1,2]).reshape(preds.shape[0])
# 				ground_counts = np.apply_over_axes(np.sum,density_arr,[1,2]).reshape(preds.shape[0])
# 				estimate_ls +=pred_counts.tolist()
# 				real_count_ls +=ground_counts.tolist()	
# 				batch_idx += 1
# 			count_arr = np.array(estimate_ls)
# 			real_count_arr =np.array(real_count_ls)
# 			mean_error = np.mean(np.abs(count_arr-real_count_arr))
# 			std_error = np.std(np.abs(count_arr-real_count_arr))
# 			mae_list.append(mean_error)
# 			std_list.append(std_error)
# 			plot_mae(model.name, mae_list, std_list)
# 			print('epoch '+str(epochIdx)+'-> val mae:'+str(mae_list[-1])+', val std ae:'+str(std_list[-1])+'--')
		valImages=None
		valDensityMaps =None
		time_end = time.time()
		epoch_time = time_end - prev_time
		print('\nepoch:{0}=> tr mse:{1:.6f}, val mse:{2:.6f}, lr:{3:.4f}, epoch_duration:{4:.2f}'.format(epochIdx,train_loss_dic['loss'][-1],val_loss_dic['loss'][-1],K.get_value(model.optimizer.lr),epoch_time))

## training with control of whether to compute the mean of absolute error
def train_model(model, X_train, X_val, Y_train, Y_val, nb_epochs=400, nb_epoch_per_record=1, input_shape=(100,100,1), batch_size =256, is_mae = False, lr_max = None, method = 'density'):
	import random
	keras.losses.mse_ct_err = mse_ct_err
	if len(X_train.shape)<3:
		X_train = X_train.reshape(X_train.shape + (1,))
	Y_train = Y_train.reshape(Y_train.shape + (1,))
	Images=np.concatenate([X_train,Y_train,Y_train],axis=3)

	if len(X_train.shape)<3:
		X_val = X_val.reshape(X_val.shape + (1,))
	Y_val = Y_val.reshape(Y_val.shape + (1,))
	val_Images=np.concatenate([X_val,Y_val, Y_val],axis=3)
	
	## train data generator
	train_datagen = ImageDataGenerator(
	horizontal_flip = True,
	vertical_flip =True,
	fill_mode = "constant",
	cval=0,
	zoom_range = 0.0,
	width_shift_range = 0.1,
	height_shift_range= 0.1,
	rotation_range=40)
	train_datagen.fit(Images)
	train_gen=train_datagen.flow(Images,None,batch_size=Images.shape[0])

	## validation data generator
	val_datagen = ImageDataGenerator(
	horizontal_flip = True,
	vertical_flip =True,
	fill_mode = "constant",
	cval=0,
	zoom_range = 0.0,
	width_shift_range = 0.1,
	height_shift_range=0.1,
	rotation_range=40)
	val_datagen.fit(val_Images)
	val_gen=val_datagen.flow(val_Images,None,batch_size=val_Images.shape[0])

	# training
	nb_sampling=100
	nb_val_sampling=100
	epochIdx=0
	tr_loss=[]
	te_loss=[]
	tr_acc =[]
	te_acc =[]
	mae_list=[]
	std_list=[]
	bs_acc =0.0
	bs_mse =float('Inf')
# 	print(input_shape[0])
	hImgSize=floor(input_shape[0]/2)
	print(hImgSize)
	time_end = time.time()
	while(epochIdx<nb_epochs):
		epochIdx+=1
		prev_time = time_end
		# update learning rate
# 		if lr_max:
# 			if epochIdx% 16 == 0:
# 				lr = lr_max
# 			elif epochIdx% 16 == 1:
# 				lr = lr_max/5
# 			else:
# 				lr = lr_max/10*0.75
# 			K.set_value(model.optimizer.lr, lr)
# 		if epochIdx > 50:
# 			K.set_value(model.optimizer.loss,'mse_ct_err')
		batch=train_gen.next()
		imglist, densitylist= patch_density_gen(batch, hImgSize, nb_sampling=nb_sampling)
		trainImages=np.array(imglist)
		imglist=None
		densityMaps = np.array(densitylist)
		densitylist =None
		## preprocess
		if method == 'count_ception':
			trainImages, densityMaps = data_trans(trainImages, densityMaps)
			densityMaps = densityMaps.reshape(np.squeeze(densityMaps).shape + (1,))
		hist=model.fit(trainImages,densityMaps,batch_size=batch_size, nb_epoch=nb_epoch_per_record, verbose=1, shuffle=True)
		trainImages=None
		densityMaps = None
		## evaluation
		val_batch=val_gen.next()
		val_imglist, val_densitylist=patch_density_gen(val_batch, hImgSize, nb_sampling=nb_val_sampling)
		valImages=np.array(val_imglist)
		valDensityMaps = np.array(val_densitylist)
		val_imglist = None
		val_densitylist = None
		if method == 'count_ception':
			valImages, valDensityMaps = data_trans(valImages, valDensityMaps)
			valDensityMaps = valDensityMaps.reshape(np.squeeze(valDensityMaps).shape + (1,))
		score=model.evaluate(valImages,valDensityMaps,verbose=1,batch_size=int(batch_size))
		tr_loss.append(hist.history['loss'][-1])
		te_loss.append(score)
		print('\nepoch:{0}=> tr mse:{1:.6f}, val mse:{2:.6f}, lr:{3:.4f}'.format(epochIdx,tr_loss[-1],te_loss[-1],K.get_value(model.optimizer.lr)))
# 		print('\nepoch '+str(epochIdx)+'-> tr mse:'+str(tr_loss[-1])+', val mse:'+str(te_loss[-1])+'--')
		plot_loss(model.name, tr_loss, te_loss)
		save_train_loss(model.name, tr_loss, te_loss)
		if bs_mse>score:
			# save model weights
			save_model_epoch_idx(model,model.name,epochIdx)
			# save the validation results
# 			img_idx=random.randint(0,valImages.shape[0]-1)
			img_idx=random.sample(range(valImages.shape[0]),3)
			image_arr = np.squeeze(valImages[img_idx,:])
			den_arr = np.squeeze(valDensityMaps[img_idx,:])/100
			est_arr = model.predict(valImages[img_idx,:])/100
			save_val_results(model.name,(8,8),image_arr,den_arr, np.squeeze(est_arr), epochIdx)
# 			save_val_results(model.name,image,gt,np.squeeze(est),epochIdx)
			bs_mse=score
		valImages=None
		valDensityMaps =None
		time_end = time.time()
		epoch_time = time_end - prev_time
		print('\nepoch:{0}=> tr mse:{1:.6f}, val mse:{2:.6f}, lr:{3:.4f}, epoch_duration:{4:.2f}'.format(epochIdx,tr_loss[-1],te_loss[-1],K.get_value(model.optimizer.lr),epoch_time))

## prepare data for buildModel_Count_ception model
## input_size: mxm
## output_size: data: (m+31*2)x(m+31*2), count map: (m+31)x(m+31)
def data_trans(image_arr, annot_arr):
	if len(np.squeeze(annot_arr).shape)>2:
		shp = image_arr.shape
		pad_widths = []
		for i in range(len(shp)):
			if i == 1 or i == 2:
				pad_widths.append((31,31))
			else:
				pad_widths.append((0,0))
		width_tuple = tuple(pad_widths)
		image_arr = np.pad(image_arr, width_tuple, 'constant', constant_values= 0)
		kernel = np.ones((1,32,32))
		map_arr = sg.convolve(np.squeeze(annot_arr),kernel)
# 		print(map_arr.shape)
# 		map_arr = map_arr.reshape(annot_arr.shape)
	else:
		shp = image_arr.shape
		pad_widths = []
		for i in range(len(shp)):
			if i == 0 or i == 1:
				pad_widths.append((31,31))
			else:
				pad_widths.append((0,0))
		width_tuple = tuple(pad_widths)
		image_arr = np.pad(image_arr, width_tuple, 'constant', constant_values= 0)
		kernel = np.ones((32,32))
		map_arr = sg.convolve(np.squeeze(annot_arr),kernel)
# 		map_arr = map_arr.reshape(annot_arr.shape)
	return image_arr, map_arr