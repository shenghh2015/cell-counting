import numpy as np

def generate_folder(folder):
	import os
	if not os.path.exists(folder):
		os.system('mkdir -p {}'.format(folder))

# plot training and validation loss
def plot_history(file_name, history):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	rows, cols, size = 1,3,5
	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
	ax[0].plot(history.history['loss']);ax[0].plot(history.history['val_loss'])
	ax[0].set_ylabel('loss');ax[0].set_xlabel('epochs');ax[0].legend(['train','valid'])
	ax[1].plot(history.history['iou_score']);ax[1].plot(history.history['val_iou_score'])
	ax[1].set_ylabel('iou_score');ax[1].set_xlabel('epochs');ax[1].legend(['train','valid'])
	ax[2].plot(history.history['f1-score']);ax[2].plot(history.history['val_f1-score'])
	ax[2].set_ylabel('f1-score');ax[2].set_xlabel('epochs');ax[2].legend(['train','valid'])
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)

def plot_deeply_history(file_name, history):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	rows, cols, size = 1,3,5
	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
	ax[0].plot(history.history['softmax_loss']);ax[0].plot(history.history['val_softmax_loss'])
	ax[0].set_ylabel('loss');ax[0].set_xlabel('epochs');ax[0].legend(['train','valid'])
	ax[1].plot(history.history['softmax_iou_score']);ax[1].plot(history.history['val_softmax_iou_score'])
	ax[1].set_ylabel('iou_score');ax[1].set_xlabel('epochs');ax[1].legend(['train','valid'])
	ax[2].plot(history.history['softmax_f1-score']);ax[2].plot(history.history['val_softmax_f1-score'])
	ax[2].set_ylabel('f1-score');ax[2].set_xlabel('epochs');ax[2].legend(['train','valid'])
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)

# plot for phase -> fluo validation loss
def plot_history_flu(file_name, history):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	rows, cols, size = 1,2,4
	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
	ax[0].plot(history.history['loss']);ax[0].plot(history.history['val_loss'])
	ax[0].set_ylabel('MSE');ax[0].set_xlabel('epochs');ax[0].legend(['train','valid'])
	ax[1].plot(history.history['psnr']);ax[1].plot(history.history['val_psnr'])
	ax[1].set_ylabel('PSNR');ax[1].set_xlabel('epochs');ax[1].legend(['train','valid'])
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=80)

def plot_flu_prediction(file_name, images, gt_maps, pr_maps, nb_images, rand_seed = 3):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	import random
	seed = rand_seed #3
	random.seed(seed)
	font_size = 24
	indices = random.sample(range(gt_maps.shape[0]),nb_images)
	rows, cols, size = nb_images,4,5
	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
	for i in range(len(indices)):
		idx = indices[i]
		image, gt_map, pr_map = images[idx,:].squeeze(), gt_maps[idx,:].squeeze(), pr_maps[idx,:].squeeze()
		image = np.uint8((image-image.min())/(image.max()-image.min())*255)
		err_map = np.sum(np.abs(gt_map-pr_map),axis=-1)
		gt_map_rgb = np.zeros(image.shape,dtype=np.uint8); # gt_map_rgb[:,:,:-1]=np.uint8((gt_map-gt_map.min())/(gt_map.max()-gt_map.min())*255)
		gt_map_rgb[:,:,:-1]=np.uint8(gt_map*255)
		pr_map_rgb = np.zeros(image.shape,dtype=np.uint8); # pr_map_rgb[:,:,:-1]=np.uint8((pr_map-pr_map.min())/(pr_map.max()-pr_map.min())*255)
		pr_map_rgb[:,:,:-1]=np.uint8(pr_map*255)
		ax[i,0].imshow(image[::4,::4,:]); ax[i,1].imshow(gt_map_rgb[::4,::4,:]); 
		ax[i,2].imshow(pr_map_rgb[::4,::4,:]); ax[i,3].imshow(err_map[::4,::4], cmap='Blues')
		ax[i,0].set_xticks([]);ax[i,1].set_xticks([]);ax[i,2].set_xticks([]);ax[i,3].set_xticks([])
		ax[i,0].set_yticks([]);ax[i,1].set_yticks([]);ax[i,2].set_yticks([]);ax[i,3].set_yticks([])
		if i == 0:
			ax[i,0].set_title('Image',fontsize=font_size); ax[i,1].set_title('GT',fontsize=font_size); 
			ax[i,2].set_title('Pred',fontsize=font_size); ax[i,3].set_title('Err Map',fontsize=font_size); 
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=60)

def plot_map_prediction(file_name, images, gt_maps, pr_maps, nb_images):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	from random import sample
	font_size = 24
	indices = sample(range(gt_maps.shape[0]),nb_images)
	rows, cols, size = nb_images,3,5
	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
	for i in range(len(indices)):
		idx = indices[i]
		image, gt_map, pr_map = images[idx,:].squeeze(), gt_maps[idx,:].squeeze(), pr_maps[idx,:].squeeze()
		image = np.uint8((image-image.min())/(image.max()-image.min())*255)
		ax[i,0].imshow(image[::4,::4,:]); ax[i,1].imshow(gt_map[::4,::4,:]); ax[i,2].imshow(pr_map[::4,::4,:])
		if i == 0:
			ax[i,0].set_title('Image',fontsize=font_size); ax[i,1].set_title('GT',fontsize=font_size); 
			ax[i,2].set_title('Pred',fontsize=font_size)
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=80)

def plot_flu_hist(file_name, gt_maps, pr_maps, nb_images):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	from random import sample
	font_size = 24
	indices = sample(range(gt_maps.shape[0]),nb_images)
	rows, cols, size = nb_images,2,5
	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
	for i in range(len(indices)):
		idx = indices[i]
		gt_map, pr_map = gt_maps[idx,:].squeeze(), pr_maps[idx,:].squeeze()
		ax[i,0].hist(gt_map.flatten(), bins = 200);ax[i,1].hist(pr_map.flatten(), bins = 200);
		if i == 0:
			ax[i,0].set_title('GT',fontsize=font_size); ax[i,1].set_title('Pred',fontsize=font_size); 
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=80)

# calculate the IoU and dice scores
def iou_calculate(y_true, y_pred):
	# one hot encoding of predictions
	num_classes = y_pred.shape[-1]
	y_pred = np.array([np.argmax(y_pred, axis=-1)==i for i in range(num_classes)]).transpose(1,2,3,0)
	print(y_pred.shape)
	axes = (1,2) # W,H axes of each image
	intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
	union = np.sum(np.logical_or(y_pred, y_true), axis=axes)
	mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

	smooth = .00001
	iou_per_image_class = (intersection + smooth) / (union + smooth)
	dice_per_image_class = (2 * intersection + smooth)/(mask_sum + smooth)

	mean_iou_over_images = np.mean(iou_per_image_class, axis = 0)
	mean_iou_over_images_class = np.mean(mean_iou_over_images)
	dice_class = np.mean(dice_per_image_class, axis = 0)
	mean_dice = np.mean(dice_per_image_class)

	return mean_iou_over_images, mean_iou_over_images_class, dice_class, mean_dice

def precision(label, confusion_matrix):
	col = confusion_matrix[:, label]
	return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
	row = confusion_matrix[label, :]
	return confusion_matrix[label, label] / row.sum()

def f1_score(label, confusion_matrix):
	row = confusion_matrix[label, :]; col = confusion_matrix[:, label]
	prec_score = confusion_matrix[label, label] / col.sum()
	recall_score = confusion_matrix[label, label] / row.sum()
	return 2*prec_score*recall_score/(prec_score + recall_score)

def psnr_score(img1, img2):
	import math
	import numpy as np
	img1 = img1.astype(np.float64)
	img2 = img2.astype(np.float64)
	mse = np.mean((img1 - img2)**2)
	if mse == 0:
		return float('inf')
	return 20 * math.log10(1.0 / math.sqrt(mse))

def calculate_psnr(imgs1, imgs2):
	if len(imgs1.shape) == 4:
		axis_tuple = (1,2,3)
	else:
		axis_tuple = (1,2)
	print('value in map: max {}, min {}'.format(imgs1.max(),imgs1.min()))
	mse = np.mean((imgs1-imgs2)**2, axis = axis_tuple)
	psnr_scores = 20*np.log10(1.0/np.sqrt(mse))
	return np.mean(psnr_scores), psnr_scores

def plot_psnr_histogram(file_name, psnr_list1, psnr_list2, rho_list1, rho_list2):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	kwargs = dict(alpha=0.7, bins=12, density= False, stacked=True)
	rows, cols, size = 1,2,6; font_size = 20
	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
	ax[0].hist(psnr_list1, **kwargs, color='r', label='fl1') # PSNR for channel 1
	ax[0].hist(psnr_list2, **kwargs, color='g', label='fl2') # PSNR for channel 2
	ax[0].legend(['fl1_average: {:.2f}'.format(np.mean(psnr_list1)), 'fl2_average: {:.2f}'.format(np.mean(psnr_list2))],fontsize=font_size-2)
	ax[0].set_title('PSNR distribution', fontsize =font_size)
	ax[0].set_ylabel('Count', fontsize =font_size);ax[0].set_xlabel('PSNR', fontsize =font_size);
	ax[0].set_xlim([np.min(np.concatenate([psnr_list1,psnr_list2]))-5, np.max(np.concatenate([psnr_list1,psnr_list2]))])	
# 	ax[0].set_xlim([np.min(np.concatenate([psnr_list1,psnr_list2])), np.max(np.concatenate([psnr_list1,psnr_list2]))])
	ax[0].tick_params(axis='x', labelsize=font_size-2); ax[0].tick_params(axis='y', labelsize=font_size-2)
# 	kwargs = dict(alpha=0.9, bins=10, density= False, stacked=True)
	ax[1].hist(rho_list1, **kwargs, color='r', label='fl1') # PSNR for channel 1
	ax[1].hist(rho_list2, **kwargs, color='g', label='fl2') # PSNR for channel 2
	ax[1].set_title(r'$\rho$ distribution', fontsize =font_size)
	ax[1].set_ylabel('Count', fontsize =font_size);ax[1].set_xlabel(r'$\rho$', fontsize =font_size);
	ax[1].tick_params(axis='x', labelsize=font_size-2); ax[1].tick_params(axis='y', labelsize=font_size-2)
	ax[1].set_xlim([0,1.0])
	ax[1].legend(['fl1_average: {:.4f}'.format(np.mean(rho_list1)), 'fl2_average: {:.4f}'.format(np.mean(rho_list2))],fontsize=font_size-2)
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=80)

SMOOTH= 1e-6; seed = 0
def calculate_pearsonr(imgs1, imgs2):
	from scipy import stats
	import numpy as np
	scores = []
	SMOOTH_ = np.random.random(imgs1[0,:].flatten().shape)*SMOOTH
	for i in range(imgs1.shape[0]):
		flat1 = imgs1[i,:].flatten(); flat2 = imgs2[i,:].flatten()
		flat1 = SMOOTH_ + flat1; flat2 = SMOOTH_ + flat2
		score, _= stats.pearsonr(flat1, flat2)
		scores.append(score)
	return np.mean(scores), scores


