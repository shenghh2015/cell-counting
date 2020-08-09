# analyze the cell counting results 6.8.2018 for ACCV2018 paper (give up the submission)
# analyze the MAE, MRE, MSE and SSIM for different methods
import numpy as np
import glob
import os
import re
import csv

from natsort import natsorted
from utils.metrics import *
from utils.file_load_save import *
from utils.plot_function import *
from utils.data_processing import *
from models import *
from data_load import *



import matplotlib.pyplot as plt

# save the figure to png file
def save_plot_figure(fig, file_name, dpi_nb = 100):
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name+'.png', dpi=dpi_nb)

## parse model name and get
## 5.31/2018
def parse_model_name(model_folder, network_key = 'buildModel', dvkey = 'v-', crosskey = 'cross-'):
	import re
	import os
	x = re.search(dvkey+'\d+', model_folder)
	dat_ver = x.group(0).split('-')[-1]
	dat_ver_int = int(float(dat_ver))
	y = re.search(crosskey+'\d+',model_folder)
	cross_nb = y.group(0).split('-')[-1]
	cross_nb_int = int(float(cross_nb))
	z = re.search(network_key+'\w+', model_folder)
	result_folder_str = os.path.split(model_folder[:z.end()])[1]
	network = z.group(0)
	return network, dat_ver_int, cross_nb_int, result_folder_str


# def cell_count_evaluation(model_folder):
# 	# map model_folder to the result folder
# 	network, dat_ver_int, cross_nb_int, result_folder_str = parse_model_name(model_folder)
# 	results_root_folder = os.path.expanduser('~/dl-cells/dlovl-framework/results')
# 	result_folder = os.path.join(results_root_folder, result_folder_str)

def mean_square_err_for_arr(den_arr, den_arr1):
	from sklearn.metrics import mean_squared_error as MSE
	shp = den_arr.shape
	mse_list = []
	for i in range(shp[0]):
		mse_list.append(MSE(den_arr[i,:].flatten(), den_arr1[i,:].flatten()))
	return np.array(mse_list)

def cosine_for_arr(den_arr, den_arr1):
	cosin_list, _ = density_Cosine_calculate_for_array(den_arr, den_arr1)
	return np.array(cosin_list)

def cosine_calculate(den, den1):
	return density_Cosine_calculate(den, den1)

# the co-registration between the gt_arr and est_arr
def coregistr_gt_est(gt_arr, est_arr):
	if len(gt_arr.shape) == 2:
		gt_arr = gt_arr.reshape((1,)+gt_arr.shape)
		est_arr = est_arr.reshape((1,)+est_arr.shape)
	shp = gt_arr.shape
	est_shp = est_arr.shape
	co_est = np.zeros(shp)
	h_dif = abs(shp[1] - est_shp[1])
	w_dif = abs(shp[2] - est_shp[2])
	co_est = est_arr[:,int(h_dif/2):est_shp[1]-int(h_dif/2),int(w_dif/2):est_shp[2]-int(w_dif/2)]
	co_est = np.squeeze(co_est)
	return co_est
		

import os
import glob

# previous experiment results analysis for ACCV2018
def cell_count_evaluation():
	cross = 0
	results_root_folder = os.path.expanduser('~/dl-cells/dlct-framework/results/ACCV2018_v1_6.6-7.6')
	result_folders = []
	results_ptrn = os.path.join(results_root_folder, 'build*')
	result_folders = glob.glob(results_ptrn)
	image_for_methods = []
	gt_for_methods = []
	est_for_methods = []
	std_for_methods = []
	runtime_for_methods = []
	mae_for_methods = []
	mre_for_methods = []
	mean_mse_for_methods = []
	mcosine_for_methods = []
	method_pool = []
	for result_folder in result_folders:
# 		result_folder = result_folders[3]
		result_version_ptrn = os.path.join(result_folder,'data_version-*')
		result_version_folders = glob.glob(result_version_ptrn)
		result_version_folders = natsorted(result_version_folders)
		method_pool.append(os.path.basename(result_folder))
		# data pool
		image_pool = []   # data from different methods
		gt_pool = []
		est_pool = []
		# metrics pool
		abs_err_pool = []
		re_err_pool = []
		cosine_pool = []
		mse_pool = []
		runtime_pool = []
	
		for result_version in result_version_folders:
# 			result_version = result_version_folders[0]
			if cross == 0:
				result_cross_folders = glob.glob(os.path.join(result_version, 'cross-{}'.format(cross)))
			else:
				result_cross_folders = glob.glob(os.path.join(result_version, 'cross*'))
			### for each type of cells
			# data: image, ground truth map, and estimated density map
			image_list = []
			gt_list = []
			est_list = []
			runtime_list = []
			# metrics of performance
			# divide 		
			for result_cross in result_cross_folders:
	# 				result_cross = result_cross_folders[0]
				result_pkl_ptrn = os.path.join(result_cross,'*.pkl')
				pkl_files = glob.glob(result_pkl_ptrn)
				for pkl_file in pkl_files:
					results = read_any_pickle(pkl_file, keys = ['est_den', 'gt_den', 'ori_img', 'run_time'])
					est_list.append(results[0])
					gt_list.append(results[1])
					image_list.append(results[2])
					runtime_list.append(results[3])
			image_arr = np.array(image_list)
			gt_arr = np.array(gt_list)
			est_arr = np.array(est_list)
			est_arr = coregistr_gt_est(gt_arr, est_arr)
			runtime_arr = np.array(runtime_list)
			image_arr = coregistr_gt_est(gt_arr, image_arr)
			# analyze the results for each type of cells
			mse_arr = mean_square_err_for_arr(gt_arr,est_arr)
			cosin_arr = cosine_for_arr(gt_arr, est_arr)
			gt_count_arr = np.squeeze(np.apply_over_axes(np.sum, gt_arr, [1,2]))
			est_count_arr = np.squeeze(np.apply_over_axes(np.sum, est_arr, [1,2]))
			abs_err_arr = np.abs(gt_count_arr-est_count_arr)
			re_err_err = np.abs(gt_count_arr-est_count_arr)/gt_count_arr
			print(np.mean(abs_err_arr))
		
			## data merge
			image_pool.append(image_arr)
			gt_pool.append(gt_arr)
			est_pool.append(est_arr)
			runtime_pool.append(runtime_arr)
# 			print('run time number:{0:}'.format(runtime_arr.shape))

			## metrics
			abs_err_pool.append(abs_err_arr)
			re_err_pool.append(re_err_err)
			mse_pool.append(mse_arr)
			cosine_pool.append(cosin_arr)
	
		## compute the total average
		mae_pool = []
		std_pool = []
		mre_pool = []
		mcosine_pool = []
		mean_mse_pool = []
		mean_runtime_pool = []
		for abs_arr in abs_err_pool:
			mae_pool.append(round(np.mean(abs_arr),5))
		for abs_arr in abs_err_pool:
			std_pool.append(round(np.std(abs_arr),5))
		for re_arr in re_err_pool:
			mre_pool.append(round(np.mean(re_arr),5))
		for cosine_arr in cosine_pool:
			mcosine_pool.append(round(np.mean(cosine_arr),5))
		for mse_arr in mse_pool:
			mean_mse_pool.append(np.mean(mse_arr))
		for runtime_arr in runtime_pool:
			mean_runtime_pool.append(round(np.mean(runtime_arr),5))

# 		total_abs_arr = np.concatenate(abs_err_pool)
# 		total_re_arr = np.concatenate(re_err_pool)
# 		total_cosine_arr = np.concatenate(cosine_pool)
# 		total_mse_arr = np.concatenate(mse_pool)
# 		average_mae = np.mean(total_abs_arr)
# 		average_re = np.mean(total_re_arr)
# 		average_cosine = np.mean(total_cosine_arr)
# 		average_mse = np.mean(total_mse_arr)
# 	
# 		# add average to each metric list: 31,32,33,34,35, average
# 		if len(result_version_folders)>1:
# 			mae_pool.append(average_mae)
# 			mre_pool.append(average_re)
# 			mcosine_pool.append(average_cosine)
# 			mean_mse_pool.append(average_mse)

		mae_for_methods.append(mae_pool)
		std_for_methods.append(std_pool)
		mre_for_methods.append(mre_pool)
		mcosine_for_methods.append(mcosine_pool)
		mean_mse_for_methods.append(mean_mse_pool)
		runtime_for_methods.append(mean_runtime_pool)

		# store the testing data
		image_for_methods.append(image_pool)
		gt_for_methods.append(gt_pool)
		est_for_methods.append(est_pool)
# 		image_for_methods.append(np.concatenate(image_pool, axis = 0))
# 		gt_for_methods.append(np.concatenate(gt_pool, axis = 0))
# 		est_for_methods.append(np.concatenate(est_pool, axis = 0))
	## store the metrics as a table
	table_pool = [mae_for_methods, mre_for_methods, mcosine_for_methods, mean_mse_for_methods]
	# for table in table_pool:
	table_dic = {}
	table_dic['MAE'] = mae_for_methods
	table_dic['STD'] = std_for_methods
	table_dic['MRE'] = mre_for_methods
	table_dic['SSIM'] = mcosine_for_methods
	table_dic['MSE'] = mean_mse_for_methods
	table_dic['RUN'] = runtime_for_methods
	file_basename = 'MAE_STD_MRE_SSIM_MSE_RUN'
# 	for date in range(len(date_pool)):
# 		file_basename += '-{}'.format(date_pool[date])
	csv_file_name = os.path.join(results_root_folder,file_basename)
	
	save_model_evaluation_cell_counting(csv_file_name, method_pool,table_dic)
	## store the results in a pickle file

## combine the evaluated image together
def save_model_evaluation_cell_counting(file_name, method_pool, dic_sim = {}):
	import csv
	base_headers = ['bacterial','MBM','H&E','hESC']
# 	methods = ['FCRN','FCRN-skip','Count-Inception','Our proposed']
	methods = method_pool
	file_name = '{}.csv'.format(file_name)
	if dic_sim == {}:
		return
	else:
		with open(file_name, 'w') as csvfile:
			while(dic_sim!={}):
				(key, values) = dic_sim.popitem()
				# create the header in the table
				fieldnames = [key]
				if(len(values[0]))==1:
					fieldnames += base_headers[-1:]
				else:
					fieldnames += base_headers
# 				if(len(values))<=2:
# 					methods = ['FCRN-skip','Our proposed']					
# 				else:
# 					methods = methods
				writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
				writer.writeheader()
				for i in range(len(values)):
					row_dic = {}			
					row_dic[fieldnames[0]] = methods[i]
					for j in range(len(values[i])):
						row_dic[fieldnames[j+1]] = values[i][j]
					writer.writerow(row_dic)

def plot_figure_for_paper(fig, image_for_methods, gt_for_methods, est_for_methods, data_idx = 0, idx = 0, font_size = 24, xlabel_font = 14):
	image = np.squeeze(image_for_methods[0][data_idx][idx])
	image = (image-np.min(image))/(np.max(image)-np.min(image))
	ax = fig.add_subplot(1,5,1)
	ax.imshow(image)
	ax.set_title('Cell image', fontweight="bold", size= font_size)
	bx = fig.add_subplot(1,5,2)
	gt_den = np.squeeze(gt_for_methods[0][data_idx][idx])
	bx.imshow(gt_den)
	bx.set_title('Ground truth', fontweight="bold", size=font_size)
	bx.set_xlabel('Count:{0:.2f}'.format(np.sum(gt_den)), fontweight="bold", size=xlabel_font)
	cx = fig.add_subplot(1,5,3)
	est_den = np.squeeze(est_for_methods[0][data_idx][idx])
	cx.imshow(est_den)
	cx.set_title('FCRN', fontweight="bold", size=font_size)
	cx.set_xlabel('Count:{0:.2f}'.format(np.sum(est_den)), fontweight="bold", size=xlabel_font)
	dx = fig.add_subplot(1,5,4)
	est_den = np.squeeze(est_for_methods[1][data_idx][idx])
	dx.imshow(est_den)
	dx.set_xlabel('Count:{0:.2f}'.format(np.sum(est_den)), fontweight="bold", size=xlabel_font)
	dx.set_title('MLFCN', fontweight="bold", size=font_size)
	# ex = fig.add_subplot(1,5,5)
	# gt_den = np.squeeze(gt_for_methods[3][data_idx][idx])
	# ex.imshow(gt_den)
	fx = fig.add_subplot(1,5,5)
	est_den = np.squeeze(est_for_methods[3][data_idx][idx])
	fx.imshow(est_den)
	fx.set_xlabel('Count:{0:.2f}'.format(np.sum(est_den)), fontweight="bold", size=xlabel_font)
	fx.set_title('Count-Ception', fontweight="bold", size=font_size)
	fig.tight_layout()
	# save the plot figure
	base_headers = ['bacterial','MBM','H&E','hESC']
	file_name = base_headers[data_idx]+'_est_result'
	save_plot_figure(fig, file_name, dpi_nb = 120)

## for SPIE 2019 oral presentation
def plot_figure_for_paper(fig, image_for_methods, gt_for_methods, est_for_methods, data_idx = 0, idx = 0, font_size = 24, xlabel_font = 14):
	image = np.squeeze(image_for_methods[0][data_idx][idx])
	image = (image-np.min(image))/(np.max(image)-np.min(image))
	ax = fig.add_subplot(2,3,1)
	ax.imshow(image)
	ax.set_title('Cell image', fontweight="bold", size= font_size)
	bx = fig.add_subplot(2,3,2)
	gt_den = np.squeeze(gt_for_methods[0][data_idx][idx])
	bx.imshow(gt_den)
	bx.set_title('Ground truth', fontweight="bold", size=font_size)
	bx.set_xlabel('Count:{0:.2f}'.format(np.sum(gt_den)), fontweight="bold", size=xlabel_font)
	cx = fig.add_subplot(2,3,3)
	est_den = np.squeeze(est_for_methods[0][data_idx][idx])
	cx.imshow(est_den)
	cx.set_title('FCRN', fontweight="bold", size=font_size)
	cx.set_xlabel('Count:{0:.2f}'.format(np.sum(est_den)), fontweight="bold", size=xlabel_font)
	dx = fig.add_subplot(2,3,4)
	est_den = np.squeeze(est_for_methods[1][data_idx][idx])
	dx.imshow(est_den)
	dx.set_xlabel('Count:{0:.2f}'.format(np.sum(est_den)), fontweight="bold", size=xlabel_font)
	dx.set_title('MLFCN', fontweight="bold", size=font_size)
	# ex = fig.add_subplot(1,5,5)
	# gt_den = np.squeeze(gt_for_methods[3][data_idx][idx])
	# ex.imshow(gt_den)
	fx = fig.add_subplot(1,5,5)
	est_den = np.squeeze(est_for_methods[3][data_idx][idx])
	fx.imshow(est_den)
	fx.set_xlabel('Count:{0:.2f}'.format(np.sum(est_den)), fontweight="bold", size=xlabel_font)
	fx.set_title('Count-Ception', fontweight="bold", size=font_size)
	fig.tight_layout()
	# save the plot figure
	base_headers = ['bacterial','MBM','H&E','hESC']
	file_name = base_headers[data_idx]+'_est_result'
	save_plot_figure(fig, file_name, dpi_nb = 120)

## visual plot for paper
def density_estimation_visualization():
	plt.ion()
	fig = plt.figure()
	plt.clf()
	# idx = 6   # the image index
	# data_idx = 0   # the dataset index
# 	idx = 10
# 	data_idx = 1
	# idx = 40
	# data_idx = 2
	idx = 7
	data_idx = 3
	font_size = 15
	xlabel_font = 14
	plot_figure_for_paper(fig, image_for_methods, gt_for_methods, est_for_methods, data_idx = data_idx, idx = idx, font_size = 14)

def figure_plot(fig, image_arr1, image_arr2, den_arr1, den_arr2, cosin_arr, pair_idx):
	fig.clf()
	i = pair_idx
	print('The {}-th figure.'.format(i))
	ax = fig.add_subplot(2,2,1)
	cax = ax.imshow(image_arr1[i,:,:])
	fig.colorbar(cax)
	ax.set_ylabel(cell_types[0])
	ax.set_title('Cell image')
	bx = fig.add_subplot(2,2,2)
	cbx = bx.imshow(den_arr1[i,:,:])
	fig.colorbar(cbx)
	bx.set_xlabel('cell count:{0:.2f}'.format(np.sum(den_arr1[i,:,:])))
	bx.set_title('Density map')
	cx = fig.add_subplot(2,2,3)
	ccx = cx.imshow(image_arr2[i,:,:])
	fig.colorbar(ccx)
	cx.set_ylabel(cell_types[1])
	dx = fig.add_subplot(2,2,4)
	cdx = dx.imshow(den_arr2[i,:,:])
	fig.colorbar(cdx)
	dx.set_xlabel('cell count:{0:.2f}\ncosine:{1:.2f}'.format(np.sum(den_arr2[i,:,:]), cosin_arr[i]))

if __name__ == '__main__':
	# analyze the cell counting accuracy
	cell_count_evaluation()