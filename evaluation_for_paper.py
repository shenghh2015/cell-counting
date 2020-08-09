## to analyze the performance in term of MAE, MRE, MSE and SSIM

from keras.optimizers import SGD
import tensorflow as tf
import keras.backend as K

import os
import helper_functions as hf
from models import *
from models_comp import *
from data_load import *
from utils.loss_fns import *
from utils.file_load_save import *
from utils.data_processing import *
import os
# import helper_functions_v2 as hf
# from models_v2 import *
import time
import matplotlib.pyplot as plt

import os
import glob
import re
from natsort import natsorted
import math

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

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


def print_block(symbol = '*', nb_sybl = 70):
	print(symbol*nb_sybl)

def divide_validation(model, net_str, cell_input_data, nb_divide):
	if 'buildMulti' in net_str:
		if nb_divide >= 1:
			den_list1 = []
			den_list2 = []
			den_list3 = []
			den_list4 = []
			sub_len = math.ceil(len(cell_input_data)/nb_divide) # the number in each divide
			for i in range(nb_divide):
				start_idx = sub_len*i
				end_idx = min(sub_len*(i+1),len(cell_input_data))
				if start_idx >= len(cell_input_data):
					continue
				preds = model.predict(cell_input_data[start_idx:end_idx])
				if preds[-1].shape[0] ==1:
					den_list4.append(np.squeeze(preds[-1]).reshape((1,)+np.squeeze(preds[-1]).shape))
					den_list3.append(np.squeeze(preds[-2]).reshape((1,)+np.squeeze(preds[-2]).shape))
					den_list2.append(np.squeeze(preds[-3]).reshape((1,)+np.squeeze(preds[-3]).shape))
					den_list1.append(np.squeeze(preds[-4]).reshape((1,)+np.squeeze(preds[-4]).shape))
				else:
					den_list4.append(np.squeeze(preds[-1]))
					den_list3.append(np.squeeze(preds[-2]))
					den_list2.append(np.squeeze(preds[-3]))
					den_list1.append(np.squeeze(preds[-4]))
			den_den1 = np.concatenate(den_list1,axis = 0)
			den_den2 = np.concatenate(den_list2,axis = 0)
			den_den3 = np.concatenate(den_list3,axis = 0)
			den_den4 = np.concatenate(den_list4,axis = 0)
			pred_den = [den_den1, den_den2, den_den3, den_den4]
	else:
		if nb_divide >= 1:
			den_list = []
			sub_len = math.ceil(len(cell_input_data)/nb_divide) # the number in each divide
			for i in range(nb_divide):
				start_idx = sub_len*i
				end_idx = min(sub_len*(i+1),len(cell_input_data))
				if start_idx < len(cell_input_data):
					preds = model.predict(cell_input_data[start_idx:end_idx])
					if preds.shape[0] ==1:
						den_list.append(np.squeeze(preds).reshape((1,)+np.squeeze(preds).shape))
					else:
						den_list.append(np.squeeze(preds))
			pred_den = np.concatenate(den_list,axis = 0)
# 	elif not 'buildMulti' in net_str and not 'Count_ception' in net_str:
# 		if nb_divide >= 1:
# 			den_list = []
# 			sub_len = math.ceil(len(cell_input_data)/nb_divide) # the number in each divide
# 			for i in range(nb_divide):
# 				start_idx = sub_len*i
# 				end_idx = min(sub_len*(i+1),len(cell_input_data))
# 				preds = model.predict(cell_input_data[start_idx:end_idx])
# 				den_list.append(np.squeeze(preds))
# 			pred_den = np.concatenate(den_list,axis = 0)
	return pred_den

def model_evaluation(model_folder, reset_best_weight = False, best_mse = False, divide = 5):
	## add the following codes to log the printout to a log file
	## edited by shenghua, Jun 19, 2019
	# import sys
	# old_lod = sys.stdout
	# log = open('model_evation.log','a')
	# sys.stdout = log

	## load dataset
# 	model_folder = model_folders[0]
	net_str, data_version, cross_nb, output_folder = parse_model_name(model_folder, network_key = 'build', dvkey = 'v-', crosskey = 'cross-')
	_, X_val, _, Y_val = load_train_data(val_version = data_version, cross_val = cross_nb, normal_proc = True)
	print(os.path.basename(model_folder))
	## create model
	fcns = globals()[net_str]
	input_shape = X_val.shape[1:3]
	if not 'Count_ception' in net_str:
		if data_version%10 == 5:
			# adjust the network input size	
			input_shape = (input_shape[0]+8,input_shape[0]+8)
			# zero-pad the input data
			X_zap = np.zeros((X_val.shape[0],)+input_shape+(3,))
			X_zap[:,4:-4,4:-4,:] = X_val
			X_val = X_zap
		elif data_version%10 == 6:
			# adjust the network input size	
			input_shape = (input_shape[0]+4,input_shape[0]+4)
			# zero-pad the input data
			X_zap = np.zeros((X_val.shape[0],)+input_shape+(3,))
			X_zap[:,2:-2,2:-2,:] = X_val
			X_val = X_zap
	else:
		X_val, Y_val = hf.data_trans(X_val, Y_val)
		Y_val = Y_val/1024
	model = fcns(input_shape)
	## load the weights
	# search the best model weights
	best_weight_file = os.path.join(model_folder,'best_weights.h5')
	
	# reset the best weight to none
	if reset_best_weight:
		if os.path.exists(best_weight_file):
			_ = os.system('rm {}'.format(best_weight_file))
		
	if not os.path.exists(best_weight_file):
		print_block()
		print('Best weight selection ...')
	# 	os.makedirs(folder)
		weight_file_ptrn = os.path.join(model_folder, 'weights*.h5')
		weights_list = glob.glob(weight_file_ptrn)
		weights_list = natsorted(weights_list)
		if len(weights_list) == 0:
			print('Empty weights')
		else:
			## summary for the models ##
	# 		print_block()
			last_weight_file =  os.path.basename(weights_list[-1])
			re_ = re.search('_\d+',last_weight_file)
			print('Total weights: {0:}, last update epoch: {1:}'.format(len(weights_list), re_.group(0)[1:]))
			mae_from_weight_list = []
			for w_file in weights_list:
				model.load_weights(w_file, by_name = True)
				if len(X_val.shape) <= 3:
					X_val = X_val.reshape(X_val.shape+(1,))
				# check whether the outputs are multiple
				if 'buildMultiModel' in net_str:
# 					if len(X_val.shape) <= 3:
# 						X_val = X_val.reshape(X_val.shape+(1,))
					preds = divide_validation(model, net_str, X_val, divide)
# 					preds = model.predict(X_val)
					count_arr_list = []
					for pred in preds:
						pred = pred/100
						count = np.squeeze(np.apply_over_axes(np.sum, pred, axes =[1,2]))
						count_arr_list.append(count.reshape(count.shape+(1,)))
					pred_matrix = np.concatenate(count_arr_list, axis = -1)
	# 				pred_count = np.apply_over_axes(np.sum,pred_matrix, [1,2])
					gt_count = np.squeeze(np.apply_over_axes(np.sum,Y_val,axes = [1,2]))			
					gt_count = gt_count.reshape((len(gt_count),1))
					abs_err_matrix = np.abs(pred_matrix - gt_count)
					mae_arr = np.mean(abs_err_matrix, axis = 0)
					std_arr = np.std(abs_err_matrix, axis = 0)
					mae_from_weight_list.append(mae_arr[-1])
					print('MAE:{}'.format(mae_arr))
	# 				print('STD:{}'.format(mae_arr))
					print_block(symbol = '-', nb_sybl = 70)
				else:
# 					if len(X_val.shape) <= 3:
# 						X_val = X_val.reshape(X_val.shape+(1,))
					pred = divide_validation(model, net_str, X_val, divide)
# 					pred = model.predict(X_val)
					if not 'Count_ception' in net_str:
						pred = pred/100
					else:
						pred = pred/1024
					pred_count = np.squeeze(np.apply_over_axes(np.sum, pred, axes =[1,2]))
					gt_count = np.squeeze(np.apply_over_axes(np.sum,Y_val,axes = [1,2]))
# 					print([pred_count[0],gt_count[0]])
					abs_err_arr = np.abs(pred_count - gt_count)
					mae_arr = np.mean(abs_err_arr)
					std_arr = np.std(abs_err_arr)
					mae_from_weight_list.append(mae_arr)
					weight_name = os.path.basename(w_file)
					print('MAE:{} {}'.format(mae_arr,weight_name))
					print_block(symbol = '-', nb_sybl = 70)
			# select the best weight
			if best_mse:
				min_idx = len(mae_from_weight_list)-1
			else:
				min_idx = np.argmin(mae_from_weight_list)
			best_w_file = weights_list[min_idx]
			_ = os.system('cp {} {}'.format(best_w_file,best_weight_file))
			w_file =  os.path.basename(best_w_file)
			re_ = re.search('_\d+',w_file)
			print('Best weight is selected, min MAE:{0:.2f}, at epoch: {1:}'.format(mae_from_weight_list[min_idx],re_.group(0)[1:]))
			print_block()

	## begin the analysis
	model.load_weights(best_weight_file, by_name = True)
	pred_dens = None
	print_block(symbol = '=', nb_sybl = 70)
	print('Model:{}'.format(os.path.basename(model_folder)))
	print('Results analysis with best weight')
	run_time = 0  # run time calculation
	if 'buildMultiModel' in net_str:
		if len(X_val.shape) <= 3:
			X_val = X_val.reshape(X_val.shape+(1,))
		start_time = time.time()
		preds = divide_validation(model, net_str, X_val, divide)
		end_time = time.time()
		run_time = end_time - start_time
# 		preds = model.predict(X_val)
		count_arr_list = []
		for pred in preds:
			pred = pred/100
			count = np.squeeze(np.apply_over_axes(np.sum, pred, axes =[1,2]))
			count_arr_list.append(count.reshape(count.shape+(1,)))
		pred_matrix = np.concatenate(count_arr_list, axis = -1)
	# 				pred_count = np.apply_over_axes(np.sum,pred_matrix, [1,2])
		gt_count = np.squeeze(np.apply_over_axes(np.sum,Y_val,axes = [1,2]))			
		gt_count = gt_count.reshape((len(gt_count),1))
		abs_err_matrix = np.abs(pred_matrix - gt_count)
		mae_arr = np.mean(abs_err_matrix, axis = 0)
		std_arr = np.std(abs_err_matrix, axis = 0)
		pred_dens = np.squeeze(preds[-1])/100
	# 	mae_from_weight_list.append(mae_arr[-1])
		print('MAE:{}'.format(mae_arr))
	# 				print('STD:{}'.format(mae_arr))
	else:
		if len(X_val.shape) <= 3:
			X_val = X_val.reshape(X_val.shape+(1,))
		start_time = time.time()
		pred = divide_validation(model, net_str, X_val, divide)
		end_time = time.time()
		run_time = end_time - start_time
# 		pred = model.predict(X_val)
		if not 'Count_ception' in net_str:
			pred = pred/100
		else:
			pred = pred/1024
# 		pred = pred/100
		pred_count = np.squeeze(np.apply_over_axes(np.sum, pred, axes =[1,2]))
		gt_count = np.squeeze(np.apply_over_axes(np.sum,Y_val,axes = [1,2]))
		abs_err_arr = np.abs(pred_count - gt_count)
		mae_arr = np.mean(abs_err_arr)
		std_arr = np.std(abs_err_arr)
		pred_dens = np.squeeze(pred)
	# 	mae_from_weight_list.append(mae_arr)
# 		print_block(symbol = '-', nb_sybl = 70)
		print(mae_arr)

	full_output_folder = os.path.join(results_root_folder, net_str, 'data_version-{}'.format(data_version), 'cross-{}'.format(cross_nb))
#	full_output_folder = os.path.join(results_root_folder, output_folder, 'data_version-{}'.format(data_version), 'cross-{}'.format(cross_nb))
	generate_folder(full_output_folder)
	run_time_per_image = round(run_time/X_val.shape[0], 5)
	for i in range(X_val.shape[0]):
		image = np.squeeze(X_val[i,:,:])
		gt_den = np.squeeze(Y_val[i,:,:])
		est_den = np.squeeze(pred_dens[i,:,:])
		output_file = os.path.join(full_output_folder, '{}.pkl'.format(i))
		save_any_pickle(output_file, data_list = [est_den, gt_den, image, run_time_per_image], keys = ['est_den', 'gt_den', 'ori_img', 'run_time'])
	
	K.clear_session()
	print('Results saved...')
	print_block(symbol = '=', nb_sybl = 70)

	## exit the log
	# sys.stdout = old_log
	# log.close()

## used in unannotated cell data processing
def save_cell_data_pair(file_name, data):
	import pickle
	with open(file_name, 'wb') as f:
		pickle.dump(data,f)


## test a certain model folder
def test_single_model_folder():
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	## the parameters
	# models_root_folder = os.path.expanduser('~/dl-cells/dlct-framework/models/')    # the folder that store all the selected models
	models_root_folder = os.path.expanduser('~/dl-cells/dlct-framework/models/paper_models_w_cv')    # the folder that store all the selected models
	results_root_dir = os.path.expanduser('~/dl-cells/dlct-framework/results')         # the folder that store the output results
	# result_version = 'ACCV2018_v1'
	result_version = 'MIA2018_v3'
	date = '6.6-7.6'    # date to the evaluation for trained models
	divide = 2
	reset_best = True  # control whether to reset the best weight file
	best_mse = False    # control whether the model with lowest mse will be used to optimize the model
	results_root_folder = os.path.join(results_root_dir, '{0:}_{1:}'.format(result_version,date))
	# print_log('model_evaluation.log')   ## begin to printout the output to the log file
#	model_folder = os.path.join(models_root_folder, 'date-6.22-buildModel_FCRN_A-mse-v-26-cross-4-batch-100-drop-None-lr-0.0001-nb_epochs-1-norm-True-t-0')
#  	model_folder = os.path.join(models_root_folder, 'date-6.25-buildModel_U_net-mse-v-26-cross-4-batch-100-drop-None-lr-0.0001-nb_epochs-1-norm-True-t-0')
#	model_folder = os.path.join(models_root_folder, 'date-6.22-buildMultiModel_U_net-mse-v-26-cross-4-batch-100-drop-None-lr-0.0001-nb_epochs-1-norm-True-t-0')
	# model_folder = os.path.join(models_root_folder, 'date-9.1-buildMultiModel_U_net-mse-v-27-cross-0-batch-100-drop-None-lr-0.00015-nb_epochs-1-norm-True-t-0')
	# model_folder = os.path.join(models_root_folder, 'date-6.9-buildModel_FCRN_A-mse-v-24-cross-4-batch-100-drop-None-lr-0.005-nb_epochs-1-norm-True-t-0')
	# model_folder = os.path.join(models_root_folder, 'date-6.10-buildModel_U_net-mse-v-24-cross-2-batch-100-drop-None-lr-0.005-nb_epochs-1-norm-True-t-0')
	# model_name = 'date-6.16-buildModel_FCRN_A-mse-v-25-cross-4-batch-60-drop-None-lr-0.0001-nb_epochs-1-norm-True-t-0'
	model_name = 'date-6.10-buildModel_U_net-mse-v-24-cross-4-batch-100-drop-None-lr-0.005-nb_epochs-1-norm-True-t-0'
	model_folder = os.path.join(models_root_folder, model_name)
	model_evaluation(model_folder, reset_best_weight = reset_best, best_mse = best_mse, divide = divide)
	# stop_print()

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	## the parameters
# 	models_root_folder = os.path.expanduser('~/dl-cells/dlct-framework/models/paper_models_wo_cv')    # the folder that store all the selected models
	models_root_folder = os.path.expanduser('~/dl-cells/dlct-framework/models/paper_models_w_cv')
	results_root_dir = os.path.expanduser('~/dl-cells/dlct-framework/results')         # the folder that store the output results
# 	result_version = 'ACCV2018_v1'
	# result_version = 'MIA2018_v2'
	date = '6.6-7.6'    # date to the evaluation for trained models
	result_version = 'MIA2018_v3'
	divide = 10
	results_root_folder = os.path.join(results_root_dir, '{0:}_{1:}'.format(result_version,date))
	reset_best = True  # control whether to reset the best weight file
	best_mse = False    # control whether the model with lowest mse will be used to optimize the model
	model_folders_ptrn = '*'
	final_model_folders_ptrn = os.path.join(models_root_folder, model_folders_ptrn)
	model_folders = glob.glob(final_model_folders_ptrn)
	model_folders = natsorted(model_folders)
	for model_folder in model_folders:
		model_evaluation(model_folder, reset_best_weight = reset_best, best_mse = best_mse, divide = divide)