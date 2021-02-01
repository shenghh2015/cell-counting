import os
import numpy as np
from utils.file_load_save import read_any_pickle

def generate_folder(folder):
	if not os.path.exists(folder):
		os.system('mkdir -p {}'.format(folder))

def crop2original_size(image, img_dim):
	shp = image.shape
	left = int((shp[0]-img_dim)/2)
	right = shp[1]-int((shp[1]-img_dim)/2)
	
	return image[left:right, left:right,:] if len(shp)==3 else image[left:right, left:right]

method = 'ception'
pr_root_dir = '/shared/planck/Cellcounting/turing/shenghua/dl-cells/dlct-framework/results/MIA2018_v3_6.6-7.6'  # the results previously generated for the paper
subset = 'val'
result_root_dir = '/home/sh38/cell_counting/results/{}/'.format(method)  # the dir to save the visualized density map figures
if method == 'dcfcrn':
	method_legacy_name = 'buildMultiModel_U_net'
elif method == 'cfcrn':
	method_legacy_name = 'buildModel_U_net'
elif method == 'fcrn':
	method_legacy_name = 'buildModel_FCRN_A'
elif method == 'ception':
	method_legacy_name = 'buildModel_Count_ception'

dataset_set = ['bacterial', 'bone_marrow', 'colorectal', 'hESC']
cross_set = [1,2,3,4,5]
# dataset = 'bacterial'
# cross = 1

for dataset in dataset_set:
	lines = []
	tags = []
	cv_lines =[]
	count_errs = []
	rel_errs = []
	cv_count_errs = []
	cv_rel_errs = []
	for cross in cross_set:
		print('{}-{}-cross-{}'.format(method, dataset, cross))
		if dataset == 'bacterial':
			dataset_legacy_name = 'data_version-24' if not method == 'ception' else 'data_version-44'
			image_dim = 256
		elif dataset == 'bone_marrow':
			dataset_legacy_name = 'data_version-25' if not method == 'ception' else 'data_version-45'
			image_dim = 600
		elif dataset == 'colorectal':
			dataset_legacy_name = 'data_version-26' if not method == 'ception' else 'data_version-46'
			image_dim = 500
		elif dataset == 'hESC':
			dataset_legacy_name = 'data_version-27' if not method == 'ception' else 'data_version-47'
			image_dim = 512

		pr_result_dir = os.path.join(pr_root_dir, method_legacy_name, dataset_legacy_name, 'cross-{}'.format(cross))
		fnames = os.listdir(pr_result_dir)

		density_dir = os.path.join(result_root_dir, dataset, 'cross-{}'.format(cross), subset)
		generate_folder(density_dir)

		cv_count_err, cv_rel_err = [], []
		for fname in fnames:
			# load the previous data
			pkl_file = pr_result_dir +'/{}'.format(fname)
			results = read_any_pickle(pkl_file, keys = ['est_den', 'gt_den', 'ori_img', 'run_time'])
			est_density = results[0]
			gt_density = results[1]
			image = results[2]
			# save the results to the new folder
			image_id = int(fname.split('.pkl')[0])
			if dataset == 'bone_marrow' or dataset == 'colorectal':
				image = crop2original_size(image, image_dim)
				gt_density = crop2original_size(gt_density, image_dim)
				est_density = crop2original_size(est_density, image_dim)
			np.save(density_dir+'/{:03d}.npy'.format(image_id), est_density)
			np.save(density_dir+'/gt_{:03d}.npy'.format(image_id), gt_density)
			np.save(density_dir+'/img_{:03d}.npy'.format(image_id), image)
			# calculate errors
			if method == 'ception':
				pr_count = est_density.sum(); gt_count = gt_density.sum()
			else:
				pr_count = est_density.sum(); gt_count = gt_density.sum()
			count_errs.append(abs(pr_count-gt_count))
			rel_errs.append(abs(pr_count-gt_count)/gt_count)
			cv_count_err.append(abs(pr_count-gt_count))
			cv_rel_err.append(abs(pr_count-gt_count)/gt_count)
			# tag
			tags.append('cross-{},{},gt {}, pr {:.2f}'.format(cross,'{:03d}.npy'.format(image_id), round(gt_count), pr_count))
		cv_count_errs.append(np.mean(cv_count_err))
		cv_rel_errs.append(np.mean(cv_rel_err))					
	print('{}-{}: MCE {:.4f}, STD {:.4f}, MRE {:.4f}, STD {:.4f}'.format(method, dataset, np.mean(count_errs), np.std(count_errs), np.mean(rel_errs), np.std(rel_errs)))
	print('CV-{}-{}: MCE {:.4f}, STD {:.4f}, MRE {:.4f}, STD {:.4f}'.format(method, dataset, np.mean(cv_count_errs), np.std(cv_count_errs), np.mean(cv_rel_errs), np.std(cv_rel_errs)))
	lines.append('{}-{}: MCE {:.4f}, STD {:.4f}, MRE {:.4f}, STD {:.4f}'.format(method, dataset, np.mean(count_errs), np.std(count_errs), np.mean(rel_errs), np.std(rel_errs)))
	cv_lines.append('CV-{}-{}: MCE {:.4f}, STD {:.4f}, MRE {:.4f}, STD {:.4f}'.format(method, dataset, np.mean(cv_count_errs), np.std(cv_count_errs), np.mean(cv_rel_errs), np.std(cv_rel_errs)))
	# save the error
	result_file = os.path.join(result_root_dir, dataset, 'err_summary.txt')
	with open(result_file, 'w+') as f:
		f.write('Count err summary:\n')
		for i, err in enumerate(count_errs):
			f.write('{}: count err {:.4f}, Relative err {:.4f}\n'.format(tags[i], err, rel_errs[i]))
	# save the count err and relative err
	np.savetxt(os.path.join(result_root_dir, dataset, 'count_err.txt'), count_errs)
	np.savetxt(os.path.join(result_root_dir, dataset, 'rel_err.txt'), rel_errs)