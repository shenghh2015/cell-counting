import os
import numpy as np
import glob
import cv2

from skimage import io
from nms_python import nms
from helper_function import generate_folder
import time

def gen_cross(image, centroid, size, thickness, color):
	x1, y1, x2, y2 = centroid[0]-int(size/2), centroid[1]-int(size/2), centroid[0]+int(size/2), centroid[1]+int(size/2)
	image[x1:x2, centroid[1]-thickness:centroid[1]+thickness,:] = color
	image[centroid[0]-thickness:centroid[0]+thickness, y1:y2,:] = color
	return image

result_root_dir = '/home/sh38/cell_counting/results'

subset = 'val'
# method = 'unet'
method = 'regnet'
lines = []
cv_lines = []
time_lines = []
# dataset = 'bacterial'
#dataset = 'bone_marrow'
# dataset = 'colorectal'
# dataset = 'hESC'
# cross = 1
dataset_set = ['bacterial','bone_marrow','colorectal','hESC']
#dataset_set = ['colorectal']
for dataset in dataset_set:
	# non-maximum suppression
	if method == 'regnet':
		if dataset == 'bacterial':
			radius = 4
			marker_size = 6
			img_dim = 256
			hf_box_size = 7
			thickness = 1
			threshold = 0.27
		if dataset == 'bone_marrow':
			radius = 8
			marker_size = 10
			img_dim = 600
			hf_box_size = 13
			thickness = 2
			threshold = 0.3
		if dataset == 'colorectal':
			radius = 5
			marker_size = 6
			img_dim = 500
			hf_box_size = 9
			thickness = 2
			threshold = 0.3
		if dataset == 'hESC':
			radius = 3
			marker_size = 4
			img_dim = 512
			hf_box_size = 7
			thickness = 1
			threshold = 0.25
	elif method == 'unet':
		if dataset == 'bacterial':
			radius = 4
			marker_size = 6
			img_dim = 256
			hf_box_size = 9
			thickness = 1
			threshold = 0.32
		if dataset == 'bone_marrow':
			radius = 8
			marker_size = 10
			img_dim = 600
			hf_box_size = 25
			thickness = 2
			threshold = 0.3
		if dataset == 'colorectal':
			radius = 5
			marker_size = 6
			img_dim = 500
			hf_box_size = 11
			thickness = 2
			threshold = 0.45
		if dataset == 'hESC':
			radius = 3
			marker_size = 4
			img_dim = 512
			hf_box_size = 7
			thickness = 1
			threshold = 0.25

	count_err = []
	rel_err = []
	cv_count_errs = []  # five values of errs evaluated on each fold
	cv_rel_errs = []	# five values of errs evaluated on each fold
	counting_times = []
	tags = []
	cross_set = [1,2,3,4,5]
	for cross in cross_set:
		# create a subfolder: result_root_dir/method/dataset/cross
		result_dir = os.path.join(result_root_dir, method, dataset, 'cross-{}/{}'.format(cross, subset))
		generate_folder(result_dir)

		# read the prediction maps
		pred_dir = os.path.join('./datasets',method, dataset, 'cross-{}/{}'.format(cross, subset))
		image_dir = pred_dir +'/images'
		dot_dir = pred_dir+'/dots'
		pr_map_dir = pred_dir +'/pr_masks'
		pr_time_dir = pred_dir+'/pr_times'

		dot_fnames = os.listdir(dot_dir)
		fnames = os.listdir(pr_map_dir)

		images = [io.imread(image_dir+'/{}'.format(fname)) for fname in dot_fnames]
		dot_maps = [io.imread(dot_dir+'/{}'.format(fname)) for fname in dot_fnames]
		pr_maps = [np.load(pr_map_dir+'/{}'.format(fname.replace('png', 'npy'))) for fname in dot_fnames]
		map_infer_time = np.load(pr_time_dir+'/ave_time_per_image.npy')

		dot_maps = np.stack(dot_maps)
		pr_maps = np.stack(pr_maps)

		cv_count_err, cv_rel_err = [], []
		for img_idx in range(len(dot_fnames)):
			# img_idx = 4
			start_time = time.time()
			image = images[img_idx].copy()
			dot_map = dot_maps[img_idx]
			pr_map = pr_maps[img_idx]
			# thickness = 1
			ys, xs = np.where(dot_map>0)
			gt_count = len(xs)
			for i in range(len(ys)):
				yi, xi = ys[i], xs[i]
				cv2.circle(image,(xi,yi),radius, color=(0,255,0), thickness= thickness)
			## non-maximum
			thres_mask = pr_map >threshold
			pr_mask = pr_map
			ys, xs = np.where(pr_mask>threshold)
			bound_boxes = []
			probs = []
			for i in range(len(xs)):
				xi, yi = xs[i], ys[i]; #print(xi,yi)
				x1, x2, y1, y2 = max(0,xi-hf_box_size), min(img_dim,xi+hf_box_size), max(0,yi-hf_box_size), min(img_dim,yi+hf_box_size)
				probs.append(pr_mask[xi,yi])
				bound_boxes.append((x1, y1, x2, y2))
			# print(len(probs)); print(len(bound_boxes))
			boxes = np.array(bound_boxes)
			det_centroids = nms.non_max_suppression_fast(boxes, probs, overlapThresh=0.4)
			det_count = len(det_centroids)
			end_time = time.time()
			counting_time = end_time-start_time + map_infer_time
			counting_times.append(counting_time)
			# print('GT: {} Detected: {}'.format(gt_count, len(det_centroids)))
			for (startX, startY, endX, endY) in det_centroids:
				xc, yc = int((endX+startX)/2), int((endY+startY)/2)
				origin = gen_cross(image, (yc,xc), marker_size, thickness, color=[255,0,0])
			io.imsave(result_dir+'/{}'.format(dot_fnames[img_idx]), origin)
			# detected cells in the proximity map
			_pr_mask = pr_mask.copy()
			_pr_mask = np.uint8((_pr_mask-_pr_mask.min())*255./(_pr_mask.max()-_pr_mask.min()))
			pr_mask_rgb = np.stack([_pr_mask,_pr_mask,_pr_mask], axis = -1).astype(np.uint8)
			for (startX, startY, endX, endY) in det_centroids:
				xc, yc = int((endX+startX)/2), int((endY+startY)/2)
				pr_mask_rgb = gen_cross(pr_mask_rgb, (yc,xc), marker_size, thickness, color=[255,0,0])
			io.imsave(result_dir+'/pr_{}'.format(dot_fnames[img_idx]), pr_mask_rgb)
				# cv2.circle(image,(xc,yc), radius, color=(255,0,0), thickness= thickness)
			count_err.append(abs(gt_count-det_count))
			rel_err.append(abs(gt_count-det_count)/gt_count)
			cv_count_err.append(abs(gt_count-det_count))
			cv_rel_err.append(abs(gt_count-det_count)/gt_count)
			tags.append('cross-{},{}, gt {}, pr {}'.format(cross, dot_fnames[img_idx], gt_count, det_count))
		cv_count_errs.append(np.mean(cv_count_err))
		cv_rel_errs.append(np.mean(cv_rel_err))
	print('{}-{}: MCE {:.4f}, STD {:.4f}, MRE {:.4f}, STD {:.4f}'.format(method,dataset, np.mean(count_err), np.std(count_err), np.mean(rel_err), np.std(rel_err)))
	print('CV-{}-{}: MCE {:.4f}, STD {:.4f}, MRE {:.4f}, STD {:.4f}'.format(method,dataset, np.mean(cv_count_errs), np.std(cv_count_errs), np.mean(cv_rel_errs), np.std(cv_rel_errs)))
	print('{}-{}: counting time:{:.5f}'.format(method,dataset, np.mean(counting_times)))
	lines.append('{}-{}: MCE {:.4f}, STD {:.4f}, MRE {:.4f}, STD {:.4f}'.format(method,dataset, np.mean(count_err), np.std(count_err), np.mean(rel_err), np.std(rel_err)))
	cv_lines.append('CV-{}-{}: MCE {:.4f}, STD {:.4f}, MRE {:.4f}, STD {:.4f}'.format(method,dataset, np.mean(cv_count_errs), np.std(cv_count_errs), np.mean(cv_rel_errs), np.std(cv_rel_errs)))
	time_lines.append('{}-{}: counting time:{:.5f}'.format(method,dataset, np.mean(counting_times)))
	# save the error
	result_file = os.path.join(result_root_dir, method, dataset, 'err_summary.txt')
	with open(result_file, 'w+') as f:
		f.write('Count err summary:\n')
		for i, err in enumerate(count_err):
			f.write('{}: count err {:.4f}, Relative err {:.4f}\n'.format(tags[i], err, rel_err[i]))
	# save the count err and relative err
	np.savetxt(os.path.join(result_root_dir, method, dataset, 'count_err.txt'), count_err)
	np.savetxt(os.path.join(result_root_dir, method, dataset, 'rel_err.txt'), rel_err)

log_file = os.path.join(result_root_dir, method, '{}_summary.txt'.format(method))
with open(log_file, 'w+') as f:
	for line in lines:
		f.write(line+'\n')

log_file = os.path.join(result_root_dir, method, '{}_cv_summary.txt'.format(method))
with open(log_file, 'w+') as f:
	for line in cv_lines:
		f.write(line+'\n')

log_file = os.path.join(result_root_dir, method, '{}_time_summary.txt'.format(method))
with open(log_file, 'w+') as f:
	for line in time_lines:
		f.write(line+'\n')