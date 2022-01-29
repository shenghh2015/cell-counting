import matplotlib.pyplot as plt
import os
import numpy as np
from utils.metrics import *

## 3.16 overlap analysis based on density map
def setup_subplot(fig, image1, gt, est,  acc, cosin, row):
# 	rgbImg, annImg, _, denImg = read_train_pickles(data_name, keys = ['rgb', 'annot', 'ori', 'den'])
	ax = fig.add_subplot(1,3,row*3+1)
	cax = ax.imshow(image1, interpolation='nearest')
# 	ax.set_ylabel(data_label)
	fig.colorbar(cax)
	bx = fig.add_subplot(1,3,row*3+2)
	cbx = bx.imshow(gt, interpolation='nearest')
	fig.colorbar(cbx)
	cx = fig.add_subplot(1,3,row*3+3)
	ccx = cx.imshow(est, interpolation='nearest')
	cx.set_xlabel('Cosine:{0:0.2f}, ACC:{1:0.2f}'.format(cosin,acc))
	fig.colorbar(ccx)
	if row == 0:
		ax.set_title('Cell image')
		bx.set_title('GT density')
		cx.set_title('Estimated density')

def display_density_map_overlap(fig, imgSet, predSet1, grdSet1, nb_slides = 20, interval = 1):
	import numpy as np
	import matplotlib.pyplot as plt
	for i in range(nb_slides):
		plt.clf()
		# read the image
		# row 1
		predSet = predSet1
		grdSet = grdSet1
		shp = predSet1.shape
		image = imgSet[i,:].reshape(shp[1],shp[2])
		ground_truth = grdSet[i,:].reshape(shp[1],shp[2])
		max_value = np.max(ground_truth)
		min_value = np.min(ground_truth)
		pred_map = predSet[i,:].reshape(shp[1],shp[2])
		pred_count = np.sum(pred_map)
		real_count = np.sum(ground_truth)
# 		nb_subfigures = len(thresholds)+1
# 		ax = fig.add_subplot(2,nb_subfigures,1)
# 		cax = ax.imshow(image)
		gt = ground_truth
		pd = pred_map
		_cosine = density_Cosine_calculate(gt,pd)
		acc = 1-np.abs(pred_count-real_count)/real_count
		pd = (pd - np.min(pd))/(np.max(pd)-np.min(pd))*(max_value-min_value)+min_value
		setup_subplot(fig,image,gt,pd,acc,_cosine,0)
		plt.pause(interval)

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

# plot and save the file
def plot_metrics(model_name, file_name, mae_list,std_list):
	generate_folder(model_name)
	f_out=os.path.join(model_name,file_name)
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig = Figure(figsize=(8,6))
	ax = fig.add_subplot(1,1,1)
	mae_arr = np.array(mae_list)
	std_arr = np.array(std_list)
	up_arr = mae_arr + std_arr
	lower_arr = mae_arr - std_arr
	ax.plot(mae_list,'b-',linewidth=1.3)
	ax.plot(up_arr,'r:',linewidth=1.3)
	ax.plot(lower_arr,'r:',linewidth=1.3)
	ax.set_title('Performance')
	ax.set_ylabel('Count')
	ax.set_xlabel('Number')
	ax.legend(['Mean', r'$\mu1+\sigma1$', r'$\mu1-\sigma1$'], loc='upper right')  
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(f_out, dpi=80)
