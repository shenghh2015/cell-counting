import os
import numpy as np
from skimage import io

docker = True
dataset = 'cell_cycle_1984_v2'; subset ='train'
data_folder = '/data/datasets/{}'.format(dataset) if docker else ''

phase_img_folder = data_folder+'/{}_images'.format(subset)
fl_img_folder = data_folder+'/{}_fmasks'.format(subset)
fl_filtered_folder = data_folder+'/{}_ffmasks'.format(subset)

fnames = os.listdir(phase_img_folder)

def denormalize(x):
	import numpy as np
	x = (x-x.min())/(x.max()-x.min())
	return np.uint8(255.*x)

## plot function
# image, fluorescent, filtered fluorescent
def plot_task_images(file_name, phase_dir, fl_img_dir, fl_filtered_dir, nb_images= 5, rand_seed = 3):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	from skimage import io 
	import random
	import os
	fnames = os.listdir(phase_dir)
	seed = rand_seed #3
	random.seed(seed)
	font_size = 24
	indices = random.sample(range(len(fnames)),nb_images)
	rows, cols, size = nb_images,3,5
	fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
	for i in range(len(indices)):
		idx = indices[i]
		image = io.imread(os.path.join(phase_dir,fnames[idx])); image = denormalize(image)
		fl = io.imread(os.path.join(fl_img_dir, fnames[idx]))
		fl[:,:,0] = denormalize(fl[:,:,0]); fl[:,:,1] = denormalize(fl[:,:,1])
		fl_filtered = io.imread(os.path.join(fl_filtered_dir, fnames[idx]))
		fl_filtered[:,:,0] = denormalize(fl_filtered[:,:,0]); fl_filtered[:,:,1] = denormalize(fl_filtered[:,:,1])
		ax[i,0].imshow(image); ax[i,1].imshow(fl); ax[i,2].imshow(fl_filtered); 
# 		ax[i,0].set_xticks([]);ax[i,1].set_xticks([]);ax[i,2].set_xticks([]);ax[i,3].set_xticks([])
# 		ax[i,0].set_yticks([]);ax[i,1].set_yticks([]);ax[i,2].set_yticks([]);ax[i,3].set_yticks([])
		if i == 0:
			ax[i,0].set_title('Image',fontsize=font_size); ax[i,1].set_title('Raw fl1+fl2',fontsize=font_size); 
			ax[i,2].set_title('Flitered fl1+fl2',fontsize=font_size)
	canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=80)

file_name = './phase_fluorescent.png'
plot_task_images(file_name, phase_img_folder, fl_img_folder, fl_filtered_folder, nb_images= 3, rand_seed = 4)