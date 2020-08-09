from data_load import *
from scipy import signal
from skimage import io

## create folder 
def generate_folder(folder):
	if not os.path.exists(folder):
		os.system('mkdir -p {}'.format(folder))

## create a solid cycle shape
def solid_cycle(radius=3):
    cycle = np.zeros((2*radius+1,2*radius+1))
    xs, ys = [],[]
    for i in range(-radius, radius+1):
        for j in range(-radius,radius+1):
            if i**2+j**2<=radius**2:
                xs.append(i+radius); ys.append(j+radius)
    cycle[xs,ys]=1
    return cycle

docker = False
dataset_root = '/data/datasets' if docker else '../datasets/'

# dataset_pool = ['bacterial', 'bone_marrow', 'colorectal', 'hESC']
dataset_pool = ['colorectal', 'hESC']
cross_pool = [1,2,3,4,5]

for dataset in dataset_pool:
	if dataset == 'bacterial':
		dataset_id = 44
		radius = 4
	elif dataset == 'bone_marrow':
		dataset_id = 45
		radius = 8
	elif dataset == 'colorectal':
		dataset_id = 46
		radius = 5
	elif dataset == 'hESC':
		dataset_id = 47
		radius = 3
	for cross in cross_pool:
		# dataset = 'bacterial'; cross = 1
		print('Processing dataset: {} cross-{}'.format(dataset, cross))
		cross_folder = 'cross-{}'.format(cross)
		cycle_object = solid_cycle(radius)
		X_train, X_test, Y_train, Y_test = load_train_data(val_version=dataset_id, cross_val = cross)

		for subset in ['train', 'test']:
			if subset == 'train':
				X, Y = X_train, Y_train
			elif subset == 'test':
				X, Y = X_test, Y_test

			subset_folder = os.path.join(dataset_root, 'mrcnn', dataset, cross_folder, subset)
			generate_folder(subset_folder)

			for index in range(X.shape[0]):
				# index = 0
				img_id = '{:03d}'.format(index)
				data_folder = subset_folder+'/'+img_id
				img_folder = data_folder+'/images'; mask_folder = data_folder+'/masks'
				generate_folder(img_folder); generate_folder(mask_folder)
				img = X[index,:,:,:].squeeze(); img = np.uint8(255.0*(img-img.min())/(img.max()-img.min()))
				io.imsave(img_folder+'/{}.png'.format(img_id), img)
				dot_map = Y[index,:,:].squeeze()
				xs, ys = np.where(dot_map>0)
				for i in range(len(xs)):
					xi, yi = xs[i], ys[i]
					mask = np.zeros(dot_map.shape,dtype = np.uint8)
					mask[xi,yi] = 255
					mask = np.uint8(signal.convolve2d(mask, cycle_object, boundary='fill', mode='same'))
					io.imsave(mask_folder+'/mask_{:03d}.png'.format(i), mask)
