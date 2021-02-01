from data_load import *
from scipy import signal
from skimage import io

## create folder 
def generate_folder(folder):
	if not os.path.exists(folder):
		os.system('mkdir -p {}'.format(folder))

## create a solid marker 
def solid_marker(radius=13,alpha = 3):
    cycle = np.zeros((2*radius+1,2*radius+1))
    xs, ys = [],[]
    for i in range(-radius, radius+1):
        for j in range(-radius,radius+1):
            if i**2+j**2<=radius**2:
                cycle[i+radius, j +radius] = (np.exp(alpha-alpha*np.sqrt(i**2+j**2)/radius)-1)/(np.exp(alpha)-1)
    return cycle

docker = False
dataset_root = '/data/datasets' if docker else '../datasets/'

dataset_pool = ['bacterial', 'bone_marrow', 'colorectal', 'hESC']
# dataset_pool = ['colorectal', 'hESC']
# dataset_pool = ['colorectal']
method_name = 'regnet'
# cross_pool = [1]
cross_pool = [1,2,3,4,5]

for dataset in dataset_pool:
    if dataset == 'bacterial':
        dataset_id = 44
#         radius = 4
    elif dataset == 'bone_marrow':
        dataset_id = 45
#         radius = 8
    elif dataset == 'colorectal':
        dataset_id = 46
#         radius = 5
    elif dataset == 'hESC':
        dataset_id = 47
#         radius = 3
    for cross in cross_pool:
        # dataset = 'bacterial'; cross = 1
        print('Processing dataset: {} cross-{}'.format(dataset, cross))
        cross_folder = 'cross-{}'.format(cross)
        cycle_object = solid_marker(13, 3)
        X_train, X_test, Y_train, Y_test = load_train_data(val_version=dataset_id, cross_val = cross)

        for subset in ['train', 'val']:
            if subset == 'train':
                X, Y = X_train, Y_train
            elif subset == 'val':
                X, Y = X_test, Y_test

            subset_folder = os.path.join(dataset_root, method_name, dataset, cross_folder, subset)
            generate_folder(subset_folder)
            img_folder = subset_folder+'/images'; mask_folder = subset_folder+'/masks'; dot_folder = subset_folder+'/dots'
            generate_folder(img_folder); generate_folder(mask_folder); generate_folder(dot_folder)

            for index in range(X.shape[0]):
                img_id = '{:03d}'.format(index)
                img = X[index,:,:,:].squeeze(); img = np.uint8(255.0*(img-img.min())/(img.max()-img.min()))
                io.imsave(img_folder+'/{}.png'.format(img_id), img)
                dot_map = np.uint8(Y[index,:,:].squeeze())
                mask_map = signal.convolve2d(dot_map, cycle_object, boundary='fill', mode='same')*3
#                 mask_map = np.uint8((mask_map>0)*1.0)
                io.imsave(mask_folder+'/{}.png'.format(img_id), mask_map)
                io.imsave(dot_folder+'/{}.png'.format(img_id), dot_map)