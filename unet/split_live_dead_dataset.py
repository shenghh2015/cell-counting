import numpy as np
import os
from skimage import io

docker = True

dataset_folder = '/data/datasets/live_dead' if docker else './'

train_folder=os.path.join(dataset_folder,'train_images')
valid_folder=os.path.join(dataset_folder,'val_images')
test_folder=os.path.join(dataset_folder,'test_images')

image_folder = train_folder; new_folder = image_folder+'2'
if not os.path.exists(new_folder):
	os.system('mkdir -p {}'.format(new_folder))
image_names = os.listdir(image_folder)
for image_name in image_names:
	image=io.imread(image_folder+'/'+image_name)
	print(image_folder+'/'+image_name)
	image[:,:,0]=image[:,:,1]
	io.imsave(new_folder+'/'+image_name, image)
	print(new_folder+'/'+image_name)

# check the regenerated results
indices = np.random.randint(0,len(image_names),10)
for index in indices:
	image_name = image_names[index]
	image1 = io.imread(image_folder+'/'+image_name)
	image2 = io.imread(new_folder+'/'+image_name)
	print(np.sum(image1[:,:,1]-image2[:,:,0]))
	print(np.sum(image1[:,:,2]-image2[:,:,0]))