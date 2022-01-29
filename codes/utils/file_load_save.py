import pickle
import csv

## save and read the processed training data
def save_train_pickles(file_name, image1, image2, image3, image4, keys = ['rgb', 'annot', 'ori', 'den']):
	pickle_item = {keys[0]:image1, keys[1]:image2, keys[2]:image3, keys[3]:image4}
	pickle.dump(pickle_item, open(file_name, 'wb'))

def read_train_pickles(file_name, keys = ['rgb', 'annot', 'ori', 'den']):
	pickle_item = pickle.load(open(file_name, 'rb'))
	return pickle_item[keys[0]], pickle_item[keys[1]], pickle_item[keys[2]], pickle_item[keys[3]]

## save and read the estimation results
def save_pickles(file_name,image1, image2, image3, keys = ['est_den', 'gt_den', 'ori_img']):
	pickle_item = {keys[0]:image1,keys[1]:image2,keys[2]:image3}
	pickle.dump(pickle_item, open(file_name, 'wb'))

def read_pickles(file_name, keys = ['est_den', 'gt_den', 'ori_img']):
	pickle_item = pickle.load(open(file_name, 'rb'))
	return pickle_item[keys[0]], pickle_item[keys[1]], pickle_item[keys[2]]

def save_any_pickle(file_name,data_list = ['', '', ''], keys = ['est_den', 'gt_den', 'ori_img']):
	pickle_item = {}
	for i in range(len(data_list)):
		pickle_item.update({keys[i]:data_list[i]})
	pickle.dump(pickle_item, open(file_name, 'wb'))

def read_any_pickle(file_name, keys = ['est_den', 'gt_den', 'ori_img']):
	pickle_item = pickle.load(open(file_name, 'rb'))
	result_list = []
	for i in range(len(keys)):
		result_list.append(pickle_item[keys[i]])
	return result_list

## load image with folder into a numpy dataset
def load_images_from_folder(folder_name, suffix='.tif', normalized = True):
	import numpy as np
	import glob
	from natsort import natsorted
	from PIL import Image
	file_names = glob.glob(folder_name+'/*'+suffix.upper()) + glob.glob(folder_name+'/*'+suffix.lower())
	file_names = natsorted(file_names)
	image_list = []
# 	if not len(file_names)==0:
	for i, file in enumerate(file_names):
		im = Image.open(file)
		image = np.array(im)
# 		print('max:{}, min{}'.format(np.max(image),np.min(image)))
		if normalized:
			imarray = 255.*(image - np.min(image))*1.0/(np.max(image)-np.min(image))
		else:
			imarray = image
# 		print('-- max:{}, min{}'.format(np.max(imarray),np.min(imarray)))
# 		imarray = image
		image_list.append(imarray)
	image_arr = np.array(image_list)
# 	for i in range(image_arr.shape[0]):
# 		img = image_arr[i,:]
# 		image_arr[i,:] = 255*(img - np.min(img))/(np.max(img)-np.min(img))
	return image_arr

def save_overlap_csv(file_name, dic_sim = {}):
	import csv
	file_name = '{}.csv'.format(file_name)
# 	file_name = os.path.join(file_folder, 'overlap_analysis.csv')
	if dic_sim == {}:
		return
	else:
		with open(file_name, 'w') as csvfile:
			(key, values) = dic_sim.popitem()
			# create the header in the table
			fieldnames = ['image_index', 'average']
			row_dic = {}
			row_dic[fieldnames[0]]= key
			row_dic[fieldnames[1]]= values[-1]
			for i in range(len(values)-1):
				fieldnames.append(i)
				row_dic[i] = values[i]
			writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
			writer.writeheader()
			writer.writerow(row_dic)
			while (dic_sim != {}):
				(key, values) = dic_sim.popitem()
				for i in range(len(values)-1):
					row_dic[fieldnames[0]]= key
					row_dic[fieldnames[1]]= values[-1]
					row_dic[i] = values[i]
				writer.writerow(row_dic) 
