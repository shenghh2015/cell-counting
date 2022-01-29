import numpy as np

# normalize the pixel value to the range [0,1]
def data_normalize(dataSet):
	shp = dataSet.shape
	norm_set = []
	for i in range(shp[0]):
		image = dataSet[i,:,:]
		max_val = np.max(image)
		min_val = np.min(image)
		if not max_val == min_val:
			image = (image-min_val)/(max_val-min_val)
		norm_set.append(image)
	im_arr = np.array(norm_set)
	return im_arr

## transfer the gray images to the RGB images
def transfer2RGB(image, channel = 0):
	shp = image.shape
	rgb_shp = shp +(3,)
	rgbImg = np.zeros(rgb_shp, dtype = np.uint8)
	tmpImg = 255.0*(image -np.min(image))/(np.max(image)-np.min(image)) 
	rgbImg[:,:,channel] = tmpImg.astype(np.uint8)
	return rgbImg

## transfer the gray images to a RGB gray image
def transfer2RGB_v2(image):
	shp = image.shape
	rgb_shp = shp +(3,)
	rgbImg = np.zeros(rgb_shp, dtype = np.uint8)
	tmpImg = 255.0*(image -np.min(image))/(np.max(image)-np.min(image)) 
	rgbImg[:,:,0] = tmpImg.astype(np.uint8)
	rgbImg[:,:,1] = tmpImg.astype(np.uint8)
	rgbImg[:,:,2] = tmpImg.astype(np.uint8)
	return rgbImg

def transfer2RGBArr(imageArr, channel = 0):
	shp = imageArr.shape
	rgb_shp = shp + (3,)
	rgbArr = np.zeros(rgb_shp, dtype = np.uint8)
	imageArr_ = np.copy(imageArr)
	for i in range(len(imageArr)):
		image = imageArr_[i,:,:]
		tmpImg = 255.0*(image -np.min(image))/(np.max(image)-np.min(image)) 
		rgbArr[i,:,:,channel] =  tmpImg.astype(np.uint8)
	return rgbArr