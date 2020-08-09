import numpy as np
from scipy.stats import pearsonr
from skimage.measure import compare_ssim as ssim


## intersection over union
# input: 
def IoU_calculate(image1, image2, bk_include = False):
	img1_vec = image1.reshape((-1,))
	img2_vec = image2.reshape((-1,))
	interArea = img1_vec * img2_vec
	unionArea = img1_vec + img2_vec
	IoU = np.count_nonzero(interArea)/np.count_nonzero(unionArea)
	return IoU

def density_Cosine_calculate(image1, image2, thres_ratio = 0.0 ,bk_include = False):
	img1_vec = image1.reshape((-1,))
	img2_vec = image2.reshape((-1,))
	img1_vec = img1_vec*(img1_vec>np.max(img1_vec)*thres_ratio)
	img2_vec = img2_vec*(img2_vec>np.max(img2_vec)*thres_ratio)
	interEnergy = np.sum(img1_vec * img2_vec)
	unionEnergy = np.sqrt(np.sum(img1_vec**2)*np.sum(img2_vec**2))
	if unionEnergy == 0:
		IOU = 0
	else:
		IoU = interEnergy/unionEnergy
	return IoU

## compute the cosine as a function of threshold for two arrays of density maps
def density_Cosine_calculate_for_array(imageArr1, imageArr2, threshold = 0):
	shp = imageArr1.shape
	cosin_list = []
	for i in range(shp[0]):
		image1 = imageArr1[i]
		image2 = imageArr2[i]
		img1_vec = image1.reshape((-1,))
		img2_vec = image2.reshape((-1,))
		img1_vec = img1_vec*(img1_vec>threshold)
		img2_vec = img2_vec*(img2_vec>threshold)
		interEnergy = np.sum(img1_vec * img2_vec)
		unionEnergy = np.sqrt(np.sum(img1_vec**2)*np.sum(img2_vec**2))
		if unionEnergy ==0:
			cosin_ = 0
		else:
			cosin_ = interEnergy/unionEnergy
		cosin_list.append(cosin_)
	return cosin_list, np.mean(cosin_list)

def density_SSIM_calculate(image1, image2, thres_ratio = 0.0):
	image_1 = image1.astype(np.float64)
	image_2 = image2.astype(np.float64)
	image_1 = image_1*(image_1>np.max(image_1)*thres_ratio)
	image_2 = image_2*(image_2>np.max(image_2)*thres_ratio)
	return ssim(image_1, image_2, data_range = image_2.max()-image_2.min())
	
def density_Manhattan_calculate(image1, image2):
	dif = np.sum(np.abs(image1-image2))
	return 1-dif/(np.sum(image1)+np.sum(image2))
	
def density_Euclidean_calculate(image1, image2):
	img1_vec = image1.reshape((-1,))
	img2_vec = image2.reshape((-1,))
	dif = np.sqrt(np.sum((img1_vec-img2_vec)**2))
	return 1-dif/(np.sqrt(np.sum(img1_vec**2))+np.sqrt(np.sum(img2_vec**2)))
	
def density_Tanimoto_calculate(image1, image2, thres_ratio = 0.0):
	img1_vec = image1.reshape((-1,))
	img2_vec = image2.reshape((-1,))
	img1_vec = img1_vec*(img1_vec>np.max(img1_vec)*thres_ratio)
	img2_vec = img2_vec*(img2_vec>np.max(img2_vec)*thres_ratio)
	interEnergy = np.sum(img1_vec*img2_vec)
	denumerator = np.sum((img1_vec-img2_vec)**2)+interEnergy
	return interEnergy*1.0/denumerator

def density_Pearsonr_calculate(image1, image2, thres_ratio = 0.0):
	img1_vec = image1.reshape((-1,))
	img2_vec = image2.reshape((-1,))
	img1_vec = img1_vec*(img1_vec>np.max(img1_vec)*thres_ratio)
	img2_vec = img2_vec*(img2_vec>np.max(img2_vec)*thres_ratio)
	corr, p_value = pearsonr(img1_vec, img2_vec)
	return corr

def density_Pearsonr_calculate_for_array(imageArr1, imageArr2, threshold = 0):
	over_list = []
	for i in range(imageArr1.shape[0]):
		over_list.append(density_Cosine_calculate(imageArr1[i],imageArr2[i],threshold))
	return over_list, np.mean(over_list)