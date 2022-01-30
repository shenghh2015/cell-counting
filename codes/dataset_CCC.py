import numpy as np
import cv2 as cv
import os
import scipy.io

def gen_dir(folder):
    if not os.path.exists(folder):
        os.system('mkdir -p {}'.format(folder))

data_root = os.path.abspath('./')

dataset = 'CCC'
dataset_dir = data_root + '/datasets/{}'.format(dataset)
img_output_dir = dataset_dir + '/images/'
dot_output_dir = dataset_dir + '/dots/'
gen_dir(img_output_dir)
gen_dir(dot_output_dir)

img_dir = data_root + '/datasets/CRCHistoPhenotypes_2016_04_28/Detection'

img_fns = sorted([fn for fn in os.listdir(img_dir) if fn.startswith('img')])
# img_fn = img_fns[0]

for i, img_fn in enumerate(img_fns):
    img = cv.imread(img_dir + '/' + img_fn + '/{}.bmp'.format(img_fn))
    dot = scipy.io.loadmat(img_dir + '/' + img_fn + '/{}_detection.mat'.format(img_fn))
    dot = dot['detection']
    xs, ys = np.int16(dot[:,0] - 1), np.int16(dot[:, 1] - 1)
    dot_map = np.zeros((H, W), dtype = np.uint8)
    dot_map[ys, xs] = 255
    cv.imwrite(img_output_dir + '/{}.png'.format(img_fn), img)
    cv.imwrite(dot_output_dir + '/{}.png'.format(img_fn), dot_map)

# res_dir = data_root + '/result/'
# gen_dir(res_dir)
# cv.imwrite(res_dir + '/img_{}.png'.format(img_fn), img)
# cv.imwrite(res_dir + '/dot_{}.png'.format(img_fn), dot_map)
# dot_rgb = np.stack([dot_map, dot_map, dot_map], axis = -1)
# save_img = cv.hconcat([img, dot_rgb])
# cv.imwrite(res_dir + '/com_{}.png'.format(img_fn), save_img)




