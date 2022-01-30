import numpy as np
import cv2 as cv
import os
import scipy.io
from skimage import io

def gen_dir(folder):
    if not os.path.exists(folder):
        os.system('mkdir -p {}'.format(folder))

def get_cordinates(file_name, cell_type):
    set1 = ['dapi', 'T', 'sox2']
    set2 = ['cdx2', 'sox17']
    xs, ys = [], []
    with open(file_name, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            segs = line.split(',')
            if segs[0] == '#': continue
            else:
                if cell_type in set1:
                    xs.append(int(float(segs[2])))
                    ys.append(int(float(segs[3])))
                else:
                    xs.append(int(float(segs[1])))
                    ys.append(int(float(segs[2])))

    return np.array(xs), np.array(ys)

import csv
def get_cordinates2(file_name, cell_type):
    group = ['T', 'sox2']
    xs, ys = [], []
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if cell_type in group:
                xs.append(int(float(row['x'])))
                ys.append(int(float(row['y'])))
            else:
                xs.append(int(float(row['y'])))
                ys.append(int(float(row['x'])))      
    return xs, ys

def denormalize(img, lower, upper):
    lower = np.percentile(img, lower)
    upper = np.percentile(img, upper)
    img = np.clip(img, lower, upper)
    return np.uint8( (img - lower) * 255 / (upper - lower))
                

data_root = os.path.abspath('../')

# output folders
dataset = 'hESC'
dataset_dir = data_root + '/datasets/{}'.format(dataset)
img_output_dir = dataset_dir + '/images/'
dot_output_dir = dataset_dir + '/dots/'
gen_dir(img_output_dir)
gen_dir(dot_output_dir)

# input folders
input_dir = data_root + '/datasets/annotated_hESC'
img_fns = sorted([fn for fn in os.listdir(input_dir) if fn.endswith('Count')])

group1 = ['cdx2', 'sox17']
group2 = ['T', 'sox2']

for img_fn in img_fns:
    cell_type = img_fn.split('Manual')[0]
    if cell_type in group1:
        input_img_dir = input_dir + '/{}/{}'.format(img_fn, cell_type)
        input_dot_dir = input_dir + '/{}/{}_annotated'.format(img_fn, cell_type)
        img_names = sorted([img_name for img_name in os.listdir(input_img_dir) if img_name.endswith('.tif')])
        print(len(img_names))
        for i, img_name in enumerate(img_names):
            img = io.imread(input_img_dir + '/' + img_name)
            img = denormalize(img, 0.1, 99.9)
            img_tag = img_name.split('.')[0].split('_')[-1]
            xs, ys = get_cordinates2(input_dot_dir + '/{}.csv'.format(img_tag), cell_type)
            H, W = img.shape
            dot_map = np.zeros((H, W), dtype = np.uint8)
            dot_map[ys, xs] = 255
            cv.imwrite(img_output_dir + '/{}_{}.png'.format(cell_type, img_tag), img)
            cv.imwrite(dot_output_dir + '/{}_{}.png'.format(cell_type, img_tag), dot_map)
        save_img = cv.hconcat([img, dot_map])
        cv.imwrite(res_dir + '/com_{}.png'.format(cell_type), save_img)
    elif cell_type in group2:
        input_img_dir = input_dir + '/{}/{}'.format(img_fn, cell_type)
        input_dot_dir = input_dir + '/{}/{}_annotated'.format(img_fn, cell_type)
        img_names = sorted([img_name for img_name in os.listdir(input_img_dir) if img_name.endswith('.tif')])
        print(len(img_names))
        for i, img_name in enumerate(img_names):
            img = io.imread(input_img_dir + '/' + img_name)
            img = denormalize(img, 0.1, 99.9)
            img_tag = img_name.split('.')[0]
            xs, ys = get_cordinates2(input_dot_dir + '/s{}.csv'.format(img_tag), cell_type)
            H, W = img.shape
            dot_map = np.zeros((H, W), dtype = np.uint8)
            dot_map[ys, xs] = 255
            cv.imwrite(img_output_dir + '/{}_{}.png'.format(cell_type, img_tag), img)
            cv.imwrite(dot_output_dir + '/{}_{}.png'.format(cell_type, img_tag), dot_map)
        save_img = cv.hconcat([img, dot_map])
        cv.imwrite(res_dir + '/com_{}.png'.format(cell_type), save_img)       
    else:
        input_img_dir = input_dir + '/{}/{}'.format(img_fn, cell_type)
        input_dot_dir = input_dir + '/{}/{}_annotated'.format(img_fn, cell_type)
        img_names = sorted([img_name for img_name in os.listdir(input_img_dir) if img_name.endswith('.tif')])
        print(len(img_names))
        for i, img_name in enumerate(img_names):
            img = io.imread(input_img_dir + '/' + img_name)
            img = denormalize(img, 0.1, 99.9)
            img_tag = img_name.split('.')[0]
            xs, ys = get_cordinates2(input_dot_dir + '/{}.csv'.format(img_tag), cell_type)
            H, W = img.shape
            dot_map = np.zeros((H, W), dtype = np.uint8)
            dot_map[ys, xs] = 255
            cv.imwrite(img_output_dir + '/{}_{}.png'.format(cell_type, img_tag), img)
            cv.imwrite(dot_output_dir + '/{}_{}.png'.format(cell_type, img_tag), dot_map)
        save_img = cv.hconcat([img, dot_map])
        cv.imwrite(res_dir + '/com_{}.png'.format(cell_type), save_img)




