import os
import re


model_root_folder = 'models/paper_models_w_cv'

log_file_name = 'eval.log'

f = open(log_file_name,'r');
lines = f.readlines();

value_list = []
weight_file_list = []
for line in lines:
	splits = re.split(':| ',line.strip('\n'))
	if splits[0] == 'Model':
		print(splits[1])
		folder_name = os.path.join(model_root_folder,splits[1])
	if len(splits) == 3:
		print('{}-{}-{}'.format(splits[0], splits[1], splits[2]))
		value_list.append(float(splits[1]))
		weight_file_list.append(splits[2])

## delete unrelated file
threshold = 2.7
for i, value in enumerate(value_list):
	if value < threshold:
		print(weight_file_list[i])
		os.system('rm {}'.format(os.path.join(folder_name,weight_file_list[i])))