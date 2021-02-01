import os
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("job_list_file", type=str)

args = parser.parse_args()
job_list_file = args.job_list_file

job_folder = './job_folder'
# job_list_file = './DA_jobs.txt'

# clean the job folder
for file in glob.glob(job_folder+'/*'):
	os.system('rm -f {}'.format(file))
	print('Remove {}'.format(os.path.basename(file)))

job_list = []
with open(job_list_file, 'rb') as f:
	lines = f.readlines()
	for line in lines:
		splits = line.strip().split(' ')
		if len(splits) > 0:
			if splits[0] == 'JOB:':
				#print(line.replace('JOB: ', ''))
				job_list.append(line.replace('JOB: ', ''))

for i in range(len(job_list)):
	job_file = os.path.join(job_folder, 'job_{}.sh'.format(i))
	with open(job_file, 'w+') as f:
		f.write('{}\n'.format(job_list[i]))
	print('Add job: {}'.format(job_list[i]))
