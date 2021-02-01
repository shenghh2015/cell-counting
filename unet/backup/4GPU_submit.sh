export LSF_DOCKER_VOLUMES='/scratch1/fs1/anastasio/Data_FDA_Breast/phase_cells:/data'
export LSF_DOCKER_NETWORK=host
export LSF_DOCKER_IPC=host 
export LSF_DOCKER_SHM_SIZE=40G
bsub -G compute-anastasio -n 8 -R 'span[ptile=8] select[mem>40000] rusage[mem=40GB]' -q anastasio -a 'docker(shenghh2020/tf_gpu_py3.5:2.0)' -gpu "num=4" -o /scratch1/fs1/anastasio/Data_FDA_Breast/phase_cells/logs/4GPU_Task$RANDOM /bin/bash /home/shenghuahe/segmentation_models/phase_cells/4GPU_task.sh