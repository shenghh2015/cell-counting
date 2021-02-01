#export LSF_DOCKER_VOLUMES='/scratch1/fs1/anastasio/Data_FDA_Breast:/scratch/xray_set'
export LSF_DOCKER_VOLUMES='/scratch1/fs1/anastasio/Data_FDA_Breast/phase_cells:/data'
export LSF_DOCKER_NETWORK=host
export LSF_DOCKER_IPC=host 
export LSF_DOCKER_SHM_SIZE=40G
bsub -G compute-anastasio -n 4 -R 'span[ptile=4] select[mem>200000] rusage[mem=200GB]' -q general -a 'docker(shenghh2020/tf_gpu_py3.5:2.0)' -gpu "num=4" -o /scratch1/fs1/anastasio/Data_FDA_Breast/phase_cells/logs/4GPU_$RANDOM /bin/bash /home/shenghuahe/segmentation_models/phase_cells/multi_loader.sh