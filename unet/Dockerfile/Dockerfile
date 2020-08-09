#FROM tensorflow/tensorflow:1.13.0rc1-gpu-py3
FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends python3-pip python3-tk python2.7 git

RUN ln -s /usr/bin/python2.7 /usr/bin/python2

RUN pip install scipy sklearn matplotlib natsort ipython 

RUN pip install jupyterlab

RUN pip install keras keras_applications>=1.0.7 image-classifiers==1.0.0 efficientnet==1.0.0

RUN pip install segmentation-models==1.0.1
RUN pip install albumentations==0.3.0


WORKDIR /data

## 
#  1. build up a docker container
#  docker build . < Dockerfile -t shenghh2020/tf_gpu_py3.5:2.0
#  2. push the docker container to the docker hub
#  docker push shenghh2020/tf_gpu_py3.5:2.0 
#  docker push shenghh2020/tf_gpu_py3.5:1.0
#  3. qsub a job to the v100_cluster
#  bsub -Is -G compute-anastasio -q anastasio-interactive -a 'docker(shenghh2020/tf_gpu_py3.5:latest)' -gpu "num=4" /bin/bash
#  when the access permission is required, use the following command:
#
# docker login -u "myusername" -p "mypassword" docker.io
# docker push myusername/myimage:0.0.1
##
