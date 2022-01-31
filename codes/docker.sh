chcon -Rt svirt_sandbox_file_t ~/cell-counting/
docker run --gpus 0 -v ~/cell-counting:/data -w /data/ -it --user $(id -u):$(id -g) shenghh2020/tf_gpu_py3.5:2.0 bash