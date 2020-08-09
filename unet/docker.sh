chcon -Rt svirt_sandbox_file_t /shared2/Data_FDA_Breast/Segmentation
docker run --gpus 0 -v /shared2/Data_FDA_Breast/Segmentation:/data -w /data/segmentation_models/phase_cells -it --user $(id -u):$(id -g) shenghh2020/tf_gpu_py3.5:2.0 bash
