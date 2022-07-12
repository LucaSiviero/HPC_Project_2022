#! /bin/sh

docker run -it -v /home/luca/HPC_project_2022/project:/root/project -v/opt/intel/oneapi:/opt/intel/oneapi --gpus=all luigicrisci/sycl_cuda11.4
