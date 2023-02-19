sudo docker run --name l4t-ml -it \
--rm --runtime nvidia --network host \
-e PATH=/usr/src/tensorrt/bin/:$PATH -e PATH=/usr/local/cuda/bin:$PATH \
-v ~/mnt-dir:/root/mount-dir nvcr.io/nvidia/l4t-ml:r32.7.1-py3
