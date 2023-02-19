# precision = [fp32, fp16, int8]
precision=fp32
root_dir=/root/mount-dir
onnx_dir=$root_dir/models/onnx-model
trt_dir=$root_dir/models/trt-model/$precision
log_dir=$root_dir/log-onnx2trt/$precision
mkdir -p $trt_dir 
mkdir -p $log_dir
for onnx_model in "$onnx_dir"/*
do
    model_name=${precision}-$(basename "${onnx_model%.*}")
    trt_model=$trt_dir/$model_name.trt

    if [ -f "$trt_model" ];
    then 
        echo "Found existing trt-model: $trt_model"
        continue
    fi
    echo "------------------------------"
    echo "   model: $model_name"
    echo "onnx-dir: $onnx_model"
    echo " trt-dir: $trt_model"

    log_file="$log_dir/${model_name}_trtexec_log.txt" 
    echo "----------------------------------------------" > "$log_file"
    echo "$model_name" >> "$log_file"
    echo "----------------------------------------------" >> "$log_file"
    # Use trtexec to convert onnx to tensorrt
    if [ "$precision" = "fp32" ];
    then
        /usr/src/tensorrt/bin/trtexec --onnx=$onnx_model --saveEngine=$trt_model --workspace=2560 --verbose >> "$log_file"
    else 
        /usr/src/tensorrt/bin/trtexec --onnx=$onnx_model --saveEngine=$trt_model --$precision --workspace=2560 --verbose >> "$log_file"
    fi
    echo "------------------------------"
done