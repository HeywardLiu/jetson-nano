precision=fp32   # precision = [fp32, fp16, int8]
root_dir=/root/mount-dir
trt_dir=$root_dir/models/trt-model/eval
log_dir=$root_dir/trt-inference/trt-acc/$precision
mkdir -p $log_dir
mkdir -p $trt_dir 

for trt_model in "$trt_dir"/*
do
    model_name=${precision}-$(basename "${trt_model%.*}")
    log_file="$log_dir/acc-${model_name}.txt" 

    if [ -f "$log_file" ];
    then 
        echo "Found existing evaluation result: $log_file"
        continue
    fi
    echo "------------------------------"
    echo "          model: $model_name"
    echo "trt engine path: $trt_model"

    echo "----------------------------------------------" > "$log_file"
    echo "$model_name" >> "$log_file"
    echo "----------------------------------------------" >> "$log_file"
    # Evaluate model's accuracy
    python3 eval_gt.py -i ~/mount-dir/.imagenet-val -a ~/mount-dir/imagenet-val-annotations.txt -e $trt_model >> "$log_file"

    echo "Complete Evaluation."
    echo "------------------------------"
done