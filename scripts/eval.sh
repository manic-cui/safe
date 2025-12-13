GPU_NUM=4
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12345

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MODEL="SAFE"
# 原始的 checkpoint 路径
RESUME_PATH="/home/mannicui/SAFE/results/SAFE/20251212_210956"
# 你想用的 checkpoint 文件
RESUME_FILE="$RESUME_PATH/checkpoint-best.pth"

eval_datasets=(
    "/data/mannicui/aigi-detection/CNNDetection/test" \
    "/data/mannicui/aigi-detection/Chameleon/test" \
    "/data/mannicui/aigi-detection/WildRF/test" \
)

# === 修改点：定义要测试的 JPEG 质量列表 ===
# "None" 表示不压缩（原始性能），然后依次测试 95, 90, 80, 70, 60, 50
# 这些是学术界常用的测试节点
jpeg_qualities=("None" 95 90 80 70 60 50)

for quality in "${jpeg_qualities[@]}"
do
    echo "--------------------------------------------------------"
    echo "Testing with JPEG Quality: $quality"
    echo "--------------------------------------------------------"

    # 根据是否压缩，设置参数和输出目录
    if [ "$quality" == "None" ]; then
        JPEG_ARG=""
        # 结果保存在 checkpoint/eval_clean 下
        OUTPUT_DIR="$RESUME_PATH/eval_clean"
    else
        JPEG_ARG="--jpeg_factor $quality"
        # 结果保存在 checkpoint/eval_jpeg_90 下
        OUTPUT_DIR="$RESUME_PATH/eval_jpeg_$quality"
    fi

    # 创建输出目录
    mkdir -p $OUTPUT_DIR

    for eval_dataset in "${eval_datasets[@]}"
    do
        # 打印正在测试的数据集
        echo "Processing $eval_dataset..."
        
        python -m torch.distributed.launch $DISTRIBUTED_ARGS main_finetune.py \
            --input_size 256 \
            --transform_mode 'crop' \
            --model $MODEL \
            --eval_data_path $eval_dataset \
            --batch_size 256 \
            --num_workers 16 \
            --output_dir $OUTPUT_DIR \
            --resume $RESUME_FILE \
            --eval True \
            $JPEG_ARG 
            
    done
done