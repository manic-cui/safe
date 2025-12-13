
GPU_NUM=4
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12588

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

train_datasets=(
    "/data/mannicui/aigi-detection/CNNDetection/train" \
)
eval_datasets=(
    "/data/mannicui/aigi-detection/CNNDetection/val" \
)

MODEL="SAFE"

USE_WANDB=${USE_WANDB:-true}
WANDB_PROJECT=${WANDB_PROJECT:-SAFE}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_GROUP=${WANDB_GROUP:-SAFE}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_TAGS=${WANDB_TAGS:-train,$MODEL}

for train_dataset in "${train_datasets[@]}" 
do
    for eval_dataset in "${eval_datasets[@]}" 
    do

        current_time=$(date +"%Y%m%d_%H%M%S")
        OUTPUT_PATH="results/$MODEL/$current_time"
        mkdir -p $OUTPUT_PATH

        train_name=$(basename "$train_dataset")
        eval_name=$(basename "$eval_dataset")
        run_name="${MODEL}_${current_time}_${train_name}_${eval_name}"
        wandb_notes="train:${train_name} eval:${eval_name}"

        python -m torch.distributed.launch $DISTRIBUTED_ARGS main_finetune.py \
            --input_size 256 \
            --transform_mode 'crop' \
            --model $MODEL \
            --data_path "$train_dataset" \
            --eval_data_path "$eval_dataset" \
            --save_ckpt_freq 1 \
            --batch_size 32 \
            --blr 1e-2 \
            --weight_decay 0.01 \
            --warmup_epochs 1 \
            --epochs 20 \
            --num_workers 16 \
            --output_dir $OUTPUT_PATH \
            --use_wandb $USE_WANDB \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_entity "$WANDB_ENTITY" \
            --wandb_group "$WANDB_GROUP" \
            --wandb_run_name "$run_name" \
            --wandb_notes "$wandb_notes" \
            --wandb_tags "$WANDB_TAGS" \
            --wandb_mode "$WANDB_MODE" \
        2>&1 | tee -a $OUTPUT_PATH/log_train.txt

    done
done