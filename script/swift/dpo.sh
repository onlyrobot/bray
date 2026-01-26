export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NNODES=${NNODES:-2}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-0.0.0.0}
export MASTER_PORT=${MASTER_PORT:-8510}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}

declare -A MAP_DIST_PARAMS=(
    ["Turn-Off"]=""
    ["Zero-1"]="zero1"
    ["Zero-2"]="zero2"
    ["Zero-3"]="zero3"
    ["Offload"]="zero3_offload"
)
DEEPSPEED=${MAP_DIST_PARAMS[$DIST_DEEPSPEED_STAGE]}

swift rlhf --rlhf_type dpo \
    --train_type ${DIST_TRAIN_TYPE,,} \
    --model $DIST_MODEL \
    --dataset $TRAIN_DATASET \
    ${VAL_DATASET:+--val_dataset} \
    ${VAL_DATASET:---split_dataset_ratio $DIST_DATASET_SPLIT} \
    --torch_dtype bfloat16 \
    --max_completion_length $DIST_MAX_SEQ_LEN \
    --num_train_epochs $DIST_EPOCH \
    --per_device_train_batch_size $DIST_BATCH_SIZE \
    --per_device_eval_batch_size $DIST_BATCH_SIZE \
    --learning_rate $DIST_LR_RATE \
    --gradient_accumulation_steps $DIST_GRAD_ACCUM \
    --eval_steps $DIST_VAL_STEP \
    --save_steps $DIST_SAVE_STEP \
    --save_total_limit ${DIST_SAVE_LIMIT:-100} \
    --logging_steps $DIST_LOG_STEP \
    --max_length $DIST_MAX_SEQ_LEN \
    --output_dir ${OUTPUT_DIR:-$DIST_TRIAL_PATH/output} \
    --add_version false \
    --warmup_ratio ${DIST_WARMUP_RATIO:-0.05} \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --system "$DIST_SYSTEM_PROMPT" \
    ${DEEPSPEED:+--deepspeed $DEEPSPEED} \
    ${DIST_RESUME_PATH:+--resume_from_checkpoint $DIST_RESUME_PATH} \
    --log_completions true "$@"