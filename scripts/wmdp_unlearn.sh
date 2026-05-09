#!/bin/bash
set -e

# Configuration - adjust these for your environment
ROOT_DIR=${ROOT_DIR:-$(pwd)}
NUM_GPUS=${NUM_GPUS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# Optional: Uncomment and set these for W&B logging
# export WANDB_PROJECT="sago"
# export WANDB_API_KEY="your-key-here"

# Optional: Uncomment if you need a HuggingFace mirror
# export HF_ENDPOINT="https://hf-mirror.com"

#########################################################
#################### WMDP Evaluation ####################
#########################################################

mmlu(){
    MODEL=$1
    BATCH_SIZE=${2:-8}
    accelerate launch --main_process_port $MASTER_PORT -m lm_eval --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16,trust_remote_code=True" \
    --tasks mmlu \
    --cache_requests true \
    --num_fewshot 5 \
    --batch_size $BATCH_SIZE \
    --log_samples \
    --output_path saves/lm_eval/mmlu
}

wmdp_cyber(){
    MODEL=$1
    BATCH_SIZE=${2:-8}
    accelerate launch --main_process_port $MASTER_PORT -m lm_eval --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16,trust_remote_code=True" \
    --tasks wmdp_cyber \
    --cache_requests true \
    --batch_size $BATCH_SIZE \
    --log_samples \
    --output_path saves/lm_eval/wmdp_cyber
}

wmdp_bio(){
    MODEL=$1
    BATCH_SIZE=${2:-8}
    accelerate launch --main_process_port $MASTER_PORT -m lm_eval --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16,trust_remote_code=True" \
    --tasks wmdp_bio \
    --cache_requests true \
    --batch_size $BATCH_SIZE \
    --log_samples \
    --output_path saves/lm_eval/wmdp_bio
}

#########################################################
#################### WMDP SAGO/PCGrad ###################
#########################################################

global_train_batch_size=32
per_device_train_batch_size=2
gradient_accumulation_steps=$((global_train_batch_size / per_device_train_batch_size / NUM_GPUS))

model=zephyr-7b-beta
INITIAL_CHECKPOINT=${INITIAL_CHECKPOINT:-"HuggingFaceH4/zephyr-7b-beta"}

experiments=(
    # PCGrad
    "GradDiff_pcgrad_global_cyber"
    "GradDiff_pcgrad_global_bio"
    "GradDiff_pcgrad_cyber"
    "GradDiff_pcgrad_bio"
    "NPO_pcgrad_cyber"
    "NPO_pcgrad_bio"
    "SimNPO_pcgrad_cyber"
    "SimNPO_pcgrad_bio"

    # SAGO
    "GradDiff_sago_cyber"
    "GradDiff_sago_bio"
    "NPO_sago_cyber"
    "NPO_sago_bio"
    "SimNPO_sago_cyber"
    "SimNPO_sago_bio"
)

for experiment in "${experiments[@]}"; do
    echo "Running WMDP GradSurgery experiment: $experiment"

    if [[ $experiment == *cyber* ]]; then
        data_split="cyber"
    elif [[ $experiment == *bio* ]]; then
        data_split="bio"
    fi

    task_name="wmdp_${experiment}"

    accelerate launch --config_file configs/accelerate/multi_gpu.yaml \
        --main_process_port $MASTER_PORT \
        --num_processes=${NUM_GPUS} \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/wmdp/${experiment}.yaml \
        paths.root_dir=${ROOT_DIR} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=${INITIAL_CHECKPOINT} \
        model.tokenizer_args.pretrained_model_name_or_path=${INITIAL_CHECKPOINT} \
        trainer.args.eval_strategy='no' \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=false \
        task_name=${task_name}

    CURRENT_MODEL=${ROOT_DIR}/saves/unlearn/${task_name}

    echo "Evaluating model for experiment: $experiment"
    mmlu ${CURRENT_MODEL} ${per_device_train_batch_size}
    if [ "$data_split" == "cyber" ]; then
        wmdp_cyber ${CURRENT_MODEL} ${per_device_train_batch_size}
    elif [ "$data_split" == "bio" ]; then
        wmdp_bio ${CURRENT_MODEL} ${per_device_train_batch_size}
    fi

    echo "Completed experiment: $experiment"
    echo "----------------------------------------"
done
