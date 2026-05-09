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

global_train_batch_size=32
per_device_train_batch_size=4
gradient_accumulation_steps=$((global_train_batch_size / per_device_train_batch_size / NUM_GPUS))

model=Llama-3-8B-Instruct
INITIAL_CHECKPOINT=${INITIAL_CHECKPOINT:-"meta-llama/Meta-Llama-3-8B-Instruct"}

experiments=(
    # PCGrad
    "Batch50_PCGrad_global"
    "Batch50_PCGrad"
    "Batch50_PCGrad_NPO"
    "Batch50_PCGrad_SimNPO"

    # SAGO
    "Batch50_SAGO"
    "Batch50_SAGO_NPO"
    "Batch50_SAGO_SimNPO"
)

for experiment in "${experiments[@]}"; do
    echo "Running RWKU GradSurgery experiment: ${experiment}"

    task_name="rwku_${experiment}"

    accelerate launch --config_file configs/accelerate/default_config.yaml \
        --main_process_port $MASTER_PORT \
        --num_processes=${NUM_GPUS} \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/rwku/${experiment}.yaml \
        paths.root_dir=${ROOT_DIR} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=${INITIAL_CHECKPOINT} \
        model.tokenizer_args.pretrained_model_name_or_path=${INITIAL_CHECKPOINT} \
        trainer.args.eval_strategy='no' \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        task_name=${task_name}

    CURRENT_MODEL=${ROOT_DIR}/saves/unlearn/${task_name}

    target="Batch/1-50"

    CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1) python src/eval.py \
        experiment=eval/rwku/default.yaml \
        paths.root_dir=${ROOT_DIR} \
        task_name=${task_name} \
        target=${target} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=${CURRENT_MODEL} \
        model.tokenizer_args.pretrained_model_name_or_path=${CURRENT_MODEL} \
        eval.rwku.overwrite=true \
        paths.output_dir=saves/rwku/${task_name}

    echo "Completed experiment: $experiment"
    echo "----------------------------------------"
done
