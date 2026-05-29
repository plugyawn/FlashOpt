#!/bin/bash
#SBATCH --job-name=distill-sft
#SBATCH --nodes=1
#SBATCH --account=<your_account>
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=logs_distill/%j-distill-sft.out
#SBATCH --error=logs_distill/%j-distill-sft.err

# Configuration
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set. Please export your Hugging Face token first."
    exit 1
fi

# Paths
DATA_DIR="distill_data/sft"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BASELINES_ROOT="${BASELINES_ROOT:-$(cd "${SCRIPT_DIR}/../baselines" && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASELINES_ROOT}"
DATA_ROOT="${DATA_ROOT:-$BASELINES_ROOT}"
VERL_DIR="${VERL_DIR:-$BASELINES_ROOT/verl}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASELINES_ROOT/distill_checkpoints}"
MODEL="Qwen/Qwen2.5-3B-Instruct"
NUM_GPUS=4

NNODES=$SLURM_NNODES
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

mkdir -p logs_distill

srun --cpu-bind=none -ul --container-writable bash -c "
    export PYTHONNOUSERSITE=1
    export HF_HOME=/tmp/$USER/huggingface
    export HF_TOKEN=$HF_TOKEN
    huggingface-cli login --token \$HF_TOKEN
    unset ROCR_VISIBLE_DEVICES

    cd $BASELINES_ROOT
    cd $VERL_DIR
    
    export PYTHONPATH=$BASELINES_ROOT:\$PYTHONPATH
    
    # Check if data exists
    if [ ! -f \"$DATA_ROOT/${DATA_DIR}/train_dedup.parquet\" ]; then
        echo \"Error: ${DATA_DIR}/train_dedup.parquet not found\"
        exit 1
    fi
    
    PYTHONUNBUFFERED=1 torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NUM_GPUS \
        --rdzv_id=\$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files=$DATA_ROOT/${DATA_DIR}/train.parquet \
        data.prompt_key=prompt \
        data.response_key=response \
        data.max_length=1024 \
        data.truncation=right \
        optim.lr=1e-4 \
        optim.lr_warmup_steps_ratio=0.1 \
        data.train_batch_size=40 \
        data.micro_batch_size_per_gpu=1 \
        model.partial_pretrain=$MODEL \
        trainer.default_local_dir=$OUTPUT_DIR \
        trainer.project_name=distill-sft \
        trainer.experiment_name=distill-sft-gsm8k-lora \
        trainer.logger=console \
        trainer.total_epochs=10 \
        trainer.save_freq=5 \
        trainer.test_freq=5 \
        +model.attn_implementation=sdpa \
        +model.lora.enable=true \
        +model.lora.rank=64 \
        +model.lora.alpha=128 \
        +model.lora.dropout=0.05 \
        '+model.lora.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
        '+trainer.checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
        +trainer.checkpoint.hf_model_path=$OUTPUT_DIR/hf_model \
"
