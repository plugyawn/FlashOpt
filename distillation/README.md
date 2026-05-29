# Distillation

Scripts for:
- generating distillation data (`distill_data_gen.py`)
- running SFT (`distill_sft.sh`)

## Quick Start

1. Generate distillation data from the top_k_seeds.json saved during training:
   ```bash
   python distill_data_gen.py --seeds_file /path/to/top_k_seeds.json
   ```
2. Launch SFT training:
   ```bash
   sbatch distill_sft.sh
   ```
   or
   ```bash
   cd ../baselines
   python -m torch.distributed.run --nproc_per_node=4 \
     -m verl.trainer.fsdp_sft_trainer \
     data.train_files=distill_data/sft/train.parquet \
     data.prompt_key=prompt \
     data.response_key=response \
     data.max_length=1024 \
     data.truncation=right \
     data.train_batch_size=40 \
     data.micro_batch_size_per_gpu=1 \
     optim.lr=1e-4 \
     optim.lr_warmup_steps_ratio=0.1 \
     model.partial_pretrain=Qwen/Qwen2.5-3B-Instruct \
     trainer.default_local_dir=distill_checkpoints \
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
     +trainer.checkpoint.hf_model_path=distill_checkpoints/hf_model
   ```