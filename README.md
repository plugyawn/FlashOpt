# RandOpt
**Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights**

[Yulu Gan](https://yulugan.com), [Phillip Isola](https://web.mit.edu/phillipi/)

[Paper](https://arxiv.org/pdf/2603.12228)          |         [Project Page](https://thickets.mit.edu)    |        Starting with a 1D Experiment: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SsBrfQ-iFKuGElWjTNiFoX4dtMaCzCGy?usp=sharing)


## ⚡ Speedrun (hypercharged runtime)

Since RandOpt's cost is *seeds evaluated per second*, this fork adds a fast,
principled runtime and a fixed standard to race the clock — nanoGPT-speedrun style.

- **Fused dense-Rademacher switching** (`core/perturb.py`): `W = W₀ + σ·R(seed)`
  reconstructed in a single Triton pass from a resident base copy. No noise tensor
  materialized, no restore pass, no per-call cache churn — **2 full-model RNG
  materializations per seed → 0**. Drift-free and bit-reproducible across backends.
- **The standard**: 8×H100-80GB × `Qwen2.5-72B-Instruct`, GSM8K selection, with a
  held-out **FineWeb bits-per-byte** quality metric (tokenizer-invariant).
- **Records**: every run logs throughput (seeds/sec) + ensemble accuracy + FineWeb
  bpb to [`RECORDS.md`](RECORDS.md).

```bash
# verify the math on CPU (no GPU needed)
python -m pytest tests/test_perturb.py tests/test_worker.py tests/test_fineweb.py tests/test_speedrun.py -q
# run the standard (on the 8×H100 node)
python speedrun.py --config configs/standard_8xh100_qwen72b.yaml
```

Full methodology, the base-resident memory trade-off, and the honest "batching
profiles" analysis are in **[docs/SPEEDRUN.md](docs/SPEEDRUN.md)**.


## Requirements

### Option1: Python / Conda
```bash
(optional) conda activate your_env
pip install -r requirements.txt
```

### Option2: Docker

From the directory containing `RandOpt/`:

| Step | Command |
|------|---------|
| **Build** | `docker build -f RandOpt/docker/Dockerfile_vllm -t randopt-vllm:latest .` |
| **Run** | `docker run -it --gpus all randopt-vllm:latest bash` |
| **Run** (with data) | `docker run -it --gpus all -v /path/to/RandOpt/data:/workspace/data randopt-vllm:latest bash` |


## Run RandOpt

### Post-train on your own dataset
Please follow the instructions in [CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md)

### Post-train on a standard dataset
First download the data here: [data/README.md](data/README.md)

Then, from the `RandOpt` directory:

| Mode | Command |
|------|---------|
| **Single node** | `sbatch scripts/single_node.sh` |
| **Multiple nodes** | `sbatch scripts/multiple_nodes.sh` |
| **Local** (no Slurm) | `bash scripts/local_run.sh` |

## Distill top-k models into a single model
Please follow the instructions in [distillation/README.md](distillation/README.md).

## Run Baselines
Please follow the instructions in [baselines/README.md](baselines/README.md)


## Citation
```bib
@misc{gan2026neuralthickets,
      title={Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights}, 
      author={Yulu Gan and Phillip Isola},
      year={2026},
      eprint={2603.12228},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.12228}, 
}
```
