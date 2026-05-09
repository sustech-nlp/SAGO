# SAGO: Sign-Align Gradient Optimization for LLM Unlearning

Official code for the paper: **"Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem"** (ACL 2026 Main)

[![arXiv](https://img.shields.io/badge/arXiv-2604.14808-b31b1b.svg)](https://arxiv.org/abs/2604.14808) <!-- [![ACL 2026](https://img.shields.io/badge/ACL%202026-Main-green)]() -->

## Overview

This paper reframes LLM unlearning as an **asymmetric two-task learning problem**: retention is the primary objective and forgetting is auxiliary. We propose:

- **SAGO (Sign-Align Gradient Optimization)**: A retention-prioritized gradient synthesis method that applies element-wise gating to ensure no parameter update opposes the retain gradient direction.
- **PCGrad adaptation**: Module-wise projection of forget gradients onto the orthogonal complement of retain gradients, adapted from multi-task learning to the unlearning setting.

Both methods compose modularly with diverse unlearning objectives (GradDiff, NPO+GD, SimNPO+GD).

## Installation

```bash
conda create -n sago python=3.11 -y
conda activate sago
pip install torch==2.4.1
pip install .[lm_eval]
pip install --no-build-isolation flash-attn==2.6.3
```

## Data Setup

```bash
# Download WMDP forget/retain corpora
python setup_data.py --wmdp

# Download RWKU data (batch unlearning setting, Batch/1-50)
python setup_data.py --rwku_batch
```

## Running Experiments

### WMDP (Zephyr-7B-beta)

**Baselines:**
```bash
# All baselines (GradAscent, GradDiff, NPO, SimNPO, RMU, etc.)
bash scripts/wmdp_baselines.sh
```

**SAGO / PCGrad:**
```bash
bash scripts/wmdp_unlearn.sh
```

**Individual experiment with Hydra overrides:**
```bash
# SAGO + GradDiff on WMDP Cyber
accelerate launch --config_file configs/accelerate/multi_gpu.yaml \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/wmdp/GradDiff_sago_cyber \
    model=zephyr-7b-beta \
    model.model_args.pretrained_model_name_or_path=HuggingFaceH4/zephyr-7b-beta \
    task_name=GradDiff_sago_cyber
```

### RWKU (LLaMA3-8B-Instruct)

**Baselines:**
```bash
bash scripts/rwku_baselines.sh
```

**SAGO / PCGrad:**
```bash
bash scripts/rwku_unlearn.sh
```

**Individual experiment:**
```bash
# SAGO + GradDiff on RWKU Batch/1-50
accelerate launch --config_file configs/accelerate/default_config.yaml \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/rwku/Batch50_SAGO \
    model=Llama-3-8B-Instruct \
    model.model_args.pretrained_model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct \
    task_name=GradDiff_SAGO_Batch50
```

## Evaluation

### WMDP (MMLU + WMDP accuracy)

```bash
# MMLU
accelerate launch -m lm_eval --model hf \
    --model_args "pretrained=PATH_TO_MODEL,dtype=bfloat16" \
    --tasks mmlu --num_fewshot 5 --batch_size 8 \
    --output_path saves/lm_eval/mmlu

# WMDP Cyber
accelerate launch -m lm_eval --model hf \
    --model_args "pretrained=PATH_TO_MODEL,dtype=bfloat16" \
    --tasks wmdp_cyber --batch_size 8 \
    --output_path saves/lm_eval/wmdp_cyber
```

### RWKU (ROUGE-L on forget/neighbor sets)

```bash
python src/eval.py experiment=eval/rwku/default.yaml \
    model=Llama-3-8B-Instruct \
    model.model_args.pretrained_model_name_or_path=PATH_TO_MODEL \
    target=Batch/1-50 \
    task_name=my_eval \
    paths.output_dir=saves/rwku/my_eval
```

## Supported Methods

| Method | Type | Config Handler | Trainer Config |
|--------|------|---------------|----------------|
| SAGO + GradDiff | Proposed | `GradDiffWithSurgery` | `GradDiffWithSurgery` |
| SAGO + NPO | Proposed | `NPOWithSurgery` | `NPOWithSurgery` |
| SAGO + SimNPO | Proposed | `SimNPOWithSurgery` | `SimNPOWithSurgery` |
| PCGrad + GradDiff | Proposed | `GradDiffWithSurgery` (strategy=pcgrad) | `GradDiffWithSurgery` |
| PCGrad + NPO | Proposed | `NPOWithSurgery` (strategy=pcgrad) | `NPOWithSurgery` |
| PCGrad (Global) | Proposed | `GradDiffWithSurgery` (strategy=pcgrad_global) | `GradDiffWithSurgery` |
| GradAscent | Baseline | `GradAscent` | `GradAscent` |
| GradDiff | Baseline | `GradDiff` | `GradDiff` |
| NPO | Baseline | `NPO` | `NPO` |
| SimNPO | Baseline | `SimNPO` | `SimNPO` |
| RMU | Baseline | `RMU` | `RMU` |

The surgery strategy is controlled via `trainer.method_args.surgery_strategy` in the config:
- `"sago"` — Sign-Align Gradient Optimization (default)
- `"pcgrad"` — Module-wise PCGrad projection
- `"pcgrad_global"` — Global PCGrad projection

## Project Structure

```
src/
  train.py                          # Training entrypoint
  eval.py                           # Evaluation entrypoint
  trainer/
    unlearn/
      grad_surgery.py               # SAGO and PCGrad implementations (GradSurgeryMixin)
      grad_ascent.py                # Gradient Ascent baseline
      grad_diff.py                  # Gradient Difference baseline
      npo.py / npo_forget.py        # NPO baselines
      simnpo.py / simnpo_forget.py  # SimNPO baselines
      rmu.py                        # RMU baseline
configs/
  trainer/                          # Hydra configs for each method
  experiment/unlearn/{wmdp,rwku}/   # Experiment configs
  eval/                             # Evaluation configs
scripts/                            # Experiment runner scripts
```

## Citation

```bibtex
@inproceedings{xiao2026sago,
  title={Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem},
  author={Xiao, Zeguan and Li, Siqing and Wang, Yong and Wei, Xuetao and Yang, Jian and Chen, Yun and Chen, Guanhua},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year={2026}
}
```

## Acknowledgments

This codebase is built on [OpenUnlearning](https://github.com/locuslab/open-unlearning).

## License

MIT License. See [LICENSE](LICENSE) for details.
