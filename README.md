# Fine-Tuning Experiments (LoRA/QLoRA)

Semi-hands-on configs and scripts for fine-tuning open-weight LLMs.
Nội dung bán thực hành: config & script cho fine-tuning LLM open-weight.

> ⚠️ Work in progress. Small configs first, real runs later.

## Goals
- Minimal LoRA/QLoRA config with PEFT.
- Reproducible train/eval commands.
- Keep GPU memory modest (8–24GB target).

## Layout
- `configs/` – YAML configs (model, LoRA ranks, train args).
- `scripts/` – train/infer helpers.
- `data/` – tiny demo JSONL (placeholder).
- `logs/` – example training logs (truncated).

## Quick Start (conceptual)
```bash
# install
pip install transformers peft accelerate datasets bitsandbytes

# dry-run (conceptual)
python scripts/train_lora.py --config configs/lora-llama3-8b.yaml
