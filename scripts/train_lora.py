import argparse, json, os
from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from peft import LoraConfig, get_peft_model

import yaml

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def format_example(ex):
    return ex["text"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model_name"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=cfg["data"]["train_file"])["train"]
    def tokenize(ex):
        t = tok(format_example(ex), truncation=True, max_length=cfg["data"]["cutoff_len"])
        t["labels"] = t["input_ids"].copy()
        return t
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    lcfg = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)

    args_tr = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        max_steps=cfg["training"]["max_steps"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        fp16=cfg["training"]["fp16"],
        report_to=[],
    )
    dc = DataCollatorForLanguageModeling(tok, mlm=False)
    trainer = Trainer(model=model, args=args_tr, train_dataset=ds, data_collator=dc)
    trainer.train()
    model.save_pretrained(os.path.join(out_dir, "peft"))

if __name__ == "__main__":
    main()
