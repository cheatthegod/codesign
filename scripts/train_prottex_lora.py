#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic LoRA SFT trainer for the ProtTeX causal language model front-end."
    )
    parser.add_argument("--train-jsonl", required=True, help="JSONL file with text or prompt/response examples.")
    parser.add_argument("--model-path", required=True, help="HF model path for ProtTeX.")
    parser.add_argument("--output-dir", required=True, help="Directory for saved checkpoints.")
    parser.add_argument("--eval-jsonl", default="", help="Optional eval JSONL file.")
    parser.add_argument("--text-key", default="text", help="Field used when samples already contain full training text.")
    parser.add_argument("--prompt-key", default="prompt", help="Prompt field for prompt/response format.")
    parser.add_argument("--response-key", default="response", help="Response field for prompt/response format.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target modules for LoRA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from torch.optim import Optimizer
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    try:
        from peft import LoraConfig, get_peft_model
    except Exception:  # pragma: no cover
        LoraConfig = get_peft_model = None

    # Compatibility shim for newer accelerate wrappers that call optimizer.train()/eval()
    # even when using standard torch.optim.AdamW.
    if not hasattr(Optimizer, "train"):
        setattr(Optimizer, "train", lambda self: None)
    if not hasattr(Optimizer, "eval"):
        setattr(Optimizer, "eval", lambda self: None)

    def load_jsonl(path: str) -> list[dict[str, Any]]:
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"No rows found in {path}")
        return rows

    train_rows = load_jsonl(args.train_jsonl)
    eval_rows = load_jsonl(args.eval_jsonl) if args.eval_jsonl else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
    )

    if args.use_lora:
        if get_peft_model is None or LoraConfig is None:
            raise RuntimeError("peft is required when --use-lora is enabled")
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    def format_chat(prompt: str, response: str) -> tuple[str, str]:
        prompt_text = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        full_text = f"{prompt_text}{response}<|eot_id|>"
        return prompt_text, full_text

    def preprocess_row(row: dict[str, Any]) -> dict[str, Any]:
        if args.text_key in row and row[args.text_key]:
            text = row[args.text_key]
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=args.max_length,
            )
            tokenized["labels"] = tokenized["input_ids"][:]
            return tokenized

        prompt = row.get(args.prompt_key)
        response = row.get(args.response_key)
        if prompt is None or response is None:
            raise ValueError(
                f"Row must contain either `{args.text_key}` or `{args.prompt_key}` + `{args.response_key}`."
            )
        prompt_text, full_text = format_chat(prompt, response)
        prompt_tokenized = tokenizer(prompt_text, truncation=True, max_length=args.max_length)
        full_tokenized = tokenizer(full_text, truncation=True, max_length=args.max_length)
        labels = list(full_tokenized["input_ids"])
        prompt_len = len(prompt_tokenized["input_ids"])
        labels[:prompt_len] = [-100] * min(prompt_len, len(labels))
        full_tokenized["labels"] = labels
        return full_tokenized

    train_dataset = Dataset.from_list([preprocess_row(row) for row in train_rows])
    eval_dataset = Dataset.from_list([preprocess_row(row) for row in eval_rows]) if eval_rows else None

    class PackedCollator:
        def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            max_len = max(len(f["input_ids"]) for f in features)
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for feature in features:
                pad_len = max_len - len(feature["input_ids"])
                batch["input_ids"].append(feature["input_ids"] + [tokenizer.pad_token_id] * pad_len)
                batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
                batch["labels"].append(feature["labels"] + [-100] * pad_len)
            return {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in batch.items()
            }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        evaluation_strategy="steps" if eval_dataset is not None and args.eval_steps > 0 else "no",
        save_strategy="steps",
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PackedCollator(),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    config_summary = {
        "train_jsonl": args.train_jsonl,
        "eval_jsonl": args.eval_jsonl,
        "model_path": args.model_path,
        "use_lora": args.use_lora,
        "max_length": args.max_length,
    }
    (Path(args.output_dir) / "train_config.json").write_text(
        json.dumps(config_summary, indent=2, ensure_ascii=False)
    )
    print(f"[train_prottex_lora] Finished. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
