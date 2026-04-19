#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a parameterized Prot2Chat-style protein-conditioned adapter."
    )
    parser.add_argument("--dataset-json", required=True, help="JSON file containing Prot2Chat-style samples.")
    parser.add_argument("--base-model-path", required=True, help="HF causal LM path.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metadata.")
    parser.add_argument(
        "--pdb-dir",
        default="",
        help="Directory containing PDB files when dataset items only store pdb filenames.",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        default="",
        help="Optional directory containing or storing per-protein embedding tensors.",
    )
    parser.add_argument(
        "--precompute-embeddings",
        action="store_true",
        help=(
            "Precompute all required protein embeddings in the main process and persist them "
            "under --embedding-cache-dir before DataLoader workers are started."
        ),
    )
    parser.add_argument(
        "--use-prot2chat-preprocess",
        action="store_true",
        help="Use Prot2Chat preprocess.get_mpnn_emb() when cached embeddings are unavailable.",
    )
    parser.add_argument(
        "--prot2chat-repo",
        default="/home/ubuntu/cqr_files/protein_design/COT_enzyme_design/Prot2Chat/prot2chat",
        help="Path to the local Prot2Chat repo containing preprocess.py.",
    )
    parser.add_argument("--eval-json", default="", help="Optional evaluation JSON.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Optional cap on the number of flattened train QA pairs for quick smoke tests.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="Optional cap on the number of flattened eval QA pairs for quick smoke tests.",
    )
    parser.add_argument("--max-text-length", type=int, default=1024)
    parser.add_argument("--protein-max-len", type=int, default=512)
    parser.add_argument("--protein-embed-dim", type=int, default=1152)
    parser.add_argument("--num-queries", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--auto-batch-size",
        action="store_true",
        help="Probe larger batch sizes on GPU before training and pick the largest stable value.",
    )
    parser.add_argument(
        "--max-auto-batch-size",
        type=int,
        default=0,
        help="Upper bound used with --auto-batch-size. Default means min(dataset_size, batch_size * 8).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--gpu-mem-log-every",
        type=int,
        default=0,
        help="When > 0 and running on CUDA, log GPU memory stats every N optimizer steps.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="cpu / cuda / auto")
    parser.add_argument(
        "--precision",
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Autocast precision when running on CUDA.",
    )
    parser.add_argument("--use-lora", action="store_true", help="Attach new LoRA layers.")
    parser.add_argument("--lora-path", default="", help="Optional existing LoRA checkpoint to load.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,v_proj",
        help="Comma-separated target modules when creating a new LoRA adapter.",
    )
    parser.add_argument("--train-base-model", action="store_true", help="Unfreeze the base LM.")
    parser.add_argument("--train-adapter-only", action="store_true", help="Freeze LM params and only train adapter.")
    parser.add_argument(
        "--adapter-init-state",
        default="",
        help=(
            "Optional adapter checkpoint to warm-start from. "
            "Supports either a plain adapter state_dict or the original Prot2Chat "
            "checkpoint format containing adapter_model_weight."
        ),
    )
    parser.add_argument(
        "--save-model-artifacts",
        action="store_true",
        help="Also save model/tokenizer artifacts. By default only adapter checkpoints and run metadata are saved.",
    )
    parser.add_argument(
        "--skip-missing-proteins",
        action="store_true",
        help="Drop flattened samples whose protein files cannot be resolved under --pdb-dir.",
    )
    return parser.parse_args()


@dataclass
class FlattenedSample:
    protein_ref: str
    prompt: str
    response: str


def load_samples(dataset_json: str) -> list[FlattenedSample]:
    raw = json.loads(Path(dataset_json).read_text())
    samples: list[FlattenedSample] = []
    for item in raw:
        protein_ref = item.get("protein_path") or item.get("pdb") or item.get("protein")
        if not protein_ref:
            raise ValueError("Each dataset item must contain one of protein_path / pdb / protein.")
        conversations = item.get("conversations", [])
        for pair in conversations:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            samples.append(FlattenedSample(protein_ref=protein_ref, prompt=pair[0], response=pair[1]))
    if not samples:
        raise ValueError(f"No valid conversation pairs found in {dataset_json}")
    return samples


def maybe_trim_samples(samples: list[FlattenedSample], max_samples: int) -> list[FlattenedSample]:
    if max_samples and max_samples > 0:
        return samples[:max_samples]
    return samples


def resolve_protein_path(protein_ref: str, pdb_dir: str) -> Path:
    protein_path = Path(protein_ref)
    if protein_path.exists():
        return protein_path
    if pdb_dir:
        candidate = Path(pdb_dir) / protein_ref
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve protein path for {protein_ref}")


def filter_resolvable_samples(
    samples: list[FlattenedSample],
    pdb_dir: str,
) -> tuple[list[FlattenedSample], int]:
    kept: list[FlattenedSample] = []
    skipped = 0
    for sample in samples:
        try:
            resolve_protein_path(sample.protein_ref, pdb_dir)
        except FileNotFoundError:
            skipped += 1
            continue
        kept.append(sample)
    return kept, skipped


def build_prot2chat_importer(repo_dir: str):
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import preprocess  # type: ignore

    return preprocess


def main() -> None:
    args = parse_args()

    import random

    import torch
    from torch import nn
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except Exception:  # pragma: no cover - optional dependency path
        LoraConfig = PeftModel = get_peft_model = None

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    samples = load_samples(args.dataset_json)
    samples = maybe_trim_samples(samples, args.max_train_samples)
    eval_samples = maybe_trim_samples(load_samples(args.eval_json), args.max_eval_samples) if args.eval_json else []
    skipped_train = 0
    skipped_eval = 0
    if args.skip_missing_proteins:
        samples, skipped_train = filter_resolvable_samples(samples, args.pdb_dir)
        eval_samples, skipped_eval = filter_resolvable_samples(eval_samples, args.pdb_dir)
        if not samples:
            raise RuntimeError("All training samples were filtered out due to missing protein files.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.embedding_cache_dir:
        Path(args.embedding_cache_dir).mkdir(parents=True, exist_ok=True)

    preprocess_module = None
    if args.use_prot2chat_preprocess:
        preprocess_module = build_prot2chat_importer(args.prot2chat_repo)

    class ProteinEmbeddingStore:
        def __init__(self) -> None:
            self.memory_cache: dict[str, torch.Tensor] = {}

        def cache_path(self, protein_path: Path) -> Path | None:
            if not args.embedding_cache_dir:
                return None
            return Path(args.embedding_cache_dir) / f"{protein_path.stem}.pt"

        def load(self, protein_ref: str) -> torch.Tensor:
            protein_path = resolve_protein_path(protein_ref, args.pdb_dir)
            cache_key = str(protein_path.resolve())
            if cache_key in self.memory_cache:
                return self.memory_cache[cache_key].clone()

            cache_path = self.cache_path(protein_path)
            if cache_path and cache_path.exists():
                embedding = torch.load(cache_path, map_location="cpu")
                if not isinstance(embedding, torch.Tensor):
                    embedding = torch.tensor(embedding)
                self.memory_cache[cache_key] = embedding.float()
                return embedding.float().clone()

            if preprocess_module is None:
                raise RuntimeError(
                    "Protein embedding not found in cache and --use-prot2chat-preprocess was not enabled."
                )

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                raise RuntimeError(
                    "Protein embedding cache miss inside a DataLoader worker. "
                    "This usually means a worker is trying to call Prot2Chat preprocess on CUDA. "
                    "Use --num-workers 0, or enable embedding precomputation with "
                    "--precompute-embeddings and --embedding-cache-dir."
                )

            embedding = preprocess_module.get_mpnn_emb(str(protein_path))
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding)
            embedding = embedding.float().cpu()
            if cache_path:
                torch.save(embedding, cache_path)
            self.memory_cache[cache_key] = embedding
            return embedding.clone()

    embedding_store = ProteinEmbeddingStore()

    should_precompute_embeddings = args.precompute_embeddings or (
        args.use_prot2chat_preprocess and args.num_workers > 0
    )
    if should_precompute_embeddings and not args.embedding_cache_dir:
        raise RuntimeError(
            "Embedding precomputation requires --embedding-cache-dir so worker processes can reuse cached tensors."
        )
    if should_precompute_embeddings:
        unique_refs = sorted({sample.protein_ref for sample in samples} | {sample.protein_ref for sample in eval_samples})
        total_refs = len(unique_refs)
        print(
            f"Precomputing Prot2Chat protein embeddings for {total_refs} unique proteins "
            f"into {args.embedding_cache_dir} ..."
        )
        for idx, protein_ref in enumerate(unique_refs, start=1):
            embedding_store.load(protein_ref)
            if idx == 1 or idx % 100 == 0 or idx == total_refs:
                print(f"[precompute {idx}/{total_refs}] cached {protein_ref}")
        if preprocess_module is not None:
            for model_name in [f"model_{i}" for i in range(1, 10)]:
                if hasattr(preprocess_module, model_name):
                    setattr(preprocess_module, model_name, None)
            preprocess_module = None
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype if device.startswith("cuda") else torch.float32,
    )

    if args.lora_path:
        if PeftModel is None:
            raise RuntimeError("peft is required to load --lora-path")
        model = PeftModel.from_pretrained(model, args.lora_path)
    elif args.use_lora:
        if get_peft_model is None or LoraConfig is None:
            raise RuntimeError("peft is required for --use-lora")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden_size from base model config.")

    class DynamicPositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self) -> torch.Tensor:
            return self.pe

    class ProteinStructureSequenceAdapter(nn.Module):
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_heads: int,
            num_queries: int,
            max_len: int,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.num_heads = num_heads
            self.num_queries = num_queries
            self.max_len = max_len
            self.linear_proj = nn.Linear(input_dim, output_dim)
            self.pos_encoder = DynamicPositionalEncoding(output_dim, max_len)
            self.learnable_queries = nn.Parameter(torch.randn(num_queries, output_dim))
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=num_heads,
                batch_first=True,
            )
            self.output_proj = nn.Linear(output_dim, output_dim)
            self.question_proj = nn.Linear(output_dim, output_dim)
            self.layer_norm1 = nn.LayerNorm(output_dim)
            self.layer_norm2 = nn.LayerNorm(output_dim)
            self.layer_norm3 = nn.LayerNorm(output_dim)

        def forward(self, x: torch.Tensor, h_state: torch.Tensor) -> torch.Tensor:
            batch_size, seq_len, _ = x.size()
            if seq_len < self.max_len:
                padding = torch.zeros(
                    batch_size,
                    self.max_len - seq_len,
                    self.input_dim,
                    device=x.device,
                    dtype=x.dtype,
                )
                x = torch.cat([x, padding], dim=1)
            elif seq_len > self.max_len:
                x = x[:, : self.max_len, :]

            x_proj = self.layer_norm1(self.linear_proj(x))
            pe = self.pos_encoder()[:, : x_proj.size(1), :].to(x_proj.dtype).to(x_proj.device)
            x_pos = x_proj + pe

            queries = self.learnable_queries.unsqueeze(0).expand(batch_size, -1, -1)
            q_state = self.question_proj(h_state)
            if q_state.dim() == 2:
                q_state = q_state.unsqueeze(1)
            queries = self.layer_norm2(queries + q_state)
            q_pe = self.pos_encoder()[:, : self.num_queries * 2 : 2, :].to(queries.dtype).to(queries.device)
            queries = queries + q_pe
            attn_output, _ = self.cross_attention(queries, x_pos, x_pos)
            attn_output = self.layer_norm3(attn_output)
            return self.output_proj(attn_output)

    adapter = ProteinStructureSequenceAdapter(
        input_dim=args.protein_embed_dim,
        output_dim=hidden_size,
        num_heads=args.num_heads,
        num_queries=args.num_queries,
        max_len=args.protein_max_len,
    )

    if args.adapter_init_state:
        checkpoint_obj = torch.load(args.adapter_init_state, map_location="cpu")
        if isinstance(checkpoint_obj, dict) and "adapter_model_weight" in checkpoint_obj:
            checkpoint_obj = checkpoint_obj["adapter_model_weight"]
        if not isinstance(checkpoint_obj, dict):
            raise ValueError(f"Unsupported adapter checkpoint format: {args.adapter_init_state}")
        checkpoint_obj = dict(checkpoint_obj)
        checkpoint_obj.pop("pos_encoder.pe", None)
        missing_keys, unexpected_keys = adapter.load_state_dict(checkpoint_obj, strict=False)
        if missing_keys:
            raise RuntimeError(
                f"Missing keys when loading adapter checkpoint {args.adapter_init_state}: {missing_keys}"
            )
        if unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys when loading adapter checkpoint {args.adapter_init_state}: {unexpected_keys}"
            )

    if args.train_adapter_only:
        for param in model.parameters():
            param.requires_grad = False
    elif not args.train_base_model:
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    model.to(device)
    adapter.to(device)

    class ConversationDataset(Dataset):
        def __init__(self, items: list[FlattenedSample]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = self.items[idx]
            prompt_text = f"Human: {item.prompt}\nAssistant: "
            full_text = f"{prompt_text}{item.response}{tokenizer.eos_token}"
            prompt_ids = tokenizer(
                prompt_text,
                truncation=True,
                max_length=args.max_text_length,
                return_tensors="pt",
            )
            full_ids = tokenizer(
                full_text,
                truncation=True,
                max_length=args.max_text_length,
                return_tensors="pt",
            )
            input_ids = full_ids.input_ids[0]
            attention_mask = full_ids.attention_mask[0]
            labels = input_ids.clone()
            prompt_len = prompt_ids.input_ids.size(1)
            labels[:prompt_len] = -100

            protein_embedding = embedding_store.load(item.protein_ref)

            return {
                "protein_ref": item.protein_ref,
                "prompt_ids": prompt_ids.input_ids[0],
                "prompt_attention_mask": prompt_ids.attention_mask[0],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "protein_embedding": protein_embedding,
            }

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        pad_id = tokenizer.pad_token_id
        input_ids = pad_sequence([x["input_ids"] for x in batch], batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence(
            [x["attention_mask"] for x in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence([x["labels"] for x in batch], batch_first=True, padding_value=-100)
        prompt_ids = pad_sequence([x["prompt_ids"] for x in batch], batch_first=True, padding_value=pad_id)
        prompt_attention_mask = pad_sequence(
            [x["prompt_attention_mask"] for x in batch],
            batch_first=True,
            padding_value=0,
        )
        protein_embeddings = []
        for x in batch:
            protein_embedding = x["protein_embedding"]
            if protein_embedding.size(0) < args.protein_max_len:
                pad_len = args.protein_max_len - protein_embedding.size(0)
                padding = torch.zeros(pad_len, protein_embedding.size(1), dtype=protein_embedding.dtype)
                protein_embedding = torch.cat([protein_embedding, padding], dim=0)
            else:
                protein_embedding = protein_embedding[: args.protein_max_len]
            protein_embeddings.append(protein_embedding)
        return {
            "protein_ref": [x["protein_ref"] for x in batch],
            "prompt_ids": prompt_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "protein_embeddings": torch.stack(protein_embeddings, dim=0),
        }

    train_dataset = ConversationDataset(samples)
    eval_dataset = ConversationDataset(eval_samples) if eval_samples else None

    trainable_params = [p for p in list(model.parameters()) + list(adapter.parameters()) if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Adjust --train-adapter-only / --train-base-model / --use-lora.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    trainable_model_params = any(p.requires_grad for p in model.parameters())

    def precision_context():
        if not device.startswith("cuda") or args.precision == "fp32":
            return torch.autocast(device_type="cpu", enabled=False)
        dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def format_gib(value_bytes: int | float) -> str:
        return f"{value_bytes / (1024 ** 3):.2f} GiB"

    def gpu_memory_snapshot() -> str:
        if not device.startswith("cuda"):
            return "gpu_mem=cpu_only"
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak_allocated = torch.cuda.max_memory_allocated()
        return (
            f"gpu_free={format_gib(free_bytes)} "
            f"gpu_total={format_gib(total_bytes)} "
            f"gpu_allocated={format_gib(allocated)} "
            f"gpu_reserved={format_gib(reserved)} "
            f"gpu_peak_allocated={format_gib(peak_allocated)}"
        )

    global_step = 0
    running_loss = 0.0

    def run_eval() -> float:
        if eval_loader is None:
            return float("nan")
        model.eval()
        adapter.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in eval_loader:
                loss = forward_loss(batch)
                losses.append(loss.item())
        model.train()
        adapter.train()
        return sum(losses) / max(len(losses), 1)

    def forward_loss(batch: dict[str, Any]) -> torch.Tensor:
        prompt_ids = batch["prompt_ids"].to(device)
        prompt_attention_mask = batch["prompt_attention_mask"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        protein_embeddings = batch["protein_embeddings"].to(device)

        with torch.set_grad_enabled(trainable_model_params):
            with precision_context():
                prompt_outputs = model(
                    input_ids=prompt_ids,
                    attention_mask=prompt_attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                question_state = prompt_outputs.hidden_states[-1][:, -1, :]
        with precision_context():
            adapter_outputs = adapter(protein_embeddings, question_state)
            text_embeds = model.get_input_embeddings()(input_ids)
            combined_embeds = torch.cat([adapter_outputs, text_embeds], dim=1)
            combined_mask = torch.cat(
                [
                    torch.ones(
                        adapter_outputs.size(0),
                        adapter_outputs.size(1),
                        device=device,
                        dtype=attention_mask.dtype,
                    ),
                    attention_mask,
                ],
                dim=1,
            )
            outputs = model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                use_cache=False,
            )
            logits = outputs.logits[:, adapter_outputs.size(1) :, :]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            return nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

    def build_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    def maybe_auto_tune_batch_size() -> int:
        if not args.auto_batch_size or not device.startswith("cuda"):
            return args.batch_size
        if len(train_dataset) == 0:
            return args.batch_size

        upper_bound = args.max_auto_batch_size
        if upper_bound <= 0:
            upper_bound = min(len(train_dataset), max(args.batch_size, args.batch_size * 8))
        upper_bound = max(args.batch_size, upper_bound)

        best_batch_size = args.batch_size
        candidate = args.batch_size
        print(
            f"[train_prot2chat_adapter] auto batch-size probe start: "
            f"initial={args.batch_size} max={upper_bound}"
        )
        while candidate <= upper_bound:
            probe_count = min(candidate, len(train_dataset))
            probe_batch = collate_fn([train_dataset[i] for i in range(probe_count)])
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                loss = forward_loss(probe_batch)
                (loss / max(args.grad_accum_steps, 1)).backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_ratio = free_bytes / max(total_bytes, 1)
                best_batch_size = candidate
                print(
                    f"[train_prot2chat_adapter] auto batch-size probe success: "
                    f"batch_size={candidate} {gpu_memory_snapshot()}"
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                print(
                    f"[train_prot2chat_adapter] auto batch-size probe OOM at batch_size={candidate}; "
                    f"keeping batch_size={best_batch_size}"
                )
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                break
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            next_candidate = candidate * 2
            if next_candidate > upper_bound:
                break
            if free_ratio < 0.35:
                print(
                    f"[train_prot2chat_adapter] auto batch-size probe stopping with free_ratio={free_ratio:.2f}; "
                    f"selected batch_size={best_batch_size}"
                )
                break
            candidate = next_candidate

        return best_batch_size

    args.batch_size = maybe_auto_tune_batch_size()
    train_loader = build_loader(train_dataset, args.batch_size, shuffle=True)
    eval_loader = build_loader(eval_dataset, args.batch_size, shuffle=False) if eval_dataset is not None else None

    metadata = {
        "dataset_json": args.dataset_json,
        "eval_json": args.eval_json,
        "base_model_path": args.base_model_path,
        "use_lora": args.use_lora,
        "lora_path": args.lora_path,
        "protein_max_len": args.protein_max_len,
        "protein_embed_dim": args.protein_embed_dim,
        "num_queries": args.num_queries,
        "num_heads": args.num_heads,
        "batch_size": args.batch_size,
        "auto_batch_size": args.auto_batch_size,
        "max_auto_batch_size": args.max_auto_batch_size,
        "gpu_mem_log_every": args.gpu_mem_log_every,
        "adapter_init_state": args.adapter_init_state,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "train_size_flattened": len(samples),
        "eval_size_flattened": len(eval_samples),
        "skip_missing_proteins": args.skip_missing_proteins,
        "skipped_train_missing": skipped_train,
        "skipped_eval_missing": skipped_eval,
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))
    if args.skip_missing_proteins and (skipped_train or skipped_eval):
        print(
            f"[train_prot2chat_adapter] skipped missing proteins: "
            f"train={skipped_train} eval={skipped_eval}"
        )

    model.train()
    adapter.train()
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader, start=1):
            loss = forward_loss(batch)
            (loss / args.grad_accum_steps).backward()
            running_loss += loss.item()
            global_step += 1

            if global_step % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                print(
                    f"[train_prot2chat_adapter] epoch={epoch + 1} step={global_step} "
                    f"batch={batch_idx}/{len(train_loader)} loss={avg_loss:.4f}"
                )
                running_loss = 0.0

            if args.eval_every and global_step % args.eval_every == 0 and eval_loader is not None:
                eval_loss = run_eval()
                print(f"[train_prot2chat_adapter] eval step={global_step} loss={eval_loss:.4f}")

            if args.gpu_mem_log_every and device.startswith("cuda") and global_step % args.gpu_mem_log_every == 0:
                print(f"[train_prot2chat_adapter] step={global_step} {gpu_memory_snapshot()}")

            if args.save_every and global_step % args.save_every == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(adapter.state_dict(), checkpoint_dir / "adapter_state.pt")
                if args.save_model_artifacts and hasattr(model, "save_pretrained"):
                    model.save_pretrained(checkpoint_dir / "model")
                    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), final_dir / "adapter_state.pt")
    if args.save_model_artifacts and hasattr(model, "save_pretrained"):
        model.save_pretrained(final_dir / "model")
        tokenizer.save_pretrained(final_dir / "tokenizer")
    print(f"[train_prot2chat_adapter] Training complete. Outputs saved to {final_dir}")


if __name__ == "__main__":
    main()
