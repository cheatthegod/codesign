#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Prot2Chat-style protein-conditioned question answering from the command line."
    )
    parser.add_argument("--base-model-path", required=True, help="HF causal LM path.")
    parser.add_argument("--adapter-state", required=True, help="Path to adapter_state.pt from training.")
    parser.add_argument("--protein-path", required=True, help="Input PDB path.")
    parser.add_argument("--question", required=True, help="Question about the protein.")
    parser.add_argument("--output-json", default="", help="Optional path to store the full result as JSON.")
    parser.add_argument("--embedding-cache-dir", default="", help="Optional cache directory for protein embeddings.")
    parser.add_argument("--use-prot2chat-preprocess", action="store_true")
    parser.add_argument(
        "--prot2chat-repo",
        default="/home/ubuntu/cqr_files/protein_design/COT_enzyme_design/Prot2Chat/prot2chat",
        help="Path to the local Prot2Chat repo containing preprocess.py.",
    )
    parser.add_argument("--lora-path", default="", help="Optional LoRA checkpoint.")
    parser.add_argument("--protein-max-len", type=int, default=512)
    parser.add_argument("--protein-embed-dim", type=int, default=1152)
    parser.add_argument("--num-queries", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def build_prot2chat_importer(repo_dir: str):
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import preprocess  # type: ignore

    return preprocess


def main() -> None:
    args = parse_args()

    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftModel
    except Exception:  # pragma: no cover - optional dependency path
        PeftModel = None

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype if device.startswith("cuda") else torch.float32,
    )
    if args.lora_path:
        if PeftModel is None:
            raise RuntimeError("peft is required to load --lora-path")
        model = PeftModel.from_pretrained(model, args.lora_path)
    model.to(device)
    model.eval()

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
        def __init__(self) -> None:
            super().__init__()
            self.input_dim = args.protein_embed_dim
            self.output_dim = hidden_size
            self.num_heads = args.num_heads
            self.num_queries = args.num_queries
            self.max_len = args.protein_max_len
            self.linear_proj = nn.Linear(self.input_dim, self.output_dim)
            self.pos_encoder = DynamicPositionalEncoding(self.output_dim, self.max_len)
            self.learnable_queries = nn.Parameter(torch.randn(self.num_queries, self.output_dim))
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.output_dim,
                num_heads=self.num_heads,
                batch_first=True,
            )
            self.output_proj = nn.Linear(self.output_dim, self.output_dim)
            self.question_proj = nn.Linear(self.output_dim, self.output_dim)
            self.layer_norm1 = nn.LayerNorm(self.output_dim)
            self.layer_norm2 = nn.LayerNorm(self.output_dim)
            self.layer_norm3 = nn.LayerNorm(self.output_dim)

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
            q_state = self.question_proj(h_state).unsqueeze(1)
            queries = self.layer_norm2(queries + q_state)
            q_pe = self.pos_encoder()[:, : self.num_queries * 2 : 2, :].to(queries.dtype).to(queries.device)
            queries = queries + q_pe
            attn_output, _ = self.cross_attention(queries, x_pos, x_pos)
            attn_output = self.layer_norm3(attn_output)
            return self.output_proj(attn_output)

    adapter = ProteinStructureSequenceAdapter().to(device)
    adapter.load_state_dict(torch.load(args.adapter_state, map_location=device))
    adapter.eval()

    preprocess_module = None
    if args.use_prot2chat_preprocess:
        preprocess_module = build_prot2chat_importer(args.prot2chat_repo)

    protein_path = Path(args.protein_path)
    if not protein_path.exists():
        raise FileNotFoundError(f"Protein file not found: {protein_path}")

    cache_path = None
    if args.embedding_cache_dir:
        cache_dir = Path(args.embedding_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{protein_path.stem}.pt"

    if cache_path and cache_path.exists():
        protein_embedding = torch.load(cache_path, map_location="cpu")
        if not isinstance(protein_embedding, torch.Tensor):
            protein_embedding = torch.tensor(protein_embedding)
    else:
        if preprocess_module is None:
            raise RuntimeError(
                "No cached embedding found and --use-prot2chat-preprocess was not enabled."
            )
        protein_embedding = preprocess_module.get_mpnn_emb(str(protein_path))
        if not isinstance(protein_embedding, torch.Tensor):
            protein_embedding = torch.tensor(protein_embedding)
        protein_embedding = protein_embedding.float().cpu()
        if cache_path:
            torch.save(protein_embedding, cache_path)

    protein_embedding = protein_embedding.float()
    if protein_embedding.size(0) < args.protein_max_len:
        pad_len = args.protein_max_len - protein_embedding.size(0)
        protein_embedding = torch.cat(
            [
                protein_embedding,
                torch.zeros(pad_len, protein_embedding.size(1), dtype=protein_embedding.dtype),
            ],
            dim=0,
        )
    else:
        protein_embedding = protein_embedding[: args.protein_max_len]
    protein_embedding = protein_embedding.unsqueeze(0).to(device)

    prompt = f"Human: {args.question}\nAssistant: "
    encoded = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        prompt_outputs = model(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        question_state = prompt_outputs.hidden_states[-1][:, -1, :]
        protein_tokens = adapter(protein_embedding, question_state)
        text_embeds = model.get_input_embeddings()(encoded.input_ids)
        combined_embeds = torch.cat([protein_tokens, text_embeds], dim=1)
        combined_attention_mask = torch.cat(
            [
                torch.ones((1, protein_tokens.size(1)), device=device, dtype=encoded.attention_mask.dtype),
                encoded.attention_mask,
            ],
            dim=1,
        )
        generated_ids = model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            top_p=args.top_p,
            temperature=args.temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:", 1)[1].strip()
    print(response)

    if args.output_json:
        result = {
            "protein_path": str(protein_path.resolve()),
            "question": args.question,
            "response": response,
            "base_model_path": args.base_model_path,
            "adapter_state": str(Path(args.adapter_state).resolve()),
            "lora_path": args.lora_path,
        }
        Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
