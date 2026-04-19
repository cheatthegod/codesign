#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path


RESTYPES = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified inference entrypoint for the ProtTeX LLM front-end."
    )
    parser.add_argument("--model-path", required=True, help="HF model path.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["function", "structure", "cot", "design", "custom"],
        help="Inference mode.",
    )
    parser.add_argument("--input-seq", default="", help="Raw amino acid sequence for structure/design/cot modes.")
    parser.add_argument("--input-protein-pkl", default="", help="Tokenized protein pkl for function mode.")
    parser.add_argument("--prompt", default="", help="Optional custom prompt override.")
    parser.add_argument("--output-text", default="", help="Optional text output path.")
    parser.add_argument("--output-pkl", default="", help="Optional pickle output path for generated responses.")
    parser.add_argument("--output-dir", default="", help="Optional directory used when detokenizing.")
    parser.add_argument("--character-aa-dict", default="./tokenizer_metadata/character_aa_dict.pkl")
    parser.add_argument("--character-protoken", default="./tokenizer_metadata/character.json")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--detokenize", action="store_true", help="Run ProtTeX detokenize_pdb.py after generation.")
    parser.add_argument(
        "--prottex-root",
        default="/home/ubuntu/cqr_files/protein_design/COT_enzyme_design/ProtTeX",
        help="Path to local ProtTeX repo, used for detokenization.",
    )
    parser.add_argument("--detokenize-ckpt", default="", help="Optional ProToken checkpoint override.")
    return parser.parse_args()


def extract_sequence(text: str) -> str | None:
    match = re.search(r"< protein sequence>(.*?)</ protein sequence>", text)
    return match.group(1) if match else None


def extract_structure(text: str) -> str | None:
    match = re.search(r"< protein structure>(.*?)</ protein structure>", text)
    return match.group(1) if match else None


def chat_wrap(user_prompt: str) -> str:
    return (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch_dtype,
    )

    with open(args.character_aa_dict, "rb") as handle:
        aa_dict = pickle.load(handle)
    with open(args.character_protoken, "r", encoding="utf-8") as handle:
        protoken_chars = json.load(handle)

    restype_dict = {i: aa for i, aa in enumerate(RESTYPES)}

    encoded_sequence = ""
    encoded_structure = ""

    if args.input_seq:
        encoded_sequence = "".join(aa_dict[aa] for aa in args.input_seq)

    if args.input_protein_pkl:
        with open(args.input_protein_pkl, "rb") as handle:
            protein_obj = pickle.load(handle)
        if isinstance(protein_obj, list):
            protein_obj = protein_obj[0]
        if not isinstance(protein_obj, dict):
            raise ValueError("input_protein_pkl must contain a dict-like tokenized protein object.")
        seq_len = int(protein_obj["seq_len"])
        vq_indexes = protein_obj["code_indices"][:seq_len]
        aa_sequence = [restype_dict[int(i)] for i in protein_obj["aatype"][:seq_len]]
        encoded_structure = "".join(protoken_chars[int(i)] for i in vq_indexes)
        encoded_sequence = "".join(aa_dict[i] for i in aa_sequence)

    if args.prompt:
        user_prompt = args.prompt
    elif args.mode == "function":
        user_prompt = (
            "Considering the protein structure above, predict its biological function by examining "
            "its structural features and comparing it to functionally characterized proteins.\n"
            f"< protein sequence>{encoded_sequence}</ protein sequence>\n"
            f"< protein structure>{encoded_structure}</ protein structure>"
        )
    elif args.mode == "structure":
        user_prompt = (
            "The table below provides protein sequence information about a specific protein.\n"
            f"< protein sequence>{encoded_sequence}</ protein sequence>\n"
            "Given above information, what is its protein structure?"
        )
    elif args.mode == "cot":
        user_prompt = (
            "Please analyze the provided protein sequence step by step, describe the likely fold "
            "and reasoning chain, and then predict the protein structure.\n"
            f"< protein sequence>{encoded_sequence}</ protein sequence>"
        )
    elif args.mode == "design":
        user_prompt = (
            "Design a functional protein sequence and predict its structure based on the following "
            "requirements. Return both < protein sequence> and < protein structure>.\n"
            f"{args.input_seq}"
        )
    else:
        user_prompt = args.prompt

    wrapped = chat_wrap(user_prompt)
    inputs = tokenizer.encode(wrapped, return_tensors="pt", add_special_tokens=True).to(model.device)
    prefix = tokenizer.decode(inputs[0], skip_special_tokens=True)

    generate_kwargs = {
        "input_ids": inputs,
        "max_new_tokens": args.max_new_tokens,
        "num_return_sequences": args.num_return_sequences,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generate_kwargs["top_p"] = args.top_p
        generate_kwargs["temperature"] = args.temperature
    elif args.num_beams > 1:
        generate_kwargs["num_beams"] = args.num_beams

    outputs = model.generate(**generate_kwargs)
    decoded = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    responses = [text[len(prefix) :].strip() if text.startswith(prefix) else text.strip() for text in decoded]

    for idx, response in enumerate(responses):
        print(f"[ProtTeX response {idx}]")
        print(response)
        print()

    if args.output_text:
        output_text = Path(args.output_text)
        output_text.parent.mkdir(parents=True, exist_ok=True)
        output_text.write_text("\n\n".join(responses), encoding="utf-8")

    if args.output_pkl:
        output_pkl = Path(args.output_pkl)
        output_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(output_pkl, "wb") as handle:
            pickle.dump(responses, handle)

    if args.detokenize:
        if not args.output_pkl:
            raise ValueError("--detokenize requires --output-pkl")
        output_dir = Path(args.output_dir or Path(args.output_pkl).with_suffix("").as_posix())
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(Path(args.prottex_root) / "scripts" / "detokenize_pdb.py"),
            "--input_path",
            str(Path(args.output_pkl).resolve()),
            "--output_dir",
            str(output_dir.resolve()),
        ]
        if args.detokenize_ckpt:
            cmd.extend(["--load_ckpt_path", args.detokenize_ckpt])
        subprocess.run(cmd, check=True, cwd=args.prottex_root)


if __name__ == "__main__":
    main()
