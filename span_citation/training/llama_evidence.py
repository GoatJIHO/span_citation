import argparse
import os
import re
from collections import defaultdict

import torch
from omegaconf import OmegaConf
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.data_loader import load_data
from span_citation.src.model_config import ModelNameMap
from span_citation.src.prompt_formatter import prompt_formatting
from span_citation.src.prompt_generator import (
    generate_prompt,
    get_allowed_label,
    get_label_information,
    get_regex_label,
)


def cfg_get(cfg, key: str, default=None):
    cur = cfg
    for part in key.split('.'):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def make_output_name(cfg, model: str, data_name: str, epochs: int, seed: int,
                     lambda_cls: float, lambda_span: float, lambda_label: float):
    run_id = str(cfg_get(cfg, 'experiment.run_id', 'run'))
    return f"{run_id}.lora_cls({lambda_cls})_span({lambda_span})_label({lambda_label})_seed({seed})_epochs({epochs})"


def derive_lora_dir_from_config(cfg, model: str, data_name: str, output_name: str | None = None):
    tuned_dir = cfg_get(cfg, 'paths.tuned_dir', './tuned')
    if output_name is None:
        epochs = cfg_get(cfg, 'experiment.epochs', 10)
        seed = cfg_get(cfg, 'experiment.seed', 42)
        lambda_cls = cfg_get(cfg, 'training.lambda_cls', 0.8)
        lambda_span = cfg_get(cfg, 'training.lambda_span', 0.7)
        lambda_label = cfg_get(cfg, 'training.lambda_label', 1.0)
        output_name = make_output_name(cfg, model, data_name, epochs, seed, lambda_cls, lambda_span, lambda_label)
    return os.path.join(tuned_dir, f"{data_name}_{model}", output_name)


def fill_eval_args_from_config(args, cfg):
    args.model = args.model or cfg_get(cfg, 'experiment.model')
    args.data_name = args.data_name or cfg_get(cfg, 'experiment.data_name')
    args.training_type = args.training_type or cfg_get(cfg, 'experiment.train_type', 'new_ours')
    args.model_type = args.model_type or cfg_get(cfg, 'eval.model_type', None)
    args.max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else cfg_get(cfg, 'eval.max_new_tokens', 160)
    args.bin_size = args.bin_size if args.bin_size is not None else cfg_get(cfg, 'eval.bin_size', 1000)
    args.attn_implementation = args.attn_implementation or cfg_get(cfg, 'eval.attn_implementation', None)
    args.lora = args.lora or cfg_get(cfg, 'eval.lora', 'true')

    if not args.llama_lora_dir:
        args.llama_lora_dir = cfg_get(cfg, 'eval.llama_lora_dir', '') or derive_lora_dir_from_config(
            cfg, args.model, args.data_name, cfg_get(cfg, 'eval.output_name', None)
        )

    missing = [name for name in ('model', 'data_name', 'training_type') if getattr(args, name) in (None, '')]
    if missing:
        raise ValueError(f"Missing required config/arguments: {missing}")
    return args


def extract_label(output_text: str, label_pattern, allowed_labels):
    match = label_pattern.search(output_text or "")
    if match:
        label = match.group(1).upper()
        return label if label in allowed_labels else None
    return None


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def add_to_bins(bin_results, sent_len, pred, true_label, bin_size):
    bin_idx = (sent_len // bin_size) * bin_size
    bin_results[bin_idx]["preds"].append(pred)
    bin_results[bin_idx]["trues"].append(true_label)


def print_metrics(name: str, bin_results, allowed_labels, bin_size: int):
    print(f"\n===== {name} =====")
    print(f"{'Length Bin':<15} | {'Count':<8} | {'Accuracy':<10} | {'Macro F1':<10}")
    print("-" * 55)

    labels_for_score = sorted(list(allowed_labels))

    for b in sorted(bin_results.keys()):
        b_preds = bin_results[b]["preds"]
        b_trues = bin_results[b]["trues"]
        if not b_preds:
            continue
        acc = accuracy_score(b_trues, b_preds)
        f1 = f1_score(b_trues, b_preds, average="macro", labels=labels_for_score, zero_division=0)
        bin_label = f"{b}-{b + bin_size - 1}"
        print(f"{bin_label:<15} | {len(b_preds):<8} | {acc:.4f}     | {f1:.4f}")

    all_preds = [p for res in bin_results.values() for p in res["preds"]]
    all_trues = [t for res in bin_results.values() for t in res["trues"]]
    print("-" * 55)
    print(f"{name} Total Macro F1:", f1_score(all_trues, all_preds, average="macro", labels=labels_for_score, zero_division=0))
    print(f"{name} Total Accuracy :", accuracy_score(all_trues, all_preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.environ.get("SPAN_CONFIG", "config.yaml"))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument("--training_type", type=str, default=None)
    parser.add_argument("--lora", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--llama_lora_dir", type=str, default="")
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--bin_size", type=int, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    args = fill_eval_args_from_config(args, cfg)

    model_id = ModelNameMap.get_model_name(args.model)
    cache_dir = cfg_get(cfg, 'paths.cache_dir', None)
    model_type = args.model_type or args.model

    LABEL_MAP = get_label_information(args.data_name)
    LABEL_PATTERN = re.compile(get_regex_label(args.data_name))
    ALLOWED_LABELS = set(get_allowed_label(args.data_name))
    SYSTEM_PROMPT = "You are an expert annotator trained to analyze the intent behind academic citations."

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = dict(
        device_map="auto",
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
    )
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    base_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id

    if args.lora == "true":
        if not args.llama_lora_dir:
            raise ValueError("--llama_lora_dir is required when --lora true")
        model = PeftModel.from_pretrained(base_model, args.llama_lora_dir)
        print(f"[lora] loaded: {args.llama_lora_dir}")
    else:
        model = base_model

    model.eval()
    df = load_data(args.data_name, config_path=args.config)

    gen_bins = defaultdict(lambda: {"preds": [], "trues": []})
    invalid_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sent = row["citation_context"]
        sent_len = len(sent)
        true_label = LABEL_MAP[row["citation_class_label"]]

        prompt = prompt_formatting(
            model_type,
            SYSTEM_PROMPT,
            generate_prompt(args.training_type, args.data_name, sent),
        )

        decoded_output = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )
        pred = extract_label(decoded_output, LABEL_PATTERN, ALLOWED_LABELS)

        if pred is None:
            pred = "__INVALID__"
            invalid_count += 1

        add_to_bins(gen_bins, sent_len, pred, true_label, args.bin_size)

    print_metrics("GENERATE_LABEL", gen_bins, ALLOWED_LABELS, args.bin_size)
    print("Generate invalid outputs:", invalid_count, "/", len(df))
