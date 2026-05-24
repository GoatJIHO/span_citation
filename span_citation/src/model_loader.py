import torch
from .model_config import ModelNameMap
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def _get_cache_dir(cfg):
    return getattr(getattr(cfg, "paths", {}), "cache_dir", getattr(cfg, "cache_dir", None))


def load_model_and_tokenizer(cfg, model_name: str):
    model_id = ModelNameMap.get_model_name(model_name)
    cache_dir = _get_cache_dir(cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
