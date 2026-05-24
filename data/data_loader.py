import os
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


def _load_project_config(config_path: str | None = None):
    """Load project configuration.

    Priority:
    1. function argument
    2. SPAN_CONFIG environment variable
    3. ./config.yaml
    """
    config_path = config_path or os.environ.get("SPAN_CONFIG", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Create one from config.yaml.example or set SPAN_CONFIG."
        )
    return OmegaConf.load(config_path)


def load_data(data_name: str, config_path: str | None = None):
    cfg = _load_project_config(config_path)

    eval_root = Path(cfg.paths.eval_data_root)
    rel_path = cfg.eval_data_files.get(data_name)
    if rel_path is None:
        raise ValueError(
            f"Unknown data_name={data_name}. "
            f"Available: {list(cfg.eval_data_files.keys())}"
        )

    data_path = eval_root / rel_path
    if not data_path.exists():
        raise FileNotFoundError(
            f"Evaluation data not found: {data_path}. "
            "Check paths.eval_data_root and eval_data_files in config.yaml."
        )

    if data_name == "acl_arc":
        df = pd.read_csv(data_path, sep="\t", index_col="unique_id")
        df = df.rename(columns={
            "context": "citation_context",
            "label": "citation_class_label",
        })
        return df

    if data_name == "act2":
        return pd.read_csv(data_path, sep="\t", index_col="unique_id")

    if data_name == "scicite":
        return pd.read_csv(data_path, sep="\t")

    raise ValueError(f"Unsupported data_name: {data_name}")
