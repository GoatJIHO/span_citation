# Explicit Evidence-Span Supervision for Citation Intent Classification

This repository provides the implementation of our evidence-span supervision framework for citation intent classification.

## Installation

```bash
pip install -r requirements.txt
```

If flash-attn is not installed, set attn_implementation: "sdpa" in config.yaml.



## Main options:
Edit config.yaml to set the dataset paths, model name, training hyperparameters, and evaluation options.
```
experiment:
  model: "qwen3"
  data_name: "act2"
  train_type: "ours"
  epochs: 10
  seed: 42

training:
  lambda_span: 0.7
  lambda_cls: 0.8
  lambda_label: 1.0
  lambda_format: 0.1
```

## Data Format

Training files should be placed according to the paths specified in config.yaml.

Each training example should be a JSONL record with the following format:

```
{"input": "citation context with #CITATION_TAG", "output": "evidence span\tLABEL"}
```

### Training and Evaluation

Run the full training and evaluation pipeline:
```
bash scripts/train_script.sh
```
