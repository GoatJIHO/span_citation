

import os
import re

SEED = int(os.environ.get("PYTHONHASHSEED", "42"))

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from span_citation.src.model_config import ModelNameMap
from span_citation.src.prompt_formatter import prompt_formatting
from span_citation.src.prompt_generator import generate_prompt


# ============================================================
# Reproducibility
# ============================================================

def set_full_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # torch 2.x precision flags
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"

    torch.use_deterministic_algorithms(True)


# ============================================================
# Callbacks
# ============================================================

class SkipSaveBeforeStepCallback(TrainerCallback):
    def __init__(self, min_save_step: int = 1000):
        self.min_save_step = min_save_step

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step < self.min_save_step:
            control.should_save = False
        return control


class StopAtStepCallback(TrainerCallback):
    def __init__(self, stop_step: int = 6000):
        self.stop_step = stop_step
        self.saved = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step >= self.stop_step:
            if not self.saved:
                control.should_save = True
                self.saved = True
            control.should_training_stop = True
            print(f"\n[알림] {self.stop_step} 스텝에 도달하여 저장 후 학습을 종료합니다.")
        return control


class WandbSpanLabelCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None:
            return
        if not hasattr(trainer, "_last_metrics"):
            return
        if wandb.run is None:
            return
        metrics = trainer._last_metrics
        if metrics:
            wandb.log(metrics, step=state.global_step)




# ============================================================
# Utils
# ============================================================

def get_assistant_end(model_name: str, tokenizer) -> str:
    name = model_name.lower()
    if "qwen" in name:
        return "<|im_end|>"
    if "llama" in name:
        return "<|eot_id|>"
    return tokenizer.eos_token or ""


def has_overlap(tok_s: int, tok_e: int, char_s: int, char_e: int) -> bool:
    return max(tok_s, char_s) < min(tok_e, char_e)


def _find_case_insensitive(text: str, pattern: str) -> int:
    if not pattern:
        return -1
    return text.lower().find(pattern.lower())


def find_local_span_offsets_in_prompt(
    prompt_text: str,
    user_context: str,
    span_text: str,
) -> tuple[int | None, int | None]:
    """
    Find gold evidence span inside the input context region of the prompt.

    Important:
    - We do NOT use answer-side span tokens as local evidence for classifier.
    - We first locate span_text inside user_context, then map it to prompt_text.
    - If exact user_context lookup fails, fallback to searching span_text inside prompt_text.
    """
    span_text = (span_text or "").strip()
    if not span_text:
        return None, None

    # 1) Exact: context in prompt + span in context
    ctx_start = prompt_text.find(user_context)
    span_start_in_ctx = user_context.find(span_text)
    if ctx_start >= 0 and span_start_in_ctx >= 0:
        s = ctx_start + span_start_in_ctx
        return s, s + len(span_text)

    # 2) Case-insensitive: context in prompt + span in context
    ctx_start_ci = _find_case_insensitive(prompt_text, user_context)
    span_start_ci = _find_case_insensitive(user_context, span_text)
    if ctx_start_ci >= 0 and span_start_ci >= 0:
        s = ctx_start_ci + span_start_ci
        return s, s + len(span_text)

    # 3) Fallback: exact span directly in prompt
    s = prompt_text.find(span_text)
    if s >= 0:
        return s, s + len(span_text)

    # 4) Fallback: case-insensitive span directly in prompt
    s = _find_case_insensitive(prompt_text, span_text)
    if s >= 0:
        return s, s + len(span_text)

    return None, None


# ============================================================
# Data collator
# ============================================================

@dataclass
class DynamicPadCollator:
    tokenizer: AutoTokenizer
    label_pad_token_id: int = -100
    pad_to_multiple_of: int | None = 8

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 1:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        pad_id = self.tokenizer.pad_token_id

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_span_mask = []
        batch_label_mask = []
        batch_format_mask = []
        batch_label_id = []
        batch_cls_pos = []
        batch_local_mask = []
        batch_span_source_valid = []

        for f in features:
            ids = f["input_ids"]
            attn = f["attention_mask"]
            labels = f["labels"]
            span_mask = f["span_mask"]
            label_mask = f["label_mask"]
            format_mask = f.get("format_mask", [0] * len(ids))
            local_mask = f.get("local_mask", [0] * len(ids))
            span_source_valid = int(f.get("span_source_valid", 0))

            pad_len = max_len - len(ids)

            batch_input_ids.append(ids + [pad_id] * pad_len)
            batch_attention_mask.append(attn + [0] * pad_len)
            batch_labels.append(labels + [self.label_pad_token_id] * pad_len)
            batch_span_mask.append(span_mask + [0] * pad_len)
            batch_label_mask.append(label_mask + [0] * pad_len)
            batch_format_mask.append(format_mask + [0] * pad_len)
            batch_local_mask.append(local_mask + [0] * pad_len)

            batch_label_id.append(int(f["label_id"]))
            batch_cls_pos.append(int(f["cls_pos"]))
            batch_span_source_valid.append(span_source_valid)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "span_mask": torch.tensor(batch_span_mask, dtype=torch.long),
            "label_mask": torch.tensor(batch_label_mask, dtype=torch.long),
            "format_mask": torch.tensor(batch_format_mask, dtype=torch.long),
            "label_id": torch.tensor(batch_label_id, dtype=torch.long),
            "cls_pos": torch.tensor(batch_cls_pos, dtype=torch.long),
            "local_mask": torch.tensor(batch_local_mask, dtype=torch.long),
            "span_source_valid": torch.tensor(batch_span_source_valid, dtype=torch.float32),
        }


# ============================================================
# Classifier heads
# ============================================================

class GlobalLocalFusionClassifier(nn.Module):
    """
    Supports three modes:
    - global:      use prompt-final representation only
    - local:       use evidence-local pooled representation only
    - global_local use [g, l, |g-l|, g*l]
    """
    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.10, mode: str = "global_local"):
        super().__init__()
        self.mode = str(mode).lower().strip()
        if self.mode not in ("global", "local", "global_local"):
            raise ValueError(f"Unknown cls_fusion mode: {mode}")

        if self.mode == "global_local":
            in_dim = hidden_size * 4
        else:
            in_dim = hidden_size

        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(in_dim, num_labels)

    def forward(self, global_hidden: torch.Tensor, local_hidden: torch.Tensor) -> torch.Tensor:
        if self.mode == "global":
            x = global_hidden
        elif self.mode == "local":
            x = local_hidden
        else:
            x = torch.cat(
                [
                    global_hidden,
                    local_hidden,
                    torch.abs(global_hidden - local_hidden),
                    global_hidden * local_hidden,
                ],
                dim=-1,
            )
        x = self.norm(x)
        x = self.dropout(x)
        return self.out(x)


# ============================================================
# Trainer
# ============================================================

class SpanLabelTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_type="ce",
        gamma=1.5,
        lambda_span=0.7,
        lambda_label=1.0,
        lambda_format=0.1,
        lambda_cls=0.8,
        cls_label_smoothing=0.05,
        class_weight=None,
        span_loss_control="winsor",
        span_loss_cap=5.0,
        span_loss_quantile=0.90,
        span_robust_mode="none",
        span_sample_loss_cap=0.0,
        span_sample_drop_threshold=0.0,
        span_sample_soft_temperature=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = str(loss_type).lower().strip()
        self.gamma = float(gamma)
        self.lambda_span = float(lambda_span)
        self.lambda_label = float(lambda_label)
        self.lambda_format = float(lambda_format)
        self.lambda_cls = float(lambda_cls)
        self.cls_label_smoothing = float(cls_label_smoothing)

        if class_weight is not None:
            self.class_weight = torch.tensor(class_weight, dtype=torch.float32)
        else:
            self.class_weight = None

        self.span_loss_control = str(span_loss_control).lower().strip()
        self.span_loss_cap = float(span_loss_cap)
        self.span_loss_quantile = float(span_loss_quantile)

        self.span_robust_mode = str(span_robust_mode).lower().strip()
        self.span_sample_loss_cap = float(span_sample_loss_cap)
        self.span_sample_drop_threshold = float(span_sample_drop_threshold)
        self.span_sample_soft_temperature = max(float(span_sample_soft_temperature), 1e-6)

        if self.loss_type not in ("ce", "ce_focal"):
            raise ValueError(f"loss_type must be 'ce' or 'ce_focal', got: {loss_type}")
        if self.span_robust_mode not in ("none", "valid_only", "drop", "soft", "hybrid"):
            raise ValueError(
                "span_robust_mode must be one of: none, valid_only, drop, soft, hybrid"
            )

        self.ce_none = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        self._last_metrics = {}

    def log(self, logs, *args, **kwargs):
        if hasattr(self, "_last_metrics") and self._last_metrics:
            logs = dict(logs)
            logs.update(self._last_metrics)
        return super().log(logs, *args, **kwargs)

    def _apply_token_control(self, vals: torch.Tensor, use_focal: bool, loss_control: str, cap: float, quantile: float):
        if vals.numel() == 0:
            return vals

        if use_focal:
            pt = torch.exp(-vals).clamp(1e-8, 1.0)
            vals = vals * (1.0 - pt).pow(self.gamma)

        if loss_control == "clip":
            vals = vals.clamp(max=cap)

        elif loss_control == "winsor":
            if vals.numel() > 1:
                q_cap = torch.quantile(vals.detach().float(), quantile).to(vals.dtype)
                if cap is not None and cap > 0:
                    q_cap = torch.minimum(q_cap, vals.new_tensor(cap))
                vals = torch.minimum(vals, q_cap)

        elif loss_control == "soft_cap":
            if cap is not None and cap > 0:
                cap_t = vals.new_tensor(cap)
                vals = cap_t * (1.0 - torch.exp(-vals / cap_t))

        elif loss_control == "none":
            pass

        else:
            raise ValueError(f"Unknown loss_control: {loss_control}")

        return vals

    def _component_loss(
        self,
        token_loss,
        mask,
        use_focal=False,
        loss_control="none",
        cap=5.0,
        quantile=0.90,
    ):
        vals = token_loss[mask]
        if vals.numel() == 0:
            zero = token_loss.new_zeros(())
            return zero, zero, zero, zero

        raw_mean = vals.mean()
        raw_max = vals.max()
        vals = self._apply_token_control(vals, use_focal, loss_control, cap, quantile)

        return vals.mean(), mask.sum().to(token_loss.dtype), raw_mean.detach(), raw_max.detach()

    def _span_component_loss(self, token_loss, span_sel, span_source_valid):
        """
        Stronger span loss control.

        Step 1: token-level winsor/clip/soft-cap.
        Step 2: sample-level source validity filtering / high-loss dropping / soft downweighting.
        Step 3: optional sample-level loss cap.
        """
        device = token_loss.device
        dtype = token_loss.dtype

        span_source_valid = span_source_valid.to(device=device, dtype=dtype)
        span_counts = span_sel.to(dtype).sum(dim=1)  # [B]
        has_span_sample = span_counts > 0

        vals = token_loss[span_sel]
        if vals.numel() == 0:
            zero = token_loss.new_zeros(())
            metrics = {
                "span_source_valid_ratio": 0.0,
                "span_effective_sample_ratio": 0.0,
                "span_dropped_sample_ratio": 0.0,
                "span_sample_raw_mean": 0.0,
                "span_sample_controlled_mean": 0.0,
                "span_sample_weight_mean": 0.0,
            }
            return zero, zero, zero, zero, metrics

        raw_global_mean = vals.mean().detach()
        raw_global_max = vals.max().detach()

        controlled_vals = self._apply_token_control(
            vals,
            use_focal=False,
            loss_control=self.span_loss_control,
            cap=self.span_loss_cap,
            quantile=self.span_loss_quantile,
        )

        controlled_token_loss = token_loss.new_zeros(token_loss.shape)
        controlled_token_loss[span_sel] = controlled_vals

        raw_token_loss = token_loss.masked_fill(~span_sel, 0.0)
        sample_raw_mean = raw_token_loss.sum(dim=1) / span_counts.clamp_min(1.0)
        sample_controlled_mean = controlled_token_loss.sum(dim=1) / span_counts.clamp_min(1.0)

        sample_weight = has_span_sample.to(dtype)

        # Drop spans that cannot be located in the original input context.
        if self.span_robust_mode in ("valid_only", "hybrid"):
            sample_weight = sample_weight * (span_source_valid > 0).to(dtype)

        # Drop very high-loss span samples.
        if self.span_robust_mode in ("drop", "hybrid") and self.span_sample_drop_threshold > 0:
            keep = (sample_raw_mean.detach() <= self.span_sample_drop_threshold).to(dtype)
            sample_weight = sample_weight * keep

        # Softly downweight high-loss span samples instead of hard dropping.
        if self.span_robust_mode == "soft" and self.span_sample_drop_threshold > 0:
            excess = torch.relu(sample_raw_mean.detach() - self.span_sample_drop_threshold)
            soft_w = torch.exp(-excess / self.span_sample_soft_temperature)
            sample_weight = sample_weight * soft_w.to(dtype)

        # Sample-level cap after token-level control.
        if self.span_sample_loss_cap is not None and self.span_sample_loss_cap > 0:
            cap_t = sample_controlled_mean.new_tensor(self.span_sample_loss_cap)
            sample_controlled_mean = torch.minimum(sample_controlled_mean, cap_t)

        weight_sum = sample_weight.sum().clamp_min(1e-8)
        L_span = (sample_controlled_mean * sample_weight).sum() / weight_sum

        # n_span is used only for has_span check. Use effective sample weight sum.
        n_span = sample_weight.sum().to(dtype)

        with torch.no_grad():
            total_span_samples = has_span_sample.float().sum().clamp_min(1.0)
            valid_ratio = ((span_source_valid > 0).float() * has_span_sample.float()).sum() / total_span_samples
            effective_ratio = (sample_weight > 0).float().sum() / total_span_samples
            dropped_ratio = 1.0 - effective_ratio
            raw_sample_mean = sample_raw_mean[has_span_sample].mean() if has_span_sample.any() else token_loss.new_zeros(())
            controlled_sample_mean = sample_controlled_mean[has_span_sample].mean() if has_span_sample.any() else token_loss.new_zeros(())
            weight_mean = sample_weight[has_span_sample].mean() if has_span_sample.any() else token_loss.new_zeros(())

            metrics = {
                "span_source_valid_ratio": float(valid_ratio.detach().cpu().item()),
                "span_effective_sample_ratio": float(effective_ratio.detach().cpu().item()),
                "span_dropped_sample_ratio": float(dropped_ratio.detach().cpu().item()),
                "span_sample_raw_mean": float(raw_sample_mean.detach().cpu().item()),
                "span_sample_controlled_mean": float(controlled_sample_mean.detach().cpu().item()),
                "span_sample_weight_mean": float(weight_mean.detach().cpu().item()),
            }

        return L_span, n_span, raw_global_mean, raw_global_max, metrics

    @staticmethod
    def _masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor):
        """Mean-pool hidden states over mask. If mask is empty, return fallback."""
        mask = mask.to(device=hidden.device, dtype=hidden.dtype)
        denom = mask.sum(dim=1, keepdim=True)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom.clamp_min(1.0)
        use_fallback = denom.squeeze(1) <= 0
        if use_fallback.any():
            pooled = torch.where(use_fallback.unsqueeze(-1), fallback, pooled)
        return pooled, denom.squeeze(1)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        span_mask = inputs.pop("span_mask").bool()
        label_mask = inputs.pop("label_mask").bool()
        format_mask = inputs.pop("format_mask").bool()
        label_id = inputs.pop("label_id").long()
        cls_pos = inputs.pop("cls_pos").long()
        local_mask = inputs.pop("local_mask").bool()
        span_source_valid = inputs.pop("span_source_valid").float()
        labels = inputs.pop("labels")

        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits  # [B,T,V]

        # causal shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_span_mask = span_mask[:, 1:].contiguous()
        shift_label_mask = label_mask[:, 1:].contiguous()
        shift_format_mask = format_mask[:, 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)

        active = shift_labels.ne(-100)
        span_sel = active & shift_span_mask
        label_sel = active & shift_label_mask
        format_sel = active & shift_format_mask

        L_span, n_span, L_span_raw, L_span_raw_max, span_metrics = self._span_component_loss(
            token_loss,
            span_sel,
            span_source_valid,
        )

        L_label, n_label, L_label_raw, L_label_raw_max = self._component_loss(
            token_loss,
            label_sel,
            use_focal=(self.loss_type == "ce_focal"),
            loss_control="none",
        )

        L_format, n_format, L_format_raw, L_format_raw_max = self._component_loss(
            token_loss,
            format_sel,
            use_focal=False,
            loss_control="none",
        )

        w_span = token_loss.new_tensor(self.lambda_span)
        w_label = token_loss.new_tensor(self.lambda_label)
        w_format = token_loss.new_tensor(self.lambda_format)

        has_span = (n_span > 0).to(token_loss.dtype)
        has_label = (n_label > 0).to(token_loss.dtype)
        has_format = (n_format > 0).to(token_loss.dtype)

        weight_sum = (
            w_span * has_span
            + w_label * has_label
            + w_format * has_format
        ).clamp_min(1e-8)

        gen_loss = (
            w_span * has_span * L_span
            + w_label * has_label * L_label
            + w_format * has_format * L_format
        ) / weight_sum

        # ------------------------------------------------------------
        # Global + local evidence-span auxiliary classification loss
        # ------------------------------------------------------------
        last_hidden = outputs.hidden_states[-1]
        cls_pos = cls_pos.to(last_hidden.device).clamp(min=0, max=last_hidden.size(1) - 1)
        b_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        global_hidden = last_hidden[b_idx, cls_pos]

        local_hidden, local_token_count = self._masked_mean_pool(
            last_hidden,
            local_mask.to(last_hidden.device),
            fallback=global_hidden,
        )

        if not hasattr(model, "label_classifier"):
            raise AttributeError(
                "model.label_classifier is not exist. Make sure to initialize it and move it to the same device as the model."
            )

        clf_device = next(model.label_classifier.parameters()).device
        if clf_device != global_hidden.device:
            model.label_classifier.to(global_hidden.device)

        cls_logits = model.label_classifier(global_hidden, local_hidden)
        class_weight = self.class_weight.to(cls_logits.device) if self.class_weight is not None else None
        cls_loss = F.cross_entropy(
            cls_logits.float(),
            label_id.to(cls_logits.device),
            weight=class_weight,
            label_smoothing=self.cls_label_smoothing,
        )

        loss = gen_loss + token_loss.new_tensor(self.lambda_cls) * cls_loss.to(gen_loss.dtype)

        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            with torch.no_grad():
                total_active = int(active.sum().detach().cpu().item())
                span_tokens = int(span_sel.sum().detach().cpu().item())
                label_tokens = int(label_sel.sum().detach().cpu().item())
                format_tokens = int(format_sel.sum().detach().cpu().item())
                pred = cls_logits.argmax(dim=-1)
                cls_acc = (pred == label_id.to(pred.device)).float().mean()
                local_has = (local_token_count > 0).float().mean()

                prefix = "train" if model.training else "eval_last_batch"

                self._last_metrics = {
                    f"{prefix}_loss_total_custom": float(loss.detach().cpu().item()),
                    f"{prefix}_loss_gen": float(gen_loss.detach().cpu().item()),
                    f"{prefix}_loss_cls": float(cls_loss.detach().cpu().item()),
                    f"{prefix}_cls_acc_batch": float(cls_acc.detach().cpu().item()),
                    f"{prefix}_lambda_cls": self.lambda_cls,
                    f"{prefix}_cls_label_smoothing": self.cls_label_smoothing,
                    f"{prefix}_loss_span": float(L_span.detach().cpu().item()),
                    f"{prefix}_loss_span_raw": float(L_span_raw.detach().cpu().item()),
                    f"{prefix}_loss_span_raw_max": float(L_span_raw_max.detach().cpu().item()),
                    f"{prefix}_span_loss_control": self.span_loss_control,
                    f"{prefix}_span_loss_cap": self.span_loss_cap,
                    f"{prefix}_span_loss_quantile": self.span_loss_quantile,
                    f"{prefix}_span_robust_mode": self.span_robust_mode,
                    f"{prefix}_span_sample_loss_cap": self.span_sample_loss_cap,
                    f"{prefix}_span_sample_drop_threshold": self.span_sample_drop_threshold,
                    f"{prefix}_loss_label": float(L_label.detach().cpu().item()),
                    f"{prefix}_loss_format": float(L_format.detach().cpu().item()),
                    f"{prefix}_w_span": float(w_span.detach().cpu().item()),
                    f"{prefix}_w_label": float(w_label.detach().cpu().item()),
                    f"{prefix}_w_format": float(w_format.detach().cpu().item()),
                    f"{prefix}_span_tokens": span_tokens,
                    f"{prefix}_label_tokens": label_tokens,
                    f"{prefix}_format_tokens": format_tokens,
                    f"{prefix}_active_tokens": total_active,
                    f"{prefix}_span_frac_of_active": (span_tokens / total_active) if total_active > 0 else 0.0,
                    f"{prefix}_label_frac_of_active": (label_tokens / total_active) if total_active > 0 else 0.0,
                    f"{prefix}_format_frac_of_active": (format_tokens / total_active) if total_active > 0 else 0.0,
                    f"{prefix}_local_tokens_mean": float(local_token_count.detach().float().mean().cpu().item()),
                    f"{prefix}_local_has_ratio": float(local_has.detach().cpu().item()),
                    f"{prefix}_loss_is_finite": 1.0 if torch.isfinite(loss.detach()).item() else 0.0,
                }

                for k, v in span_metrics.items():
                    self._last_metrics[f"{prefix}_{k}"] = v

        return (loss, outputs) if return_outputs else loss


# ============================================================
# Training args
# ============================================================

def build_training_args(args, output_dir):
    common = dict(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        remove_unused_columns=False,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        dataloader_drop_last=False,
        log_level="error",
        log_level_replica="error",
        report_to="wandb",
        run_name=f"{args.model}-{args.data_name}-lora-{args.loss}",
    )

    if args.eval_strategy == "steps":
        common.update(
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy=args.save_strategy,
            save_steps=args.eval_steps if args.save_strategy == "steps" else None,
        )
    else:
        common.update(
            eval_strategy="epoch",
            save_strategy=args.save_strategy,
        )

    common = {k: v for k, v in common.items() if v is not None}
    return TrainingArguments(**common)




# ============================================================
# Config / argument handling
# ============================================================

def cfg_get(cfg, key: str, default=None):
    cur = cfg
    for part in key.split('.'):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def fill_args_from_config(args, cfg):
    # Experiment settings
    args.model = args.model or cfg_get(cfg, 'experiment.model')
    args.data_name = args.data_name or cfg_get(cfg, 'experiment.data_name')
    args.train_type = args.train_type or cfg_get(cfg, 'experiment.train_type', 'new_ours')
    args.epochs = args.epochs if args.epochs is not None else cfg_get(cfg, 'experiment.epochs', 10)
    args.seed = args.seed if args.seed is not None else cfg_get(cfg, 'experiment.seed', 42)
    args.unique = args.unique if args.unique is not None else str(cfg_get(cfg, 'experiment.run_id', 'run'))

    # Training objective / optimization
    args.loss = args.loss or cfg_get(cfg, 'training.loss', 'ce_focal')
    args.focal_gamma = args.focal_gamma if args.focal_gamma is not None else cfg_get(cfg, 'training.focal_gamma', 1.5)
    args.lambda_span = args.lambda_span if args.lambda_span is not None else cfg_get(cfg, 'training.lambda_span', 0.7)
    args.lambda_label = args.lambda_label if args.lambda_label is not None else cfg_get(cfg, 'training.lambda_label', 1.0)
    args.lambda_format = args.lambda_format if args.lambda_format is not None else cfg_get(cfg, 'training.lambda_format', 0.1)
    args.lambda_cls = args.lambda_cls if args.lambda_cls is not None else cfg_get(cfg, 'training.lambda_cls', 0.8)
    args.cls_label_smoothing = args.cls_label_smoothing if args.cls_label_smoothing is not None else cfg_get(cfg, 'training.cls_label_smoothing', 0.05)
    args.use_class_weight = bool(args.use_class_weight or cfg_get(cfg, 'training.use_class_weight', False))
    args.cls_fusion = args.cls_fusion or cfg_get(cfg, 'training.cls_fusion', 'global_local')
    args.cls_dropout = args.cls_dropout if args.cls_dropout is not None else cfg_get(cfg, 'training.cls_dropout', 0.10)

    args.learning_rate = args.learning_rate if args.learning_rate is not None else cfg_get(cfg, 'training.learning_rate', 1e-4)
    args.warmup_ratio = args.warmup_ratio if args.warmup_ratio is not None else cfg_get(cfg, 'training.warmup_ratio', 0.05)
    args.weight_decay = args.weight_decay if args.weight_decay is not None else cfg_get(cfg, 'training.weight_decay', 0.01)
    args.lr_scheduler_type = args.lr_scheduler_type or cfg_get(cfg, 'training.lr_scheduler_type', 'cosine')
    args.optim = args.optim or cfg_get(cfg, 'training.optim', 'adamw_torch')
    args.save_total_limit = args.save_total_limit if args.save_total_limit is not None else cfg_get(cfg, 'training.save_total_limit', 20)
    args.max_grad_norm = args.max_grad_norm if args.max_grad_norm is not None else cfg_get(cfg, 'training.max_grad_norm', 1.0)
    args.logging_steps = args.logging_steps if args.logging_steps is not None else cfg_get(cfg, 'training.logging_steps', 200)
    args.eval_strategy = args.eval_strategy or cfg_get(cfg, 'training.eval_strategy', 'steps')
    args.eval_steps = args.eval_steps if args.eval_steps is not None else cfg_get(cfg, 'training.eval_steps', 200)
    args.save_strategy = args.save_strategy or cfg_get(cfg, 'training.save_strategy', 'epoch')
    args.stop_step = args.stop_step if args.stop_step is not None else cfg_get(cfg, 'training.stop_step', 0)

    args.bf16 = bool(args.bf16 or cfg_get(cfg, 'training.bf16', False))
    args.attn_implementation = args.attn_implementation or cfg_get(cfg, 'training.attn_implementation', 'flash_attention_2')
    args.lora_r = args.lora_r if args.lora_r is not None else cfg_get(cfg, 'training.lora_r', 32)
    args.lora_alpha = args.lora_alpha if args.lora_alpha is not None else cfg_get(cfg, 'training.lora_alpha', 64)
    args.lora_dropout = args.lora_dropout if args.lora_dropout is not None else cfg_get(cfg, 'training.lora_dropout', 0.05)
    args.gradient_checkpointing = bool(args.gradient_checkpointing or cfg_get(cfg, 'training.gradient_checkpointing', False))
    args.debug_tokenization = bool(args.debug_tokenization or cfg_get(cfg, 'training.debug_tokenization', False))
    args.debug_n_samples = args.debug_n_samples if args.debug_n_samples is not None else cfg_get(cfg, 'training.debug_n_samples', 3)

    args.span_loss_control = args.span_loss_control or cfg_get(cfg, 'training.span_loss_control', 'winsor')
    args.span_loss_cap = args.span_loss_cap if args.span_loss_cap is not None else cfg_get(cfg, 'training.span_loss_cap', 5.0)
    args.span_loss_quantile = args.span_loss_quantile if args.span_loss_quantile is not None else cfg_get(cfg, 'training.span_loss_quantile', 0.90)
    args.span_robust_mode = args.span_robust_mode or cfg_get(cfg, 'training.span_robust_mode', 'none')
    args.span_sample_loss_cap = args.span_sample_loss_cap if args.span_sample_loss_cap is not None else cfg_get(cfg, 'training.span_sample_loss_cap', 0.0)
    args.span_sample_drop_threshold = args.span_sample_drop_threshold if args.span_sample_drop_threshold is not None else cfg_get(cfg, 'training.span_sample_drop_threshold', 0.0)
    args.span_sample_soft_temperature = args.span_sample_soft_temperature if args.span_sample_soft_temperature is not None else cfg_get(cfg, 'training.span_sample_soft_temperature', 1.0)

    if args.output_name is None:
        run_id = str(cfg_get(cfg, 'experiment.run_id', 'run'))
        args.output_name = (
            f"{run_id}.lora_cls({args.lambda_cls})"
            f"_span({args.lambda_span})"
            f"_label({args.lambda_label})"
            f"_seed({args.seed})"
            f"_epochs({args.epochs})"
        )

    missing = [name for name in ('model', 'data_name', 'train_type') if getattr(args, name) in (None, '')]
    if missing:
        raise ValueError(f"Missing required config/arguments: {missing}")
    return args

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.environ.get("SPAN_CONFIG", "config.yaml"))

    # Optional CLI overrides. If omitted, values are read from config.yaml.
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--loss", type=str, default=None, choices=["ce", "ce_focal"])
    parser.add_argument("--focal_gamma", type=float, default=None)
    parser.add_argument("--train_type", type=str, default=None)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--lambda_span", type=float, default=None)
    parser.add_argument("--lambda_label", type=float, default=None)
    parser.add_argument("--lambda_format", type=float, default=None)

    # auxiliary classification
    parser.add_argument("--lambda_cls", type=float, default=None)
    parser.add_argument("--cls_label_smoothing", type=float, default=None)
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--cls_fusion", type=str, default=None, choices=["global", "local", "global_local"])
    parser.add_argument("--cls_dropout", type=float, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--unique", type=str, default=None)

    # training hyperparams
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--lr_scheduler_type", type=str, default=None)
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--eval_strategy", type=str, default=None, choices=["steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_strategy", type=str, default=None, choices=["no", "steps", "epoch"])
    parser.add_argument("--stop_step", type=int, default=None)

    # model/LoRA
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # debug
    parser.add_argument("--debug_tokenization", action="store_true")
    parser.add_argument("--debug_n_samples", type=int, default=None)

    # token-level span loss control
    parser.add_argument(
        "--span_loss_control",
        type=str,
        default=None,
        choices=["none", "clip", "winsor", "soft_cap"],
    )
    parser.add_argument("--span_loss_cap", type=float, default=None)
    parser.add_argument("--span_loss_quantile", type=float, default=None)

    # sample-level robust span control
    parser.add_argument(
        "--span_robust_mode",
        type=str,
        default=None,
        choices=["none", "valid_only", "drop", "soft", "hybrid"],
        help=(
            "none: no sample-level filtering, "
            "valid_only: use span loss only if gold span is found in input context, "
            "drop: drop high span-loss samples, "
            "soft: softly downweight high span-loss samples, "
            "hybrid: valid_only + drop + sample cap"
        ),
    )
    parser.add_argument("--span_sample_loss_cap", type=float, default=None)
    parser.add_argument("--span_sample_drop_threshold", type=float, default=None)
    parser.add_argument("--span_sample_soft_temperature", type=float, default=None)

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    args = fill_args_from_config(args, cfg)

    cache_dir = cfg_get(cfg, "paths.cache_dir", None)

    # set_full_determinism(args.seed)

    model_id = ModelNameMap.get_model_name(args.model)
    data_name = args.data_name

    wandb.init(
        project=f"{data_name}-citation-lora",
        name=(
            f"unique:{args.unique}|"
            f"(span:{args.lambda_span}, label:{args.lambda_label}, format:{args.lambda_format}, "
            f"cls:{args.lambda_cls}, fusion:{args.cls_fusion}, robust:{args.span_robust_mode})"
            f"{args.model}-(data:{args.train_type}, loss:{args.loss})"
        ),
    )

    train_data_root = Path(cfg.paths.train_data_root)
    train_type_dir = getattr(getattr(cfg, "train_type_dirs", {}), args.train_type, args.train_type)
    data_dir = train_data_root / train_type_dir / args.data_name

    output_name_path = Path(args.output_name)
    if output_name_path.is_absolute():
        output_dir = output_name_path
    else:
        tuned_root = Path(cfg.paths.tuned_dir) / f"{args.data_name}_{args.model}"
        output_dir = tuned_root / args.output_name

    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    os.makedirs(output_dir, exist_ok=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_files = {"train": str(train_file), "validation": str(val_file)}
    dataset = load_dataset("json", data_files=data_files)

    def extract_gold_label(raw_output: str) -> str:
        if "\t" in raw_output:
            _, label_text = raw_output.split("\t", 1)
        else:
            label_text = raw_output
        return label_text.strip()

    train_gold_labels = [extract_gold_label(x) for x in dataset["train"]["output"]]
    valid_gold_labels = [extract_gold_label(x) for x in dataset["validation"]["output"]]
    label_list = sorted(set(train_gold_labels + valid_gold_labels))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    print("[label mapping]", label2id)

    if args.use_class_weight:
        counter = Counter(train_gold_labels)
        total = sum(counter.values())
        num_labels = len(label_list)
        class_weight = [total / (num_labels * max(counter[label], 1)) for label in label_list]
        print("[class_weight]", dict(zip(label_list, class_weight)))
    else:
        class_weight = None

    # Training-only classifier head. Inference can still use generated span\tlabel.
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = model.get_input_embeddings().embedding_dim
    head_device = model.get_input_embeddings().weight.device
    head_dtype = model.get_input_embeddings().weight.dtype
    model.label_classifier = GlobalLocalFusionClassifier(
        hidden_size=hidden_size,
        num_labels=len(label_list),
        dropout=args.cls_dropout,
        mode=args.cls_fusion,
    ).to(device=head_device, dtype=head_dtype)
    model.config.label2id = label2id
    model.config.id2label = id2label

    assistant_end = get_assistant_end(args.model, tokenizer)
    sep_text = "\t"
    system_prompt = "You are an expert annotator trained to analyze the intent behind academic citations."

    def tokenize_function(example):
        user_context = example["input"]
        raw_output = example["output"]

        if "\t" in raw_output:
            span_text, label_text = raw_output.split("\t", 1)
        else:
            span_text, label_text = raw_output, ""

        span_text = span_text.strip()
        label_text = label_text.strip()
        if label_text not in label2id:
            raise ValueError(f"Unknown label: {label_text} / available={list(label2id.keys())}")
        label_id = label2id[label_text]

        prompt_text = prompt_formatting(
            args.model,
            system_prompt,
            generate_prompt(args.train_type, args.data_name, user_context),
        )

        answer_text = span_text + sep_text + label_text + assistant_end
        full_text = prompt_text + answer_text

        enc = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        offsets = enc["offset_mapping"]

        prompt_e = len(prompt_text)
        prompt_token_positions = [
            i for i, (tok_s, tok_e) in enumerate(offsets)
            if tok_s != tok_e and tok_s < prompt_e and tok_e <= prompt_e
        ]
        cls_pos = prompt_token_positions[-1] if prompt_token_positions else 0

        labels = [-100] * len(input_ids)
        span_mask = [0] * len(input_ids)
        label_mask = [0] * len(input_ids)
        format_mask = [0] * len(input_ids)
        local_mask = [0] * len(input_ids)

        # Answer-side span/label/format mask for generation loss.
        span_s = len(prompt_text)
        span_e = span_s + len(span_text)
        sep_s = span_e
        sep_e = sep_s + len(sep_text)
        label_s = sep_e
        label_e = label_s + len(label_text)
        end_s = label_e
        end_e = end_s + len(assistant_end)

        # Prompt-side local evidence span mask for auxiliary classifier.
        local_s, local_e = find_local_span_offsets_in_prompt(
            prompt_text=prompt_text,
            user_context=user_context,
            span_text=span_text,
        )
        span_source_valid = 1 if local_s is not None and local_e is not None else 0

        for i, (tok_s, tok_e) in enumerate(offsets):
            if tok_s == tok_e:
                continue

            # Generation target masks: answer region only.
            if has_overlap(tok_s, tok_e, span_s, span_e):
                labels[i] = input_ids[i]
                span_mask[i] = 1
            elif has_overlap(tok_s, tok_e, label_s, label_e):
                labels[i] = input_ids[i]
                label_mask[i] = 1
            elif has_overlap(tok_s, tok_e, sep_s, sep_e) or has_overlap(tok_s, tok_e, end_s, end_e):
                labels[i] = input_ids[i]
                format_mask[i] = 1

            # Classifier local evidence mask: prompt/input context region only.
            if span_source_valid and tok_e <= prompt_e and has_overlap(tok_s, tok_e, local_s, local_e):
                local_mask[i] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "span_mask": span_mask,
            "label_mask": label_mask,
            "format_mask": format_mask,
            "label_id": label_id,
            "cls_pos": cls_pos,
            "local_mask": local_mask,
            "span_source_valid": span_source_valid,
        }

    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=["input", "output"],
    )

    data_collator = DynamicPadCollator(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    training_args = build_training_args(args, output_dir)

    callbacks = []
    if args.data_name.lower() == "scicite" and args.stop_step > 0:
        callbacks.append(StopAtStepCallback(stop_step=args.stop_step))

    trainer = SpanLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        lambda_span=args.lambda_span,
        lambda_label=args.lambda_label,
        lambda_format=args.lambda_format,
        lambda_cls=args.lambda_cls,
        cls_label_smoothing=args.cls_label_smoothing,
        class_weight=class_weight,
        span_loss_control=args.span_loss_control,
        span_loss_cap=args.span_loss_cap,
        span_loss_quantile=args.span_loss_quantile,
        span_robust_mode=args.span_robust_mode,
        span_sample_loss_cap=args.span_sample_loss_cap,
        span_sample_drop_threshold=args.span_sample_drop_threshold,
        span_sample_soft_temperature=args.span_sample_soft_temperature,
        gamma=args.focal_gamma,
        processing_class=tokenizer,
        data_collator=data_collator,
        loss_type=args.loss,
        callbacks=callbacks,
    )

    if args.debug_tokenization:
        train_loader = trainer.get_train_dataloader()
        batch = next(iter(train_loader))
        print("\n===== ACTUAL BATCH TO MODEL =====")
        print("input_ids.shape        :", batch["input_ids"].shape)
        print("attention_mask.shape   :", batch["attention_mask"].shape)
        print("labels.shape           :", batch["labels"].shape)
        print("span_mask.shape        :", batch["span_mask"].shape)
        print("label_mask.shape       :", batch["label_mask"].shape)
        print("format_mask.shape      :", batch["format_mask"].shape)
        print("label_id.shape         :", batch["label_id"].shape)
        print("cls_pos.shape          :", batch["cls_pos"].shape)
        print("local_mask.shape       :", batch["local_mask"].shape)
        print("span_source_valid.shape:", batch["span_source_valid"].shape)

        for i in range(min(args.debug_n_samples, batch["input_ids"].size(0))):
            seq_len = int(batch["attention_mask"][i].sum().item())
            active_count = int((batch["labels"][i] != -100).sum().item())

            input_text = tokenizer.decode(
                batch["input_ids"][i][:seq_len].tolist(),
                skip_special_tokens=False,
            )
            active_ids = batch["input_ids"][i][batch["labels"][i] != -100].tolist()
            active_text = tokenizer.decode(active_ids, skip_special_tokens=False)
            local_ids = batch["input_ids"][i][batch["local_mask"][i].bool()].tolist()
            local_text = tokenizer.decode(local_ids, skip_special_tokens=False)

            print(f"\n--- batch sample {i} ---")
            print("full_tensor_len    :", batch["input_ids"][i].shape[0])
            print("non_pad_len        :", seq_len)
            print("active_count       :", active_count)
            print("label_id           :", int(batch["label_id"][i].item()))
            print("label_text         :", id2label[int(batch["label_id"][i].item())])
            print("cls_pos            :", int(batch["cls_pos"][i].item()))
            print("span_source_valid  :", int(batch["span_source_valid"][i].item()))
            print("local_token_count  :", int(batch["local_mask"][i].sum().item()))
            print("active_text        :", repr(active_text[:500]))
            print("local_text         :", repr(local_text[:500]))
            print("full_text          :", repr(input_text[:1000]))

    trainer.train()

    # Save final LoRA adapter and tokenizer.
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
