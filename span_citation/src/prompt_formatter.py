from __future__ import annotations


def prompt_formatting(mt:str, SYSTEM_PROMPT:str, CONTEXT:str) -> str:
    if "llama" in mt:
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{CONTEXT}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    if "qwen" in mt:
        return (
            "<|im_start|>system\n"
            f"{SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{CONTEXT}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


