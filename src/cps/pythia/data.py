from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import DataConfig


def build_batch(tokenizer: Any, config: DataConfig, device: Any) -> dict[str, Any]:
    import torch

    if config.sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")

    if config.source == "token_file":
        if config.token_file is None:
            raise ValueError("token_file is required when source=token_file")
        dtype = np.dtype(config.token_dtype)
        tokens = np.memmap(Path(config.token_file), dtype=dtype, mode="r")
        needed = config.batch_size * config.sequence_length
        segment = np.asarray(tokens[config.start_index : config.start_index + needed], dtype=np.int64)
        if segment.size != needed:
            raise ValueError("token file does not contain the requested batch window")
        input_ids = torch.from_numpy(segment.reshape(config.batch_size, config.sequence_length))
        attention_mask = torch.ones_like(input_ids)
    elif config.source == "builtin":
        text = "\n".join(config.prompts)
        encoded = tokenizer(
            [text] * config.batch_size,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=config.sequence_length,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    else:
        raise ValueError(f"unsupported data source: {config.source}")

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }
