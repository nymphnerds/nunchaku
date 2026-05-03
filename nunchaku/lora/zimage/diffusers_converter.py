"""
Convert Z-Image LoRAs into a stable Diffusers-style intermediate format.
"""

from __future__ import annotations

import torch

from ...utils import load_state_dict_in_safetensors


def _normalize_key(key: str) -> str:
    for prefix in ("diffusion_model.",):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def to_diffusers(input_lora: str | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Normalize incoming Z-Image LoRA weights to the key layout expected by the
    converter. Current Manager / AI Toolkit outputs are already close to
    Diffusers format, so this mainly strips wrapper prefixes and normalizes
    tensor dtypes.
    """
    if isinstance(input_lora, str):
        tensors = load_state_dict_in_safetensors(input_lora, device="cpu")
    else:
        tensors = {k: v for k, v in input_lora.items()}

    normalized: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        new_key = _normalize_key(key)
        if value.dtype not in [torch.float64, torch.float32, torch.bfloat16, torch.float16]:
            value = value.to(torch.bfloat16)
        normalized[new_key] = value
    return normalized
