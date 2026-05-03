"""
Convert Z-Image LoRAs from Diffusers-style weights into the internal layout
consumed by :class:`NunchakuZImageTransformer2DModel`.
"""

from __future__ import annotations

from typing import Any

import torch

from ...utils import pad_tensor
from .diffusers_converter import to_diffusers


def pack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."
    lane_n, lane_k = 1, 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4
    frag_n = n_pack_size * num_n_lanes * lane_n
    frag_k = k_pack_size * num_k_lanes * lane_k
    weight = pad_tensor(weight, frag_n, 0)
    weight = pad_tensor(weight, frag_k, 1)
    if down:
        r, c = weight.shape
        r_frags, c_frags = r // frag_n, c // frag_k
        weight = weight.view(r_frags, frag_n, c_frags, frag_k).permute(2, 0, 1, 3)
    else:
        c, r = weight.shape
        c_frags, r_frags = c // frag_n, r // frag_k
        weight = weight.view(c_frags, frag_n, r_frags, frag_k).permute(0, 2, 1, 3)
    weight = weight.reshape(c_frags, r_frags, n_pack_size, num_n_lanes, k_pack_size, num_k_lanes, lane_k)
    weight = weight.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    return weight.view(c, r)


def unpack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    c, r = weight.shape
    assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."
    lane_n, lane_k = 1, 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4
    frag_n = n_pack_size * num_n_lanes * lane_n
    frag_k = k_pack_size * num_k_lanes * lane_k
    if down:
        r_frags, c_frags = r // frag_n, c // frag_k
    else:
        c_frags, r_frags = c // frag_n, r // frag_k
    weight = weight.view(c_frags, r_frags, num_n_lanes, num_k_lanes, n_pack_size, k_pack_size, lane_k)
    weight = weight.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    weight = weight.view(c_frags, r_frags, frag_n, frag_k)
    if down:
        weight = weight.permute(1, 2, 0, 3).contiguous().view(r, c)
    else:
        weight = weight.permute(0, 2, 1, 3).contiguous().view(c, r)
    return weight


def _fuse_lora_pairs(
    lora_pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if all(down.equal(lora_pairs[0][0]) for down, _ in lora_pairs[1:]):
        lora_down = lora_pairs[0][0]
        lora_up = torch.cat([up for _, up in lora_pairs], dim=0)
        return lora_down, lora_up

    lora_down = torch.cat([down for down, _ in lora_pairs], dim=0)
    total_rank = lora_down.shape[0]
    total_out = sum(up.shape[0] for _, up in lora_pairs)
    lora_up = torch.zeros(total_out, total_rank, dtype=lora_pairs[0][1].dtype, device=lora_pairs[0][1].device)
    row_offset = 0
    col_offset = 0
    for down, up in lora_pairs:
        out_features = up.shape[0]
        rank = down.shape[0]
        lora_up[row_offset : row_offset + out_features, col_offset : col_offset + rank] = up
        row_offset += out_features
        col_offset += rank
    return lora_down, lora_up


def convert_to_nunchaku_zimage_lowrank_dict(
    lora: dict[str, torch.Tensor],
    *,
    base_model: dict[str, tuple[torch.Tensor, torch.Tensor]],
    base_unquantized_sd: dict[str, torch.Tensor],
) -> dict[str, Any]:
    grouped_loras: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in lora.items():
        if ".lora_A." in key:
            prefix = key.split(".lora_A.", 1)[0]
            grouped_loras.setdefault(prefix, {})["A"] = value
        elif ".lora_B." in key:
            prefix = key.split(".lora_B.", 1)[0]
            grouped_loras.setdefault(prefix, {})["B"] = value

    quantized_loras: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    unquantized_loras: dict[str, torch.Tensor] = {}
    qkv_groups: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    swiglu_groups: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    direct_quantized_matches = 0
    unmatched_prefixes: list[str] = []

    def _store_single(module_name: str, pair: tuple[torch.Tensor, torch.Tensor]):
        if module_name in base_model:
            quantized_loras[module_name] = pair
        else:
            unmatched_prefixes.append(module_name)

    for prefix, pair in grouped_loras.items():
        if "A" not in pair or "B" not in pair:
            continue
        lora_down = pair["A"].contiguous()
        lora_up = pair["B"].contiguous()
        module_pair = (lora_down, lora_up)
        matched = True

        if prefix.endswith(".attention.to_q"):
            target = prefix[: -len(".attention.to_q")] + ".attention.fused_module"
            qkv_groups.setdefault(target, {})[0] = module_pair
        elif prefix.endswith(".attention.to_k"):
            target = prefix[: -len(".attention.to_k")] + ".attention.fused_module"
            qkv_groups.setdefault(target, {})[1] = module_pair
        elif prefix.endswith(".attention.to_v"):
            target = prefix[: -len(".attention.to_v")] + ".attention.fused_module"
            qkv_groups.setdefault(target, {})[2] = module_pair
        elif prefix.endswith(".attention.to_out.0"):
            _store_single(prefix, module_pair)
            direct_quantized_matches += 1
        elif prefix.endswith(".feed_forward.w1"):
            target = prefix[: -len(".feed_forward.w1")] + ".feed_forward.net.0.proj"
            swiglu_groups.setdefault(target, {})[0] = module_pair
        elif prefix.endswith(".feed_forward.w3"):
            target = prefix[: -len(".feed_forward.w3")] + ".feed_forward.net.0.proj"
            swiglu_groups.setdefault(target, {})[1] = module_pair
        elif prefix.endswith(".feed_forward.w2"):
            target = prefix[: -len(".feed_forward.w2")] + ".feed_forward.net.2"
            _store_single(target, module_pair)
            direct_quantized_matches += 1
        elif f"{prefix}.weight" in base_unquantized_sd:
            unquantized_loras[f"{prefix}.lora_A.weight"] = lora_down
            unquantized_loras[f"{prefix}.lora_B.weight"] = lora_up
        else:
            matched = False

        if not matched:
            unmatched_prefixes.append(prefix)

    fused_qkv_matches = 0
    for module_name, parts in qkv_groups.items():
        if set(parts.keys()) != {0, 1, 2}:
            unmatched_prefixes.append(module_name)
            continue
        if module_name not in base_model:
            unmatched_prefixes.append(module_name)
            continue
        quantized_loras[module_name] = _fuse_lora_pairs([parts[0], parts[1], parts[2]])
        fused_qkv_matches += 1

    fused_swiglu_matches = 0
    for module_name, parts in swiglu_groups.items():
        if set(parts.keys()) != {0, 1}:
            unmatched_prefixes.append(module_name)
            continue
        if module_name not in base_model:
            unmatched_prefixes.append(module_name)
            continue
        quantized_loras[module_name] = _fuse_lora_pairs([parts[1], parts[0]])
        fused_swiglu_matches += 1

    return {
        "quantized": quantized_loras,
        "unquantized": unquantized_loras,
        "debug": {
            "state_tensor_count": len(lora),
            "grouped_prefix_count": len(grouped_loras),
            "quantized_module_matches": len(quantized_loras),
            "direct_quantized_matches": direct_quantized_matches,
            "fused_qkv_matches": fused_qkv_matches,
            "fused_swiglu_matches": fused_swiglu_matches,
            "unquantized_weight_matches": len(unquantized_loras) // 2,
            "unmatched_prefix_count": len(unmatched_prefixes),
            "unmatched_prefix_examples": unmatched_prefixes[:8],
        },
    }


def to_nunchaku(
    input_lora: str | dict[str, torch.Tensor],
    *,
    base_sd: dict[str, tuple[torch.Tensor, torch.Tensor]],
    base_unquantized_sd: dict[str, torch.Tensor],
) -> dict[str, Any]:
    diffusers_lora = to_diffusers(input_lora)
    converted = convert_to_nunchaku_zimage_lowrank_dict(
        diffusers_lora,
        base_model=base_sd,
        base_unquantized_sd=base_unquantized_sd,
    )
    converted["debug"]["path"] = input_lora if isinstance(input_lora, str) else "<state_dict>"
    return converted
