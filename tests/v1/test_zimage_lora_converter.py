from __future__ import annotations

import torch

from nunchaku.lora.zimage.nunchaku_converter import convert_to_nunchaku_zimage_lowrank_dict


def test_convert_to_nunchaku_zimage_lowrank_dict_maps_fused_and_unquantized_targets():
    rank = 2
    in_features = 4

    base_model = {
        "layers.0.attention.fused_module": (
            torch.zeros(rank, in_features, dtype=torch.bfloat16),
            torch.zeros(12, rank, dtype=torch.bfloat16),
        ),
        "layers.0.attention.to_out.0": (
            torch.zeros(rank, in_features, dtype=torch.bfloat16),
            torch.zeros(4, rank, dtype=torch.bfloat16),
        ),
        "layers.0.feed_forward.net.0.proj": (
            torch.zeros(rank, in_features, dtype=torch.bfloat16),
            torch.zeros(8, rank, dtype=torch.bfloat16),
        ),
        "layers.0.feed_forward.net.2": (
            torch.zeros(rank, 8, dtype=torch.bfloat16),
            torch.zeros(4, rank, dtype=torch.bfloat16),
        ),
    }
    base_unquantized_sd = {
        "layers.0.adaLN_modulation.0.weight": torch.zeros(6, 4, dtype=torch.bfloat16),
    }

    lora = {
        "diffusion_model.layers.0.attention.to_q.lora_A.weight": torch.full((rank, in_features), 1, dtype=torch.bfloat16),
        "diffusion_model.layers.0.attention.to_q.lora_B.weight": torch.full((4, rank), 11, dtype=torch.bfloat16),
        "diffusion_model.layers.0.attention.to_k.lora_A.weight": torch.full((rank, in_features), 1, dtype=torch.bfloat16),
        "diffusion_model.layers.0.attention.to_k.lora_B.weight": torch.full((4, rank), 22, dtype=torch.bfloat16),
        "diffusion_model.layers.0.attention.to_v.lora_A.weight": torch.full((rank, in_features), 1, dtype=torch.bfloat16),
        "diffusion_model.layers.0.attention.to_v.lora_B.weight": torch.full((4, rank), 33, dtype=torch.bfloat16),
        "diffusion_model.layers.0.attention.to_out.0.lora_A.weight": torch.full((rank, in_features), 2, dtype=torch.bfloat16),
        "diffusion_model.layers.0.attention.to_out.0.lora_B.weight": torch.full((4, rank), 44, dtype=torch.bfloat16),
        "diffusion_model.layers.0.feed_forward.w1.lora_A.weight": torch.full((rank, in_features), 3, dtype=torch.bfloat16),
        "diffusion_model.layers.0.feed_forward.w1.lora_B.weight": torch.full((4, rank), 55, dtype=torch.bfloat16),
        "diffusion_model.layers.0.feed_forward.w3.lora_A.weight": torch.full((rank, in_features), 3, dtype=torch.bfloat16),
        "diffusion_model.layers.0.feed_forward.w3.lora_B.weight": torch.full((4, rank), 66, dtype=torch.bfloat16),
        "diffusion_model.layers.0.feed_forward.w2.lora_A.weight": torch.full((rank, 8), 4, dtype=torch.bfloat16),
        "diffusion_model.layers.0.feed_forward.w2.lora_B.weight": torch.full((4, rank), 77, dtype=torch.bfloat16),
        "diffusion_model.layers.0.adaLN_modulation.0.lora_A.weight": torch.full((rank, 4), 5, dtype=torch.bfloat16),
        "diffusion_model.layers.0.adaLN_modulation.0.lora_B.weight": torch.full((6, rank), 88, dtype=torch.bfloat16),
    }

    converted = convert_to_nunchaku_zimage_lowrank_dict(
        lora,
        base_model=base_model,
        base_unquantized_sd=base_unquantized_sd,
    )

    quantized = converted["quantized"]
    unquantized = converted["unquantized"]
    debug = converted["debug"]

    qkv_down, qkv_up = quantized["layers.0.attention.fused_module"]
    assert qkv_down.shape == (rank, in_features)
    assert qkv_up.shape == (12, rank)
    assert torch.all(qkv_up[:4] == 11)
    assert torch.all(qkv_up[4:8] == 22)
    assert torch.all(qkv_up[8:12] == 33)

    swiglu_down, swiglu_up = quantized["layers.0.feed_forward.net.0.proj"]
    assert swiglu_down.shape == (rank, in_features)
    assert swiglu_up.shape == (8, rank)
    assert torch.all(swiglu_up[:4] == 66)
    assert torch.all(swiglu_up[4:8] == 55)

    assert "layers.0.attention.to_out.0" in quantized
    assert "layers.0.feed_forward.net.2" in quantized
    assert "layers.0.adaLN_modulation.0.lora_A.weight" in unquantized
    assert "layers.0.adaLN_modulation.0.lora_B.weight" in unquantized

    assert debug["quantized_module_matches"] == 4
    assert debug["unquantized_weight_matches"] == 1
    assert debug["unmatched_prefix_count"] == 0
