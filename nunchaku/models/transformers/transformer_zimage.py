"""
This module provides Nunchaku ZImageTransformer2DModel and its building blocks in Python.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_z_image import FeedForward as ZImageFeedForward
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel, ZImageTransformerBlock
from huggingface_hub import utils

from nunchaku.models.unets.unet_sdxl import NunchakuSDXLFeedForward

from ...ops.gemm import svdq_gemm_w4a4_cuda
from ...ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
from ...utils import get_precision, load_state_dict_in_safetensors, pad_tensor
from ..attention import NunchakuBaseAttention
from ..attention_processors.zimage import NunchakuZSingleStreamAttnProcessor
from ..embeddings import pack_rotemb
from ..linear import SVDQW4A4Linear
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin, convert_fp16, patch_scale_key


class NunchakuZImageRopeHook:
    """
    Hook class for caching and substition of packed `freqs_cis` tensor.
    """

    def __init__(self):
        self.packed_cache = {}

    def __call__(self, module: nn.Module, input_args: tuple, input_kwargs: dict):
        freqs_cis: torch.Tensor = input_kwargs.get("freqs_cis", None)
        if freqs_cis is None:
            return None
        cache_key = freqs_cis.data_ptr()
        packed_freqs_cis = self.packed_cache.get(cache_key, None)
        if packed_freqs_cis is None:
            packed_freqs_cis = torch.view_as_real(freqs_cis).unsqueeze(3)
            packed_freqs_cis = torch.flip(packed_freqs_cis, dims=[-1])
            packed_freqs_cis = pack_rotemb(pad_tensor(packed_freqs_cis, 256, 1))
            self.packed_cache[cache_key] = packed_freqs_cis
        new_input_kwargs = input_kwargs.copy()
        new_input_kwargs["freqs_cis"] = packed_freqs_cis
        return input_args, new_input_kwargs


class NunchakuZImageFusedModule(nn.Module):
    """
    Fused module for quantized QKV projection, RMS normalization, and rotary embedding for ZImage attention.

    Parameters
    ----------
    qkv : SVDQW4A4Linear
        Quantized QKV projection layer.
    norm_q : RMSNorm
        RMSNorm for query.
    norm_k : RMSNorm
        RMSNorm for key.
    """

    def __init__(self, qkv: SVDQW4A4Linear, norm_q: RMSNorm, norm_k: RMSNorm):
        super().__init__()
        for name, param in qkv.named_parameters(prefix="qkv_"):
            setattr(self, name.replace(".", ""), param)
        self.qkv_precision = qkv.precision
        self.qkv_out_features = qkv.out_features
        for name, param in norm_q.named_parameters(prefix="norm_q_"):
            setattr(self, name.replace(".", ""), param)
        for name, param in norm_k.named_parameters(prefix="norm_k_"):
            setattr(self, name.replace(".", ""), param)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None):
        """
        Fuse QKV projection, RMS normalizaion and rotary embedding.

        Parameters
        ----------
        x : torch.Tensor
            The hidden states tensor
        freqs_cis : torch.Tensor, optional
            The rotary embedding tensor

        Returns
        -------
        The projection results of q, k, v. q result and k result are RMS-normalized and applied RoPE.
        """
        batch_size, seq_len, channels = x.shape
        x = x.view(batch_size * seq_len, channels)
        quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x,
            lora_down=self.qkv_proj_down,
            smooth=self.qkv_smooth_factor,
            fp4=self.qkv_precision == "nvfp4",
            pad_size=256,
        )
        output = torch.empty(batch_size * seq_len, self.qkv_out_features, dtype=x.dtype, device=x.device)
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=self.qkv_qweight,
            out=output,
            ascales=ascales,
            wscales=self.qkv_wscales,
            lora_act_in=lora_act_out,
            lora_up=self.qkv_proj_up,
            bias=getattr(self, "qkv_bias", None),
            fp4=self.qkv_precision == "nvfp4",
            alpha=1.0 if self.qkv_precision == "nvfp4" else None,
            wcscales=self.qkv_wcscales if self.qkv_precision == "nvfp4" else None,
            norm_q=self.norm_q_weight,
            norm_k=self.norm_k_weight,
            rotary_emb=freqs_cis,
        )

        output = output.view(batch_size, seq_len, -1)
        return output


class NunchakuZImageAttention(NunchakuBaseAttention):
    """
    Nunchaku-optimized Attention module for ZImage with quantized and fused QKV projections.

    Parameters
    ----------
    other : Attention
        The original Attention module in ZImage model.
    processor : str, optional
        The attention processor to use ("flashattn2" or "nunchaku-fp16").
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, orig_attn: Attention, processor: str = "flashattn2", **kwargs):
        super(NunchakuZImageAttention, self).__init__(processor)
        self.inner_dim = orig_attn.inner_dim
        self.query_dim = orig_attn.query_dim
        self.use_bias = orig_attn.use_bias
        self.dropout = orig_attn.dropout
        self.out_dim = orig_attn.out_dim
        self.context_pre_only = orig_attn.context_pre_only
        self.pre_only = orig_attn.pre_only
        self.heads = orig_attn.heads
        self.rescale_output_factor = orig_attn.rescale_output_factor
        self.is_cross_attention = orig_attn.is_cross_attention

        # region sub-modules
        self.norm_q = orig_attn.norm_q
        self.norm_k = orig_attn.norm_k
        with torch.device("meta"):
            to_qkv = fuse_linears([orig_attn.to_q, orig_attn.to_k, orig_attn.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)
        self.to_out = orig_attn.to_out
        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)
        # end of region

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for NunchakuZImageAttention.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor.
        encoder_hidden_states : torch.Tensor, optional
            Encoder hidden states for cross-attention.
        attention_mask : torch.Tensor, optional
            Attention mask.
        **cross_attention_kwargs
            Additional arguments for cross attention.

        Returns
        -------
        Output of the attention processor.
        """
        return self.processor(
            attn=self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def set_processor(self, processor: str):
        """
        Set the attention processor.

        Parameters
        ----------
        processor : str
            Name of the processor ("flashattn2").

            - ``"flashattn2"``: Standard FlashAttention-2. See :class:`~nunchaku.models.attention_processors.zimage.NunchakuZSingleStreamAttnProcessor`.

        Raises
        ------
        ValueError
            If the processor is not supported.
        """
        if processor == "flashattn2":
            self.processor = NunchakuZSingleStreamAttnProcessor()
        else:
            raise ValueError(f"Processor {processor} is not supported")


def _convert_z_image_ff(z_ff: ZImageFeedForward) -> FeedForward:
    """
    Replace custom FeedForward module in `ZImageTransformerBlock`s with standard FeedForward in diffusers lib.

    Parameters
    ----------
    z_ff : ZImageFeedForward
        The feed forward sub-module in the ZImageTransformerBlock module

    Returns
    -------
    FeedForward
        A diffusers FeedForward module which is equivalent to the input `z_ff`

    """
    assert isinstance(z_ff, ZImageFeedForward)
    assert z_ff.w1.in_features == z_ff.w3.in_features
    assert z_ff.w1.out_features == z_ff.w3.out_features
    assert z_ff.w1.out_features == z_ff.w2.in_features
    converted_ff = FeedForward(
        dim=z_ff.w1.in_features,
        dim_out=z_ff.w2.out_features,
        dropout=0.0,
        activation_fn="swiglu",
        inner_dim=z_ff.w2.in_features,
        bias=False,
    ).to(dtype=z_ff.w1.weight.dtype, device=z_ff.w1.weight.device)
    return converted_ff


def replace_fused_module(module, incompatible_keys):
    assert isinstance(module, NunchakuZImageAttention)
    module.fused_module = NunchakuZImageFusedModule(module.to_qkv, module.norm_q, module.norm_k)
    del module.to_qkv
    del module.norm_q
    del module.norm_k


class NunchakuZImageFeedForward(NunchakuSDXLFeedForward):
    """
    Quantized feed-forward block for :class:`NunchakuZImageTransformerBlock`.

    Replaces linear layers in a FeedForward block with :class:`~nunchaku.models.linear.SVDQW4A4Linear` for quantized inference.

    Parameters
    ----------
    ff : FeedForward
        Source ZImage FeedForward module to quantize.
    **kwargs :
        Additional arguments for SVDQW4A4Linear.
    """

    def __init__(self, ff: ZImageFeedForward, **kwargs):
        converted_ff = _convert_z_image_ff(ff)
        # forward pass are equivalent to NunchakuSDXLFeedForward
        NunchakuSDXLFeedForward.__init__(self, converted_ff, **kwargs)


def pack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    """
    Pack a low-rank weight tensor into the layout expected by Nunchaku's
    fused W4A4 kernels.

    Parameters
    ----------
    weight : torch.Tensor
        Unpacked low-rank weight tensor.
    down : bool
        True for the down-projection (rank x in), False for the up-projection
        (out x rank).
    """
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


def _fuse_lora_pairs(
    lora_pairs: List[tuple[torch.Tensor, torch.Tensor]],
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


def _replace_module_parameter(module: nn.Module, attr_name: str, tensor: torch.Tensor):
    old_param = getattr(module, attr_name)
    new_param = nn.Parameter(
        tensor.to(device=old_param.device, dtype=old_param.dtype),
        requires_grad=old_param.requires_grad,
    )
    setattr(module, attr_name, new_param)


def _zimage_lora_attrs(module: nn.Module) -> tuple[str, str] | None:
    if isinstance(module, SVDQW4A4Linear):
        return "proj_down", "proj_up"
    if isinstance(module, NunchakuZImageFusedModule):
        return "qkv_proj_down", "qkv_proj_up"
    return None


def _normalize_external_lora_key(key: str) -> str:
    for prefix in ("diffusion_model.",):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


class NunchakuZImageTransformer2DModel(ZImageTransformer2DModel, NunchakuModelLoaderMixin):
    """
    Nunchaku-optimized ZImageTransformer2DModel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._quantized_part_sd: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._quantized_part_loras: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._unquantized_part_sd: Dict[str, torch.Tensor] = {}
        self._unquantized_part_loras: Dict[str, torch.Tensor] = {}

    def _patch_model(self, skip_refiners: bool = False, **kwargs):
        """
        Patch the model by replacing attention and feed_forward modules in the orginal ZImageTransformerBlock.

        Parameters
        ----------
        skip_refiners: bool
            Default to `False`
            if `True`, transformer blocks of `noise_refiner` and `context_refiner` will NOT be replaced.
        **kwargs
            Additional arguments for quantization.

        Returns
        -------
        self : NunchakuZImageTransformer2DModel
            The patched model.
        """

        def _patch_transformer_block(block_list: List[ZImageTransformerBlock]):
            for _, block in enumerate(block_list):
                block.attention = NunchakuZImageAttention(block.attention, **kwargs)
                block.attention.register_load_state_dict_post_hook(replace_fused_module)
                block.feed_forward = NunchakuZImageFeedForward(block.feed_forward, **kwargs)

        def _convert_feed_forward(block_list: List[ZImageTransformerBlock]):
            for _, block in enumerate(block_list):
                block.feed_forward = _convert_z_image_ff(block.feed_forward)

        self.skip_refiners = skip_refiners
        _patch_transformer_block(self.layers)
        if skip_refiners:
            _convert_feed_forward(self.noise_refiner)
            _convert_feed_forward(self.context_refiner)
        else:
            _patch_transformer_block(self.noise_refiner)
            _patch_transformer_block(self.context_refiner)
        return self

    def register_rope_hook(self, rope_hook: NunchakuZImageRopeHook):
        self.rope_hook_handles = []
        for _, ly in enumerate(self.layers):
            self.rope_hook_handles.append(ly.attention.register_forward_pre_hook(rope_hook, with_kwargs=True))
        if not self.skip_refiners:
            for _, nr in enumerate(self.noise_refiner):
                self.rope_hook_handles.append(nr.attention.register_forward_pre_hook(rope_hook, with_kwargs=True))
            for _, cr in enumerate(self.context_refiner):
                self.rope_hook_handles.append(cr.attention.register_forward_pre_hook(rope_hook, with_kwargs=True))

    def unregister_rope_hook(self):
        for h in self.rope_hook_handles:
            h.remove()
        self.rope_hook_handles.clear()

    def _capture_base_lora_state(self):
        base_lora_state: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        base_unquantized_state: Dict[str, torch.Tensor] = {}
        for module_name, module in self.named_modules():
            lora_attrs = _zimage_lora_attrs(module)
            if lora_attrs is None:
                if isinstance(module, nn.Linear):
                    if module.weight is not None:
                        base_unquantized_state[f"{module_name}.weight"] = module.weight.detach().clone()
                    if module.bias is not None:
                        base_unquantized_state[f"{module_name}.bias"] = module.bias.detach().clone()
                continue
            down_attr, up_attr = lora_attrs
            lora_down = getattr(module, down_attr, None)
            lora_up = getattr(module, up_attr, None)
            if lora_down is None or lora_up is None:
                continue
            base_lora_state[module_name] = (
                lora_down.detach().clone(),
                lora_up.detach().clone(),
            )
        self._quantized_part_sd = base_lora_state
        self._unquantized_part_sd = base_unquantized_state

    def _update_unquantized_part_lora_params(self, strength: float = 1.0):
        if len(self._unquantized_part_sd) == 0:
            return

        device = next(self.parameters()).device
        new_state_dict = {}
        for key, base_tensor in self._unquantized_part_sd.items():
            base_tensor = base_tensor.to(device)
            self._unquantized_part_sd[key] = base_tensor

            if base_tensor.ndim == 1 and key in self._unquantized_part_loras:
                diff = strength * self._unquantized_part_loras[key].to(device=device, dtype=base_tensor.dtype)
                if diff.shape[0] < base_tensor.shape[0]:
                    diff = torch.cat(
                        [
                            diff,
                            torch.zeros(base_tensor.shape[0] - diff.shape[0], device=device, dtype=base_tensor.dtype),
                        ],
                        dim=0,
                    )
                new_state_dict[key] = base_tensor + diff
            elif (
                base_tensor.ndim == 2
                and key.replace(".weight", ".lora_B.weight") in self._unquantized_part_loras
                and key.replace(".weight", ".lora_A.weight") in self._unquantized_part_loras
            ):
                lora_a = self._unquantized_part_loras[key.replace(".weight", ".lora_A.weight")].to(
                    device=device, dtype=base_tensor.dtype
                )
                lora_b = self._unquantized_part_loras[key.replace(".weight", ".lora_B.weight")].to(
                    device=device, dtype=base_tensor.dtype
                )

                if lora_a.shape[1] < base_tensor.shape[1]:
                    lora_a = torch.cat(
                        [
                            lora_a,
                            torch.zeros(
                                lora_a.shape[0],
                                base_tensor.shape[1] - lora_a.shape[1],
                                device=device,
                                dtype=base_tensor.dtype,
                            ),
                        ],
                        dim=1,
                    )
                if lora_b.shape[0] < base_tensor.shape[0]:
                    lora_b = torch.cat(
                        [
                            lora_b,
                            torch.zeros(
                                base_tensor.shape[0] - lora_b.shape[0],
                                lora_b.shape[1],
                                device=device,
                                dtype=base_tensor.dtype,
                            ),
                        ],
                        dim=0,
                    )

                diff = strength * (lora_b @ lora_a)
                new_state_dict[key] = base_tensor + diff
            else:
                new_state_dict[key] = base_tensor

        self.load_state_dict(new_state_dict, strict=False)

    def _map_external_lora_to_quantized_modules(
        self,
        path_or_state_dict: str | dict[str, torch.Tensor],
    ) -> tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor]]:
        if isinstance(path_or_state_dict, dict):
            state_dict = dict(path_or_state_dict)
        else:
            state_dict = load_state_dict_in_safetensors(path_or_state_dict, device="cpu")

        grouped_loras: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, value in state_dict.items():
            key = _normalize_external_lora_key(key)
            if ".lora_A." in key:
                prefix = key.split(".lora_A.", 1)[0]
                grouped_loras.setdefault(prefix, {})["A"] = value
            elif ".lora_B." in key:
                prefix = key.split(".lora_B.", 1)[0]
                grouped_loras.setdefault(prefix, {})["B"] = value

        quantized_loras: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        unquantized_loras: Dict[str, torch.Tensor] = {}
        qkv_groups: Dict[str, Dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        swiglu_groups: Dict[str, Dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}

        def _store_single(module_name: str, pair: tuple[torch.Tensor, torch.Tensor]):
            lora_down, lora_up = pair
            quantized_loras[module_name] = (
                pack_lowrank_weight(lora_down, down=True),
                pack_lowrank_weight(lora_up, down=False),
            )

        for prefix, pair in grouped_loras.items():
            if "A" not in pair or "B" not in pair:
                continue
            lora_down = pair["A"].contiguous()
            lora_up = pair["B"].contiguous()
            module_pair = (lora_down, lora_up)

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
            elif prefix.endswith(".feed_forward.w1"):
                target = prefix[: -len(".feed_forward.w1")] + ".feed_forward.net.0.proj"
                swiglu_groups.setdefault(target, {})[0] = module_pair
            elif prefix.endswith(".feed_forward.w3"):
                target = prefix[: -len(".feed_forward.w3")] + ".feed_forward.net.0.proj"
                swiglu_groups.setdefault(target, {})[1] = module_pair
            elif prefix.endswith(".feed_forward.w2"):
                target = prefix[: -len(".feed_forward.w2")] + ".feed_forward.net.2"
                _store_single(target, module_pair)
            elif f"{prefix}.weight" in self._unquantized_part_sd:
                unquantized_loras[f"{prefix}.lora_A.weight"] = lora_down
                unquantized_loras[f"{prefix}.lora_B.weight"] = lora_up

        for module_name, parts in qkv_groups.items():
            if set(parts.keys()) != {0, 1, 2}:
                continue
            lora_down, lora_up = _fuse_lora_pairs([parts[0], parts[1], parts[2]])
            quantized_loras[module_name] = (
                pack_lowrank_weight(lora_down, down=True),
                pack_lowrank_weight(lora_up, down=False),
            )

        for module_name, parts in swiglu_groups.items():
            if set(parts.keys()) != {0, 1}:
                continue
            # diffusers' SwiGLU proj is ordered as [hidden, gate], while the
            # original Z-Image feed-forward uses w2(silu(w1(x)) * w3(x)).
            # That means:
            #   - hidden chunk should map to w3
            #   - gate chunk should map to w1
            # So the fused LoRA order for net.0.proj must be [w3, w1].
            lora_down, lora_up = _fuse_lora_pairs([parts[1], parts[0]])
            quantized_loras[module_name] = (
                pack_lowrank_weight(lora_down, down=True),
                pack_lowrank_weight(lora_up, down=False),
            )

        return quantized_loras, unquantized_loras

    def _apply_quantized_lora_params(self, strength: float = 1.0):
        for module_name, (base_down, base_up) in self._quantized_part_sd.items():
            module = self.get_submodule(module_name)
            lora_attrs = _zimage_lora_attrs(module)
            if lora_attrs is None:
                continue
            down_attr, up_attr = lora_attrs
            if module_name in self._quantized_part_loras:
                extra_down, extra_up = self._quantized_part_loras[module_name]
                merged_down = torch.cat([base_down, extra_down], dim=1)
                merged_up = torch.cat([base_up, extra_up * strength], dim=1)
            else:
                merged_down = base_down
                merged_up = base_up
            _replace_module_parameter(module, down_attr, merged_down)
            _replace_module_parameter(module, up_attr, merged_up)
            if hasattr(module, "rank"):
                module.rank = merged_down.shape[1]

    def update_lora_params(self, path_or_state_dict: str | dict[str, torch.Tensor]):
        if len(self._quantized_part_sd) == 0:
            self._capture_base_lora_state()
        quantized_loras, unquantized_loras = self._map_external_lora_to_quantized_modules(path_or_state_dict)
        if len(quantized_loras) == 0 and len(unquantized_loras) == 0:
            raise ValueError("No Z-Image LoRA weights matched the current Nunchaku transformer modules.")
        self._quantized_part_loras = quantized_loras
        self._unquantized_part_loras = unquantized_loras
        self._apply_quantized_lora_params(1.0)
        self._update_unquantized_part_lora_params(1.0)

    def set_lora_strength(self, strength: float = 1.0):
        if len(self._quantized_part_sd) == 0:
            self._capture_base_lora_state()
        self._apply_quantized_lora_params(strength)
        self._update_unquantized_part_lora_params(strength)

    def reset_lora(self):
        self._quantized_part_loras = {}
        self._unquantized_part_loras = {}
        if len(self._quantized_part_sd) == 0:
            self._capture_base_lora_state()
        self._apply_quantized_lora_params(1.0)
        self._update_unquantized_part_lora_params(1.0)

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        return_dict: bool = True,
    ):
        """
        Adapted from diffusers.models.transformers.transformer_z_image.ZImageTransformer2DModel#forward

        Register pre-forward hooks for caching and substitution of packed `freqs_cis` tensor for all attention submodules and unregister after forwarding is done.
        """
        rope_hook = NunchakuZImageRopeHook()
        self.register_rope_hook(rope_hook)
        try:
            return super().forward(x, t, cap_feats, patch_size, f_patch_size, return_dict)
        finally:
            self.unregister_rope_hook()
            del rope_hook

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Load a pretrained NunchakuZImageTransformer2DModel from a safetensors file.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the safetensors file. It can be a local file or a remote HuggingFace path.
        **kwargs
            Additional arguments (e.g., device, torch_dtype).

        Returns
        -------
        NunchakuZImageTransformer2DModel
            The loaded and quantized model.

        Raises
        ------
        NotImplementedError
            If offload is requested.
        AssertionError
            If the file is not a safetensors file.
        """
        device = kwargs.get("device", "cpu")
        offload = kwargs.get("offload", False)

        if offload:
            raise NotImplementedError("Offload is not supported for ZImageTransformer2DModel")

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"
        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))

        rank = quantization_config.get("rank", 32)
        skip_refiners = quantization_config.get("skip_refiners", False)
        transformer = transformer.to(torch_dtype)

        precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"

        print(f"quantization_config: {quantization_config}, rank={rank}, skip_refiners={skip_refiners}")

        transformer._patch_model(skip_refiners=skip_refiners, precision=precision, rank=rank, **kwargs)
        transformer = transformer.to_empty(device=device)

        patch_scale_key(transformer, model_state_dict)
        if torch_dtype == torch.float16:
            convert_fp16(transformer, model_state_dict)

        transformer.load_state_dict(model_state_dict)
        transformer._capture_base_lora_state()

        return transformer
