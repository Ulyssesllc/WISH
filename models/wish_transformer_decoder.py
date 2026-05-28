"""WISH transformer decoder: vanilla Mask2Former decoder + a prompt-prediction head.

Adds H_prompt (MLP) emitting predicted SAM prompt latents Ẑ in R^{Nq x 256}
(paper Eq. 10). Class & mask heads stay identical to the vendor decoder so
existing pretrained weights load cleanly.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from detectron2.config import configurable

from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import (
    TRANSFORMER_DECODER_REGISTRY,
)
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MLP,
    MultiScaleMaskedTransformerDecoder,
)


@TRANSFORMER_DECODER_REGISTRY.register()
class WISHMultiScaleMaskedTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    """Extends Mask2Former decoder with `prompt_embed` head producing [B, Nq, prompt_dim]."""

    @configurable
    def __init__(self, *args, prompt_dim: int = 256, prompt_head_layers: int = 3,
                 prompt_head_hidden: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = kwargs.get("hidden_dim") or args[0]  # detectron2 configurable passes via kwargs
        self.prompt_embed = MLP(
            input_dim=hidden_dim,
            hidden_dim=prompt_head_hidden,
            output_dim=prompt_dim,
            num_layers=prompt_head_layers,
        )

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = super().from_config(cfg, in_channels, mask_classification)
        ret["prompt_dim"] = cfg.MODEL.WISH.PROMPT_DIM
        ret["prompt_head_layers"] = cfg.MODEL.WISH.PROMPT_HEAD_LAYERS
        ret["prompt_head_hidden"] = cfg.MODEL.WISH.PROMPT_HEAD_HIDDEN
        return ret

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        # Re-runs parent prediction heads, plus prompt head on the decoder-norm'd output.
        # We can't just call super() because we also need decoder_output for prompt_embed.
        decoder_output = self.decoder_norm(output).transpose(0, 1)  # [B, Nq, C]
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        prompt_pred = self.prompt_embed(decoder_output)  # [B, Nq, prompt_dim]

        attn_mask = torch.nn.functional.interpolate(
            outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False
        )
        attn_mask = (
            attn_mask.sigmoid().flatten(2).unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5
        ).bool().detach()
        return outputs_class, outputs_mask, attn_mask, prompt_pred

    def forward(self, x, mask_features, mask=None):
        # Mirror parent forward but also collect prompt predictions per layer.
        assert len(x) == self.num_feature_levels
        src, pos, size_list = [], [], []
        del mask
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
        _, bs, _ = src[0].shape
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        preds_cls, preds_mask, preds_prompt = [], [], []
        outputs_class, outputs_mask, attn_mask, prompt_pred = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        preds_cls.append(outputs_class)
        preds_mask.append(outputs_mask)
        preds_prompt.append(prompt_pred)

        for i in range(self.num_layers):
            lvl = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            output = self.transformer_cross_attention_layers[i](
                output, src[lvl], memory_mask=attn_mask,
                memory_key_padding_mask=None, pos=pos[lvl], query_pos=query_embed,
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed,
            )
            output = self.transformer_ffn_layers[i](output)
            outputs_class, outputs_mask, attn_mask, prompt_pred = self.forward_prediction_heads(
                output, mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
            )
            preds_cls.append(outputs_class)
            preds_mask.append(outputs_mask)
            preds_prompt.append(prompt_pred)

        return {
            "pred_logits": preds_cls[-1],
            "pred_masks": preds_mask[-1],
            "pred_prompts": preds_prompt[-1],
            "aux_outputs": [
                {"pred_logits": c, "pred_masks": m, "pred_prompts": p}
                for c, m, p in zip(preds_cls[:-1], preds_mask[:-1], preds_prompt[:-1])
            ],
        }
