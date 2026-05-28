"""WISH SetCriterion: classification + prompt + adaptive-SAM mask loss (Eq. 15).

Padded GT instances of class no-object (∅) contribute only the class term —
no prompt/mask supervision (consistent with vendor Mask2Former).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from mask2former.modeling.criterion import (
    dice_loss_jit,
    sigmoid_ce_loss_jit,
    calculate_uncertainty,
)


class WISHSetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: dict,
        eos_coef: float,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    # ---- losses --------------------------------------------------------------
    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"].float()
        idx = self._src_perm(indices)
        target_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target = torch.full(src_logits.shape[:2], self.num_classes,
                            dtype=torch.int64, device=src_logits.device)
        target[idx] = target_o
        return {"loss_ce": F.cross_entropy(src_logits.transpose(1, 2), target, self.empty_weight)}

    def loss_prompts(self, outputs, targets, indices):
        """KLD(Ẑ_i || z_gt) over matched pairs (Eq. 12 as a loss, paper Eq. 15)."""
        src_idx = self._src_perm(indices)
        pred = outputs["pred_prompts"][src_idx]
        if pred.shape[0] == 0:
            return {"loss_prompt": pred.sum() * 0.0}
        tgt = torch.cat([t["prompts"][J] for t, (_, J) in zip(targets, indices)], dim=0)
        log_p = F.log_softmax(pred, dim=-1)
        log_q = F.log_softmax(tgt.to(pred), dim=-1)
        kld = (log_p.exp() * (log_p - log_q)).sum(-1)
        return {"loss_prompt": kld.mean()}

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Mask loss against the SAM mask chosen by the matcher (lowest cost of 3).

        Falls back to the middle level (index 1) if `matched_sam_idx` is missing
        (e.g. aux outputs whose matcher pass didn't stash it).
        """
        src_idx = self._src_perm(indices)
        src_masks = outputs["pred_masks"][src_idx]  # [N_matched, H, W]

        tgt_masks_list = []
        for b, (_, J) in enumerate(indices):
            if J.numel() == 0:
                continue
            sam_masks_b = targets[b]["sam_masks"]  # [k, 3, H, W]
            chosen = targets[b].get("matched_sam_idx", None)
            if chosen is not None and (chosen[J] >= 0).all():
                lvl = chosen[J].to(sam_masks_b.device)
            else:
                lvl = torch.ones(J.shape[0], dtype=torch.long, device=sam_masks_b.device)
            tgt_masks_list.append(sam_masks_b[J.to(sam_masks_b.device), lvl])

        if not tgt_masks_list or src_masks.shape[0] == 0:
            zero = src_masks.sum() * 0.0 if src_masks.numel() > 0 else outputs["pred_masks"].sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}

        target_masks = torch.cat(tgt_masks_list, dim=0).to(src_masks)

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = point_sample(
                target_masks, point_coords, align_corners=False
            ).squeeze(1)

        point_logits = point_sample(
            src_masks, point_coords, align_corners=False
        ).squeeze(1)

        return {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

    # ---- helpers -------------------------------------------------------------
    def _src_perm(self, indices):
        bi = torch.cat([torch.full_like(s, i) for i, (s, _) in enumerate(indices)])
        si = torch.cat([s for (s, _) in indices])
        return bi, si

    def forward(self, outputs, targets):
        outputs_no_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_no_aux, targets)
        num_masks = max(sum(int(len(t["labels"])) for t in targets), 1)

        losses = {}
        losses.update(self.loss_labels(outputs_no_aux, targets, indices))
        losses.update(self.loss_prompts(outputs_no_aux, targets, indices))
        losses.update(self.loss_masks(outputs_no_aux, targets, indices, num_masks))

        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux, targets)
                for k, v in self.loss_labels(aux, targets, aux_indices).items():
                    losses[f"{k}_{i}"] = v
                for k, v in self.loss_prompts(aux, targets, aux_indices).items():
                    losses[f"{k}_{i}"] = v
                for k, v in self.loss_masks(aux, targets, aux_indices, num_masks).items():
                    losses[f"{k}_{i}"] = v
        return losses