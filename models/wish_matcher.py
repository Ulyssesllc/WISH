"""Hungarian matcher for WISH (paper Eq. 11-14).

Total cost = alpha * cost_cls + beta * cost_prompt + gamma * cost_mask.
  - cost_cls    = -p_softmax[c_gt]                              (vendor)
  - cost_prompt = KLD( softmax(Ẑ_i / T), softmax(z_j / T) )      (Eq. 12)
  - cost_mask   = min over n in {1,2,3} of d(M̂_i, M_SAM^{j,n})   (Eq. 13)
                  d = cost_bce * sigmoid-CE + cost_dice * dice

The chosen SAM-mask index per matched pair is stored on the target dict as
`matched_sam_idx` so the criterion can reuse it without re-computing argmin.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample

from mask2former.modeling.matcher import batch_sigmoid_ce_loss_jit, batch_dice_loss_jit


def batch_prompt_kld(pred: torch.Tensor, tgt: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Pairwise KLD cost between predicted prompt latents and GT.

    pred: [Nq, D],  tgt: [Ntgt, D]   -> cost [Nq, Ntgt].
    KLD( softmax(pred/T) || softmax(tgt/T) ).
    """
    log_p = F.log_softmax(pred / temperature, dim=-1)  # [Nq, D]
    log_q = F.log_softmax(tgt / temperature, dim=-1)   # [Ntgt, D]
    p = log_p.exp()                                    # [Nq, D]
    cost = (p.unsqueeze(1) * (log_p.unsqueeze(1) - log_q.unsqueeze(0))).sum(-1)
    return cost


class WISHMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_prompt: float = 5.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        cost_bce: float = 5.0,
        num_points: int = 12544,
        prompt_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_prompt = cost_prompt
        # `cost_mask` is the paper's γ. Keep it as an outer multiplier so the
        # BCE/Dice split (vendor convention) can remain configurable while we
        # also apply the paper's γ scaling to the combined mask cost.
        self.cost_mask_outer = float(cost_mask)
        self.cost_dice = cost_dice
        self.cost_bce = cost_bce
        self.num_points = num_points
        self.prompt_temperature = prompt_temperature

    @torch.no_grad()
    def forward(self, outputs, targets):
        """outputs:
              pred_logits  [B, Nq, C+1]
              pred_masks   [B, Nq, H, W]
              pred_prompts [B, Nq, D]
           targets[b]:
              labels    [k]
              prompts   [k, D]
              sam_masks [k, 3, H, W]
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []

        for b in range(bs):
            tgt = targets[b]
            tgt_ids = tgt["labels"]
            k = int(tgt_ids.shape[0])
            if k == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.int64),
                        torch.empty(0, dtype=torch.int64),
                    )
                )
                tgt["matched_sam_idx"] = torch.empty(0, dtype=torch.long)
                continue

            # ---- class cost ------------------------------------------------
            out_prob = outputs["pred_logits"][b].softmax(-1)  # [Nq, C+1]
            cost_class = -out_prob[:, tgt_ids]                # [Nq, k]

            # ---- prompt cost (Eq. 12) -------------------------------------
            pred_pr = outputs["pred_prompts"][b]              # [Nq, D]
            tgt_pr = tgt["prompts"].to(pred_pr)               # [k, D]
            cost_prompt = batch_prompt_kld(pred_pr, tgt_pr, self.prompt_temperature)

            # ---- mask cost (Eq. 13: min over 3 SAM masks) -----------------
            out_mask = outputs["pred_masks"][b]               # [Nq, H, W]
            sam_masks = tgt["sam_masks"].to(out_mask)         # [k, 3, H, W]

            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)

            out_pts = point_sample(
                out_mask[:, None],
                point_coords.repeat(num_queries, 1, 1),
                align_corners=False,
            ).squeeze(1)  # [Nq, num_points]

            sam_flat = sam_masks.flatten(0, 1)[:, None]       # [k*3, 1, H, W]
            sam_pts = point_sample(
                sam_flat,
                point_coords.repeat(k * 3, 1, 1),
                align_corners=False,
            ).squeeze(1).reshape(k, 3, self.num_points)

            with autocast(enabled=False):
                out_pts = out_pts.float()
                sam_pts = sam_pts.float()
                lvl_costs = []
                for n in range(3):
                    tgt_pts = sam_pts[:, n]
                    c_ce = batch_sigmoid_ce_loss_jit(out_pts, tgt_pts)
                    c_dice = batch_dice_loss_jit(out_pts, tgt_pts)
                    lvl_costs.append(self.cost_bce * c_ce + self.cost_dice * c_dice)
                cost_mask_stack = torch.stack(lvl_costs, dim=-1)  # [Nq, k, 3]
                cost_mask, mask_idx = cost_mask_stack.min(dim=-1)  # [Nq, k]

            # Apply outer mask multiplier (paper's γ) to the per-level BCE/Dice
            # combination so the final cost matches Eq.11-13 semantics.
            C = (
                self.cost_class * cost_class
                + self.cost_prompt * cost_prompt
                + self.cost_mask_outer * cost_mask
            )
            C = C.reshape(num_queries, k).cpu()

            row, col = linear_sum_assignment(C)
            row_t = torch.as_tensor(row, dtype=torch.int64)
            col_t = torch.as_tensor(col, dtype=torch.int64)
            indices.append((row_t, col_t))

            # Stash chosen SAM level per target index for criterion reuse.
            matched = torch.full((k,), -1, dtype=torch.long)
            matched[col_t] = mask_idx[row_t, col_t].detach().cpu()
            tgt["matched_sam_idx"] = matched

        return indices