"""AdamW optimizer for NexteraBERT pre-training (OptiBERT recipe).

OptiBERT (Dervishi et al., EMNLP 2025, "Training compute-optimal transformer
encoder models", Appendix A) pretrains with *standard* AdamW
(Loshchilov & Hutter): fixed weight decay 0.1, betas (0.9, 0.95), gradient
clipping 1.0, linear warmup then cosine decay to 10% of the peak LR.  The
paper does not state eps; 1e-8 follows NeoBERT (whose lr/batch the
OptiBERTneo runs reuse) and Meta Lingua, the codebase OptiBERT builds on —
both use ``torch.optim.AdamW`` with its default eps.  This is the default
(``decoupled=False``).

Note the weight-decay semantics: standard AdamW decays ``lr * wd * p`` per
step, so the paper's ``wd = 0.1`` is mild (~5e-5/step at lr 5e-4).  Do NOT
carry this value into a per-step-decoupled variant (Composer/MosaicBERT
style, ``(lr/initial_lr) * wd`` per step), where 0.1 would decay the weights
10%/step and stall training — and vice versa, that variant's ``wd = 1e-5``
is effectively no regularisation under standard AdamW.

``decoupled=True`` selects the ModernBERT recipe instead (Warner et al.,
2024, Appendix A / Table 3): betas (0.90, 0.98), eps 1e-6, weight decay 1e-5
applied *fully decoupled* and exempted from biases and normalization layers
(Appendix A.3).  Fully decoupled means the per-step shrink is
``schedule_factor * wd * p`` — independent of the LR's magnitude — whereas
``torch.optim.AdamW`` shrinks by ``lr_t * wd * p``; since our schedulers set
``lr_t = lr_peak * schedule_factor``, passing ``wd / lr_peak`` to torch makes
the two identical step for step.  (ModernBERT's StableAdamW update clipping
is not implemented; ``--grad_clip`` stabilises instead.)

Bias correction: both recipes run ``torch.optim.AdamW``, which ALWAYS applies
Adam's bias-correction terms — there is no switch to disable them.  This is
deliberate.  Zhang et al. (2021, "Revisiting Few-sample BERT Fine-tuning")
identify the debiasing omission in the legacy BERTAdam optimizer as the main
cause of degenerate few-sample fine-tuning runs, so no BERTAdam-style
optimizer should be substituted here or in scripts/evaluate_glue.py.
"""

from __future__ import annotations

import torch

# Substrings marking parameters excluded from weight decay under the
# ModernBERT recipe (Appendix A.3: "we did not apply weight decay to the bias
# terms or normalization layers").  Everything with ndim < 2 is excluded
# anyway, which already covers biases and per-channel norm scales; the tags
# catch the remaining norm-like parameters (DyT's alpha/beta/gamma).
NO_DECAY_TAGS = ("bias", "norm", "alpha", "beta", "gamma", "dyt")


def build_optimizer(model, lr, betas, weight_decay, eps=1e-8, decoupled=False):
    """Standard ``torch.optim.AdamW``; the fused CUDA kernel when available.

    ``decoupled=False`` (OptiBERT, the default) puts every trainable parameter
    in a SINGLE group with ``weight_decay`` as given — matching Meta Lingua /
    OptiBERT, which decay every parameter.

    ``decoupled=True`` (ModernBERT) splits biases and normalization parameters
    into a second, decay-free group and converts ``weight_decay`` to the
    equivalent torch coefficient (``wd / lr``) so the per-step shrink matches
    ModernBERT's fully decoupled convention.
    """
    if not decoupled:
        params = [p for p in model.parameters() if p.requires_grad]
        fused = bool(params) and all(p.is_cuda for p in params)
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps,
                                 weight_decay=weight_decay, fused=fused)

    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or any(tag in name.lower() for tag in NO_DECAY_TAGS):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    wd = weight_decay / lr if lr > 0 else weight_decay
    params = decay_params + no_decay_params
    fused = bool(params) and all(p.is_cuda for p in params)
    return torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": wd},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr, betas=betas, eps=eps, weight_decay=wd, fused=fused)


__all__ = ["build_optimizer", "NO_DECAY_TAGS"]
