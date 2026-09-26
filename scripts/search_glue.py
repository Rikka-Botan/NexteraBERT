#!/usr/bin/env python
"""Efficient per-task GLUE hyperparameter search around the ModernBERT recipe,
followed by the final multi-seed evaluation with the winning hyperparameters.

The search space is ModernBERT's own published sweep (Warner et al., 2024,
Appendix E.1): lr in {1e-5, 3e-5, 5e-5, 8e-5}, weight decay in {1e-6, 5e-6,
8e-6, 1e-5} — swept as the torch-AdamW equivalents {0.02, 0.1, 0.16, 0.2}, see
WD_GRID — epochs in {1,2,3} (sst2/mnli/rte) or {2,5,10} (the rest). Three
pruning layers stack on top of each other, so the full 4x4x3 = 48-run-per-task
grid (384 full fine-tunings for standard GLUE) collapses to a small fraction:

  1. Coordinate descent, anchored at ModernBERT-base's Table 6 values: sweep
     one axis at a time (lr -> wd -> epochs), keeping the winner. At most
     4+3+2 = 9 configs per task instead of 48. Under --stages auto (default)
     the large, stable tasks (sst2/qnli/qqp/mnli) sweep lr only — their
     rankings barely move with wd/epochs and their runs dominate the compute.
  2. Successive halving within each sweep: every candidate first runs a cheap
     SCREENING rung — --screen_epochs (default 1) epoch(s), with the LR
     schedule still stretched over the full epoch budget so candidates are
     compared mid-schedule on equal footing — and only the top --promote
     (default 2) are re-run at full budget. The incumbent is always given a
     full-budget score before it can be displaced, and ties go to the
     incumbent, so ModernBERT's published setting must be strictly beaten.
  3. Proxy data for the big tasks: search runs cap the training set at
     --train_subsample examples (default 32768; a seed-fixed random subset, so
     every candidate sees the same data). qqp/mnli/qnli/sst2 candidates then
     cost small-task money. The subset only ranks candidates — the final
     evaluation always runs on full data.

Together: a screening rung on qqp costs ~32k examples instead of ~1.5M for a
full-data fine-tuning — the whole default search is roughly 5-6x faster than
coordinate descent on full data, and cheaper than the final evaluation itself.
--exact turns layers 2 and 3 off (full-data, full-length candidates); the
individual knobs (--screen_epochs 0, --train_subsample 0, --promote,
--stages full) tune the speed/fidelity trade per layer.

Protocol notes:

  * Candidates are scored by ONE fine-tuning run (--seed, default 19 =
    ModernBERT's default_seed) on the dev set, with the same early stopping
    (--patience) as the final evaluation.
  * MNLI transfer is preserved: MNLI is searched first (on the proxy), then
    the transfer source is trained ONCE with the winning config on FULL data,
    and only afterwards are rte/mrpc/stsb searched — initialised from that
    encoder, exactly as in the final protocol. That single full-data MNLI run
    is the search's critical path, but it is not wasted: the final evaluation
    reuses the encoder via --mnli_source instead of retraining it. If MNLI is
    not among --tasks, the source is trained with the anchor recipe.
  * Every rung result is cached in --work_dir (keyed by task, config, rung and
    proxy settings), so an interrupted search resumes for free — rerun the
    same command. Delete the work dir to start from scratch.
  * Candidates run one per GPU (CUDA_VISIBLE_DEVICES pinned), scheduled across
    --num_gpus like evaluate_glue.py --task_parallel; per-task batch sizes are
    totals and stay exact.

The search writes --output (default glue_search.json): per-task winning
hyperparameters at the top level — the exact format `evaluate_glue.py
--hparams` consumes — plus the full search history under "_search". Unless
--no_final is given, the final evaluation is then launched automatically:
multi-seed (ModernBERT's 19 8364 717 10536 90166, with the usual per-task seed
caps), full data, MNLI transfer from the searched source, task-parallel across
the same GPUs, writing --final_output (default glue_results.json).

    # search + final evaluation on 4 GPUs (fast defaults)
    python scripts/search_glue.py --model checkpoints/discriminator --num_gpus 4

    # paper-faithful search: full data, no screening (slow)
    python scripts/search_glue.py --model checkpoints/discriminator --exact

    # quick subset, search only
    python scripts/search_glue.py --model checkpoints/discriminator \
        --tasks rte mrpc stsb cola --no_final

The anchor is evaluate_glue.py's own per-task table; the sweep grids are
ModernBERT's, and the incumbent value is always added to a stage's candidate
set when it falls outside the grid, so the anchor is never displaced unscored.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

# Windows consoles default to a legacy code page (e.g. cp932) that cannot encode
# the em-dashes in the help text.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from evaluate_glue import (  # noqa: E402
    GLUE_AVG_TASKS,
    GLUE_TASKS,
    MNLI_INIT_TASKS,
    TASK_DEFAULTS_MODERNBERT,
)

# ModernBERT's published GLUE sweep (Appendix E.1). The epoch grid is per-task:
# {1,2,3} for sst2/mnli/rte, {2,5,10} for qnli/qqp/cola/mrpc/stsb.
#
# WD_GRID is the paper's {1e-6, 5e-6, 8e-6, 1e-5} converted to the TORCH weight
# decays evaluate_glue.py takes (wd_published / lr, at the grid's central lr 5e-5) —
# the published values fed to torch AdamW verbatim would decay by ~1e-10 a step.
LR_GRID = (1e-5, 3e-5, 5e-5, 8e-5)
WD_GRID = (0.02, 0.1, 0.16, 0.2)
EPOCH_GRIDS = {
    "sst2": (1, 2, 3), "mnli": (1, 2, 3), "rte": (1, 2, 3),
    "qnli": (2, 5, 10), "qqp": (2, 5, 10), "cola": (2, 5, 10),
    "mrpc": (2, 5, 10), "stsb": (2, 5, 10), "wnli": (1, 2, 3),
}

# --stages auto: full coordinate descent (lr -> wd -> epochs) only for the
# small, high-variance tasks; the large stable ones sweep lr alone.
FULL_STAGE_TASKS = ("cola", "mrpc", "stsb", "rte", "wnli")

_MISSING = object()   # sentinel: no cached result on disk


def cfg_key(cfg):
    return (cfg["lr"], cfg["weight_decay"], cfg["epochs"])


def cfg_tag(cfg):
    """Deterministic, filesystem-safe id for a config — the resume-cache key."""
    return f"lr{cfg['lr']:g}_wd{cfg['weight_decay']:g}_ep{cfg['epochs']}"


def fmt_cfg(cfg):
    return (f"lr={cfg['lr']:g} wd={cfg['weight_decay']:g} "
            f"ep={cfg['epochs']} bs={cfg['batch_size']}")


def fmt_score(sc):
    return f"{sc:.4f}" if sc is not None else "n/a"


def load_cached(path):
    """Cached rung score: _MISSING if never run, else the primary metric
    (None when the run completed but produced no usable score)."""
    if not os.path.exists(path):
        return _MISSING
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return _MISSING
    best = data.get("best")
    if not isinstance(best, dict) or "_primary" not in best:
        return None
    return float(best["_primary"])


def is_pruned_marker(path):
    """True if `path` is a prune marker written by open_stage (a candidate that
    lost the screening rung and must stay resolved-but-unscored on resume)."""
    try:
        with open(path, encoding="utf-8") as f:
            return bool(json.load(f).get("pruned"))
    except (OSError, json.JSONDecodeError):
        return False


def stage_candidates(task, best, stage):
    """One-axis sweep around the incumbent. The incumbent's own value is always
    a candidate (prepended when outside the grid, e.g. a hand-tuned anchor), so the
    winner is only ever displaced by a config with a real full-budget score."""
    if stage == "lr":
        axis, grid = "lr", LR_GRID
    elif stage == "wd":
        axis, grid = "weight_decay", WD_GRID
    else:
        axis, grid = "epochs", EPOCH_GRIDS.get(task, (1, 2, 3))
    vals = list(grid)
    if best[axis] not in vals:
        vals = [best[axis]] + vals
    return [dict(best, **{axis: v}) for v in vals]


def hist_entry(cfg, stage, primary, rung, cached=False):
    e = {"stage": stage, "rung": rung, "lr": cfg["lr"],
         "weight_decay": cfg["weight_decay"], "epochs": cfg["epochs"],
         "primary": primary}
    if cached:
        e["cached"] = True
    return e


class TaskSearch:
    """Coordinate-descent state for one task, with a successive-halving
    (screen -> promote) rung pair inside every stage."""

    def __init__(self, task, anchor, stages, args):
        self.task = task
        self.anchor = dict(anchor)
        self.best = dict(anchor)
        self.best_primary = None
        self.stages = list(stages)
        self.stage_idx = 0
        self.scores = {}          # cfg_key -> FULL-budget primary, or None
        self.screen_scores = {}   # cfg_key -> screening-rung primary, or None
        self.history = []
        self.stage_cands = []     # configs of the stage currently in flight
        self.outstanding = 0      # jobs of the current rung still running
        self.done = False
        self.note = None
        self.init = None          # encoder dir candidates initialise from (MNLI transfer)
        self.subsample = args.train_subsample
        self.screen_epochs = args.screen_epochs
        self.promote = max(1, args.promote)

    def full_path(self, work, cfg):
        sub = f"_sub{self.subsample}" if self.subsample else ""
        return work / f"res_{self.task}{sub}_{cfg_tag(cfg)}.json"

    def screen_path(self, work, cfg):
        sub = f"_sub{self.subsample}" if self.subsample else ""
        return work / f"res_{self.task}{sub}_scr{self.screen_epochs}_{cfg_tag(cfg)}.json"


def pick_winner(search, stage):
    """Close a stage: keep the incumbent unless a candidate strictly beats its
    full-budget score. Screen-only (pruned) candidates carry no full score and
    can never win."""
    best_cfg, best_sc = search.best, search.scores.get(cfg_key(search.best))
    for cfg in search.stage_cands:
        sc = search.scores.get(cfg_key(cfg))
        if sc is None:
            continue
        if best_sc is None or sc > best_sc:
            best_cfg, best_sc = cfg, sc
    search.best = dict(best_cfg)
    search.best_primary = best_sc
    print(f"[search] {search.task}: stage '{stage}' -> {fmt_cfg(search.best)} "
          f"(primary={fmt_score(best_sc)})")
    search.stage_idx += 1


def make_job(search, cfg, stage, work, rung):
    path = search.screen_path(work, cfg) if rung == "screen" else search.full_path(work, cfg)
    job = {"kind": "search", "rung": rung, "search": search, "task": search.task,
           "cfg": cfg, "stage": stage, "out": str(path)}
    if search.init:
        job["init"] = search.init
    return job


def open_stage(search, work):
    """Advance the search as far as cached scores allow; return the jobs that
    must actually run for the current rung (empty when the search finished).

    Each stage runs in up to two rungs: a cheap screening pass over every new
    candidate, then full-budget runs for the top --promote of them (plus the
    incumbent, which must always hold a full score before it can be beaten).
    Candidates that lose the screening are pruned: recorded with a None full
    score so they are resolved but can never win."""
    while search.stage_idx < len(search.stages):
        stage = search.stages[search.stage_idx]
        cands = stage_candidates(search.task, search.best, stage)
        search.stage_cands = cands
        need_full = []
        for cfg in cands:
            k = cfg_key(cfg)
            if k in search.scores:
                continue          # full score known (earlier stage, or pruned)
            path = search.full_path(work, cfg)
            cached = load_cached(path)
            if cached is not _MISSING:
                rung = "pruned" if cached is None and is_pruned_marker(path) else "full"
                search.scores[k] = cached
                search.history.append(hist_entry(cfg, stage, cached, rung, cached=True))
                continue
            need_full.append(cfg)
        if search.screen_epochs and len(need_full) > search.promote:
            to_screen = []
            for cfg in need_full:
                k = cfg_key(cfg)
                if k in search.screen_scores:
                    continue
                cached = load_cached(search.screen_path(work, cfg))
                if cached is not _MISSING:
                    search.screen_scores[k] = cached
                    search.history.append(hist_entry(cfg, stage, cached, "screen", cached=True))
                    continue
                to_screen.append(cfg)
            if to_screen:
                search.outstanding = len(to_screen)
                return [make_job(search, c, stage, work, "screen") for c in to_screen]
            # Promote the screening top-k to full budget. sorted() is stable,
            # so ties keep grid order; failed screens (None) rank last.
            ranked = sorted(
                need_full,
                key=lambda c: (search.screen_scores.get(cfg_key(c)) is not None,
                               search.screen_scores.get(cfg_key(c)) or 0.0),
                reverse=True)
            run_full = ranked[:search.promote]
            inc_key = cfg_key(search.best)
            if inc_key not in search.scores and \
                    all(cfg_key(c) != inc_key for c in run_full):
                run_full.append(dict(search.best))   # incumbent always gets a full score
            for cfg in need_full:
                if any(cfg_key(c) == cfg_key(cfg) for c in run_full):
                    continue
                search.scores[cfg_key(cfg)] = None   # pruned: resolved, cannot win
                search.history.append(hist_entry(cfg, stage, None, "pruned"))
                # Persist the prune so a resumed search does not re-run the
                # loser at full budget (the marker parses as an unscored run).
                with open(search.full_path(work, cfg), "w", encoding="utf-8") as f:
                    json.dump({"pruned": True, "task": search.task, "stage": stage,
                               "screen_primary": search.screen_scores.get(cfg_key(cfg))},
                              f)
            pruned = len(need_full) - len(run_full)
            if pruned > 0:
                print(f"[search] {search.task}: stage '{stage}' screening pruned "
                      f"{pruned}/{len(need_full)} candidate(s)")
        else:
            run_full = need_full
        if run_full:
            search.outstanding = len(run_full)
            return [make_job(search, c, stage, work, "full") for c in run_full]
        pick_winner(search, stage)
    search.done = True
    return []


def worker_cmd(args, job):
    """argv for one rung: a single-(task, seed) evaluate_glue.py worker with
    this config's lr / wd / epochs / batch size as explicit overrides. Search
    rungs add the proxy flags; the MNLI transfer source runs at full fidelity."""
    cfg = job["cfg"]
    cmd = [
        sys.executable, str(SCRIPTS / "evaluate_glue.py"),
        "--model", args.model,
        "--tokenizer", args.tokenizer,
        "--max_len", str(args.max_len),
        "--dtype", args.dtype,
        "--patience", str(args.patience),
        "--lr", str(cfg["lr"]),
        "--weight_decay", str(cfg["weight_decay"]),
        "--epochs", str(cfg["epochs"]),
        "--batch_size", str(cfg["batch_size"]),
        "--worker_task", job["task"],
        "--worker_seed", str(args.seed),
        "--worker_out", job["out"],
    ]
    if job.get("rung") in ("screen", "full") and args.train_subsample:
        cmd += ["--train_subsample", str(args.train_subsample)]
    if job.get("rung") == "screen":
        cmd += ["--stop_after_epochs", str(args.screen_epochs)]
    if not args.compile:
        cmd.append("--no_compile")
    if job.get("init"):
        cmd += ["--worker_init", job["init"]]
    if job.get("save_bb"):
        cmd += ["--worker_save_backbone", job["save_bb"]]
    return cmd


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

class Ctx:
    """Shared scheduler state."""

    def __init__(self, args, work, searches, dependents, need_transfer):
        self.args = args
        self.work = work
        self.searches = searches
        self.dependents = dependents          # dependent tasks present in --tasks
        self.need_transfer = need_transfer
        self.mnli_best = work / "mnli_best"
        self.mnli_tag = work / "mnli_best.tag"
        self.queue = []


def release_dependents(ctx, init_path):
    """Unblock the rte/mrpc/stsb searches once the MNLI source is ready.
    init_path None means the source failed: keep their anchors, search nothing
    (searching cold would tune for a different init than the final protocol)."""
    for t in ctx.dependents:
        s = ctx.searches[t]
        if init_path is None:
            s.done = True
            s.note = "mnli transfer source failed; anchor kept unsearched"
            print(f"[search][warn] {t}: {s.note}")
            continue
        s.init = init_path
        jobs = open_stage(s, ctx.work)
        if s.done:
            on_search_done(s, ctx)
        else:
            ctx.queue.extend(jobs)


def ensure_mnli_source(ctx, cfg):
    """Train the MNLI transfer source ONCE with `cfg` on FULL data (search
    candidates only ran on the proxy), or reuse a cached one built with the
    same config. Dependents are released when it is ready."""
    tag = cfg_tag(cfg)
    if ctx.mnli_best.is_dir() and ctx.mnli_tag.exists() and \
            ctx.mnli_tag.read_text(encoding="utf-8").strip() == tag:
        print(f"[search] reusing cached MNLI transfer source ({tag})")
        release_dependents(ctx, str(ctx.mnli_best))
        return
    shutil.rmtree(ctx.mnli_best, ignore_errors=True)
    ctx.mnli_tag.unlink(missing_ok=True)
    out = ctx.work / "res_mnli_source.json"
    out.unlink(missing_ok=True)
    print(f"[search] training the MNLI transfer source on full data "
          f"({fmt_cfg(cfg)}) — reused later by the final evaluation")
    ctx.queue.append({"kind": "mnli_source", "rung": None, "search": None,
                      "task": "mnli", "cfg": dict(cfg), "out": str(out),
                      "save_bb": str(ctx.mnli_best)})


def on_search_done(search, ctx):
    print(f"[search] {search.task}: DONE -> {fmt_cfg(search.best)} "
          f"(primary={fmt_score(search.best_primary)})")
    if search.task == "mnli" and ctx.need_transfer:
        ensure_mnli_source(ctx, search.best)


def on_job_done(job, ret, ctx):
    if ret != 0:
        print(f"[search][error] {job['task']} {cfg_tag(job['cfg'])} "
              f"({job.get('rung') or job['kind']}) exited with code {ret}")
    if job["kind"] == "mnli_source":
        ok = ret == 0 and ctx.mnli_best.is_dir()
        if ok:
            ctx.mnli_tag.write_text(cfg_tag(job["cfg"]), encoding="utf-8")
            # Park this run's own MNLI score inside the exported encoder. The
            # source was trained on FULL data with the winning config and the
            # final evaluation's first seed, so it IS the mnli run the final
            # evaluation would otherwise repeat — evaluate_glue.py --mnli_source
            # picks the score up from here instead of retraining the single most
            # expensive task.
            try:
                shutil.copyfile(job["out"], ctx.mnli_best / "glue_score.json")
            except OSError as e:
                print(f"[search][warn] could not record the MNLI source score: {e}")
            print(f"[search] MNLI transfer source ready: {ctx.mnli_best}")
        release_dependents(ctx, str(ctx.mnli_best) if ok else None)
        return
    search = job["search"]
    primary = load_cached(job["out"]) if ret == 0 else None
    if primary is _MISSING:
        primary = None
    if job["rung"] == "screen":
        search.screen_scores[cfg_key(job["cfg"])] = primary
    else:
        search.scores[cfg_key(job["cfg"])] = primary
    search.history.append(hist_entry(job["cfg"], job["stage"], primary, job["rung"]))
    search.outstanding -= 1
    if search.outstanding > 0:
        return
    jobs = open_stage(search, ctx.work)
    if search.done:
        on_search_done(search, ctx)
    else:
        ctx.queue.extend(jobs)


def run_scheduler(args, ctx, num_gpus):
    """GPU-pool scheduler: one rung per GPU, mirroring --task_parallel."""
    free = list(range(num_gpus))
    running = {}                    # proc -> (job, gpu)
    while ctx.queue or running:
        progressed = False
        while ctx.queue and free:
            job = ctx.queue.pop(0)
            gpu = free.pop()
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            proc = subprocess.Popen(worker_cmd(args, job), env=env)
            running[proc] = (job, gpu)
            progressed = True
            rung = f" [{job['rung']}]" if job.get("rung") else " [transfer source]"
            tag = "  (init: MNLI)" if job.get("init") else ""
            print(f"[search] launch {job['task']} {cfg_tag(job['cfg'])}{rung} "
                  f"-> GPU{gpu}{tag}")
        for proc in list(running):
            ret = proc.poll()
            if ret is None:
                continue
            job, gpu = running.pop(proc)
            free.append(gpu)
            progressed = True
            on_job_done(job, ret, ctx)
        if not progressed:
            time.sleep(2)


def stages_for(task, mode):
    if mode == "full":
        return ("lr", "wd", "epochs")
    if mode == "auto" and task in FULL_STAGE_TASKS:
        return ("lr", "wd", "epochs")
    return ("lr",)


def write_output(args, ctx):
    out = {}
    for task in args.tasks:
        s = ctx.searches[task]
        out[task] = {k: s.best[k]
                     for k in ("lr", "weight_decay", "epochs", "batch_size")
                     if k in s.best}
    out["_search"] = {
        "model": str(args.model), "seed": args.seed,
        "stages": args.stages, "patience": args.patience, "max_len": args.max_len,
        "train_subsample": args.train_subsample,
        "screen_epochs": args.screen_epochs, "promote": args.promote,
        "mnli_transfer": bool(ctx.need_transfer),
        "grid": {"lr": list(LR_GRID), "weight_decay": list(WD_GRID),
                 "epochs": {t: list(EPOCH_GRIDS.get(t, (1, 2, 3)))
                            for t in args.tasks}},
        "tasks": {t: {"anchor": ctx.searches[t].anchor,
                      "best_primary": ctx.searches[t].best_primary,
                      **({"note": ctx.searches[t].note}
                         if ctx.searches[t].note else {}),
                      "history": ctx.searches[t].history}
                  for t in args.tasks},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    n_screen = sum(1 for t in args.tasks for h in ctx.searches[t].history
                   if h["rung"] == "screen")
    n_full = sum(1 for t in args.tasks for h in ctx.searches[t].history
                 if h["rung"] == "full")
    print(f"\n[search] winners (search seed {args.seed}; incumbent wins ties; "
          f"{n_screen} screening + {n_full} full-budget runs):")
    for task in args.tasks:
        s = ctx.searches[task]
        anchor_sc = s.scores.get(cfg_key(s.anchor))
        moved = cfg_key(s.best) != cfg_key(s.anchor)
        label = "moved " if moved else "anchor"
        print(f"  {task:5s} [{label}] {fmt_cfg(s.best)}  "
              f"primary={fmt_score(s.best_primary)} "
              f"(anchor {fmt_score(anchor_sc)})")
    print(f"[search] hyperparameters + history -> {args.output}")


def final_cmd(args, ctx, num_gpus):
    cmd = [
        sys.executable, str(SCRIPTS / "evaluate_glue.py"),
        "--model", args.model,
        "--tokenizer", args.tokenizer,
        "--hparams", args.output,
        "--tasks", *args.tasks,
        "--seeds", *[str(s) for s in args.final_seeds],
        "--patience", str(args.patience),
        "--max_len", str(args.max_len),
        "--dtype", args.dtype,
        "--task_parallel", "--num_gpus", str(num_gpus),
        "--output", args.final_output,
    ]
    if ctx.need_transfer and ctx.mnli_best.is_dir():
        # Hand the searched transfer source to the final run: it was trained on
        # full data with the winning recipe and the final run's own seed, so it
        # is exactly what the final run would rebuild. rte/mrpc/stsb start
        # immediately, and the duplicate MNLI source train is skipped.
        cmd += ["--mnli_source", str(ctx.mnli_best)]
    if not args.mnli_transfer:
        cmd.append("--no_mnli_transfer")
    if not args.compile:
        cmd.append("--no_compile")
    if args.wandb_project:
        cmd += ["--wandb_project", args.wandb_project]
        if args.wandb_entity:
            cmd += ["--wandb_entity", args.wandb_entity]
        if args.wandb_run_name:
            cmd += ["--wandb_run_name", args.wandb_run_name]
    return cmd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="pretrained backbone dir or checkpoint")
    p.add_argument("--tokenizer", default=None,
                   help="default: ModernBERT's for a NexteraBERT backbone, the "
                        "baseline's own for a Hugging Face model")
    p.add_argument("--tasks", nargs="+", default=["all"],
                   help="tasks to search (and finally evaluate); 'all' = the 8 "
                        "standard GLUE tasks")
    p.add_argument("--stages", choices=["auto", "full", "lr"], default="auto",
                   help="which axes to sweep: 'auto' (default) = lr+wd+epochs on "
                        "the small tasks (cola/mrpc/stsb/rte), lr only on the "
                        "large stable ones; 'full' = all axes everywhere; 'lr' = "
                        "lr only everywhere")
    # --- speed / fidelity knobs (successive halving + proxy data) ----------
    p.add_argument("--train_subsample", type=int, default=32768,
                   help="cap each candidate's TRAINING set at this many examples "
                        "(seed-fixed random subset; only affects tasks larger "
                        "than it). The proxy that makes qqp/mnli/qnli/sst2 "
                        "searchable at small-task cost. 0 disables (full data).")
    p.add_argument("--screen_epochs", type=int, default=1,
                   help="successive-halving screening rung: every candidate "
                        "first runs this many epochs (LR schedule still spans "
                        "its full epoch budget), and only the top --promote "
                        "continue at full budget. 0 disables screening.")
    p.add_argument("--promote", type=int, default=2,
                   help="how many screening winners advance to a full-budget "
                        "run per sweep (the incumbent is always scored at full "
                        "budget as well). 1 = most aggressive halving.")
    p.add_argument("--exact", action="store_true",
                   help="paper-faithful search: full data and full-length runs "
                        "for every candidate (sets --train_subsample 0 and "
                        "--screen_epochs 0). Roughly 5-6x slower.")
    # -----------------------------------------------------------------------
    p.add_argument("--seed", type=int, default=19,
                   help="single fine-tuning seed used to score candidates "
                        "(default 19, ModernBERT's default_seed)")
    p.add_argument("--patience", type=int, default=2,
                   help="early stopping for every candidate run, same semantics "
                        "as evaluate_glue.py (0 disables)")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--no_compile", dest="compile", action="store_false")
    p.set_defaults(compile=True)
    p.add_argument("--num_gpus", type=int, default=None,
                   help="GPUs to schedule candidates across (default: all visible)")
    p.add_argument("--work_dir", default="glue_search_work",
                   help="cache dir for rung results + the MNLI transfer source; "
                        "an interrupted search resumes from it")
    p.add_argument("--output", default="glue_search.json",
                   help="where to write the winning per-task hyperparameters "
                        "(consumable by evaluate_glue.py --hparams)")
    p.add_argument("--no_mnli_transfer", dest="mnli_transfer", action="store_false",
                   help="search (and finally evaluate) rte/mrpc/stsb from the cold "
                        "backbone instead of an MNLI-tuned encoder")
    p.set_defaults(mnli_transfer=True)
    p.add_argument("--no_final", dest="final", action="store_false",
                   help="stop after the search; print the final-evaluation "
                        "command instead of running it")
    p.set_defaults(final=True)
    p.add_argument("--final_seeds", nargs="+", type=int,
                   default=[19, 8364, 717, 10536, 90166],
                   help="seeds for the final evaluation (per-task caps apply as "
                        "in evaluate_glue.py)")
    p.add_argument("--final_output", default="glue_results.json")
    p.add_argument("--dry_run", action="store_true",
                   help="print the search plan (stages, anchors, run-count "
                        "bounds) and exit without training")
    p.add_argument("--wandb_project", default=None,
                   help="passed through to the final evaluation")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    args = p.parse_args()
    from nexterabert.hf_baselines import is_hf_model_dir, resolve_hub_model

    args.model = resolve_hub_model(args.model)
    if args.tokenizer is None:
        args.tokenizer = (args.model if is_hf_model_dir(args.model)
                          else "answerdotai/ModernBERT-base")
    return args


def plan_bounds(search):
    """Upper bounds on (screening runs, full-budget runs) for one task."""
    screens = fulls = 0
    for i, stage in enumerate(search.stages):
        n_new = len(stage_candidates(search.task, search.anchor, stage)) - int(i > 0)
        if search.screen_epochs and n_new > search.promote:
            screens += n_new
            # top --promote, +1 for the incumbent in stage 1 (no full score yet)
            fulls += search.promote + int(i == 0)
        else:
            fulls += n_new
    return screens, fulls


def main():
    args = parse_args()
    if args.tasks == ["all"]:
        args.tasks = list(GLUE_AVG_TASKS)
    unknown = [t for t in args.tasks if t not in GLUE_TASKS]
    if unknown:
        sys.exit(f"unknown GLUE task(s): {unknown}")
    if args.exact:
        args.train_subsample = 0
        args.screen_epochs = 0

    table = TASK_DEFAULTS_MODERNBERT
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    num_gpus = args.num_gpus if (args.num_gpus and args.num_gpus > 0) \
        else (torch.cuda.device_count() or 1)

    searches = {t: TaskSearch(t, table[t], stages_for(t, args.stages), args)
                for t in args.tasks}
    init_tasks = MNLI_INIT_TASKS
    dependents = [t for t in args.tasks if t in init_tasks]
    need_transfer = args.mnli_transfer and bool(dependents)
    ctx = Ctx(args, work, searches, dependents, need_transfer)

    proxy = (f"subsample={args.train_subsample or 'off'} "
             f"screen={args.screen_epochs or 'off'}ep promote={args.promote}")
    if args.dry_run:
        n_screen = n_full = 0
        print(f"[search] plan: stages={args.stages} "
              f"seed={args.seed} gpus={num_gpus} {proxy}")
        for t in args.tasks:
            s = searches[t]
            sc, fu = plan_bounds(s)
            n_screen += sc
            n_full += fu
            dep = "  (init: MNLI)" if need_transfer and t in dependents else ""
            print(f"  {t:5s} stages={'+'.join(s.stages):12s} "
                  f"<={sc} screen + <={fu} full runs{dep}  "
                  f"anchor: {fmt_cfg(s.anchor)}")
        extra = ""
        if need_transfer:
            n_full += 1
            extra = " (incl. 1 full-data MNLI transfer source)"
        print(f"[search] <={n_screen} screening ({args.screen_epochs or 0} ep, "
              f"subsampled) + <={n_full} full-budget runs{extra} "
              f"vs {48 * len(args.tasks)} for the full grid")
        print("[search] final evaluation command:")
        print("  " + " ".join(final_cmd(args, ctx, num_gpus)))
        return

    print(f"[search] {len(args.tasks)} task(s), stages={args.stages}, "
          f"seed={args.seed}, {num_gpus} GPU(s), {proxy}, cache={work}")

    # Seed the scheduler. MNLI (transfer source) starts immediately; its
    # dependents wait for the full-data source encoder; everything else runs
    # freely.
    if need_transfer:
        print(f"[search] MNLI-transfer ON: {dependents} will be searched from "
              f"the winning MNLI encoder")
        if "mnli" in searches:
            jobs = open_stage(searches["mnli"], work)
            if searches["mnli"].done:
                on_search_done(searches["mnli"], ctx)   # fully cached resume
            else:
                ctx.queue.extend(jobs)
        else:
            print("[search] mnli not searched -> the transfer source uses the "
                  "anchor recipe")
            ensure_mnli_source(ctx, table["mnli"])
    for t in args.tasks:
        if t == "mnli" and need_transfer:
            continue                       # seeded above
        if need_transfer and t in dependents:
            continue                       # released once the source is ready
        jobs = open_stage(searches[t], work)
        if searches[t].done:
            on_search_done(searches[t], ctx)
        else:
            ctx.queue.extend(jobs)

    run_scheduler(args, ctx, num_gpus)
    write_output(args, ctx)

    cmd = final_cmd(args, ctx, num_gpus)
    if not args.final:
        print("[search] --no_final: run the final evaluation with:")
        print("  " + " ".join(cmd))
        return
    print("[search] launching the final evaluation:")
    print("  " + " ".join(cmd))
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        sys.exit(ret)
    print(f"[search] done: hyperparameters in {args.output}, "
          f"final scores in {args.final_output}")


if __name__ == "__main__":
    main()
