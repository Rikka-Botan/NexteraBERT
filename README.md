![NexteraBERT Logo](https://huggingface.co/RikkaBotan/NexteraBERT-Mezzoforte-220M-en/resolve/main/assets/NexteraBERT_logo.png)

# NexteraBERT

NexteraBERT is a fast, long-context bidirectional encoder. This repository contains
the model code and the code used to pretrain and evaluate the released NexteraBERT
models.

- **Model:** [RikkaBotan/NexteraBERT-Mezzoforte-220M-en](https://huggingface.co/RikkaBotan/NexteraBERT-Mezzoforte-220M-en)
  (212.6M parameters, 130B pretraining tokens)
- **Paper:** [NexteraBERT.pdf](https://huggingface.co/RikkaBotan/NexteraBERT-Mezzoforte-220M-en/blob/main/NexteraBERT.pdf)
- **Pretraining data (tokenized shards):** [RikkaBotan/NexteraBERT-data-mix](https://huggingface.co/datasets/RikkaBotan/NexteraBERT-data-mix)

## Released models

| Model | Pretraining tokens | GLUE (mean of 8 tasks) | MTEB v2 (mean over task types) |
|---|---:|---:|---:|
| [NexteraBERT-Mezzoforte-220M-en](https://huggingface.co/RikkaBotan/NexteraBERT-Mezzoforte-220M-en) | 130B | 87.90 | 54.70 |
| [NexteraBERT-Mezzoforte-220M-13B-en](https://huggingface.co/RikkaBotan/NexteraBERT-Mezzoforte-220M-13B-en) | 13B | 85.45 | 51.92 |
| [NexteraBERT-Mezzoforte-220M-1.3B-en](https://huggingface.co/RikkaBotan/NexteraBERT-Mezzoforte-220M-1.3B-en) | 1.3B | 77.54 | 47.82 |

The three models share the architecture and the recipe. Each budget is a separate run,
not an intermediate checkpoint of the 130B run.

![Main results of NexteraBERT](https://huggingface.co/RikkaBotan/NexteraBERT-Mezzoforte-220M-en/resolve/main/assets/summary.png)

The 130B model against ModernBERT-base. Both models are fine-tuned and evaluated with
the same protocol, using the scripts in this repository.

| Benchmark | Metric | NexteraBERT | ModernBERT-base |
|---|---|---:|---:|
| GLUE | mean of 8 tasks | 87.90 | **87.97** |
| MTEB v2 (English, 41 tasks) | mean over task types | **54.70** | 53.63 |
| BEIR (15 datasets) | nDCG@10 | **43.89** | 43.09 |
| NanoBEIR (13 subsets) | nDCG@10 | **55.92** | 53.76 |
| CodeSearchNet (CoIR) | nDCG@10 | **57.22** | 56.65 |
| StackOverflowQA | nDCG@10 | 65.71 | **69.20** |
| MultiLongDocRetrieval (MLDR) | nDCG@10 | **36.25** | 31.19 |
| Masked-token loss, 8,192 → 65,536 tokens | increase (lower is better) | **+3.8%** | +83.0% |
| Throughput at 65,536 tokens (H200 NVL) | relative to ModernBERT-base | **5.22×** | 1.00× |

## Architecture

The encoder has 18 blocks. Each block has a token mixer and a squared-ReLU MLP, both
with pre-normalization and a residual connection (`x = x + f(norm(x))`).

| Token mixer | `block_types` key | Blocks | Description |
|---|---|---:|---|
| `SnowLily` | `lily` | 8 | Input-dependent gating liquid mixer derived from liquid time-constant networks. Cost is linear in input length. |
| `NexteraSlidingWindowAttention` | `swa` | 5 | Sliding-window attention with a 256-token window. Uses its own RoPE table (base 10,000) and no SSSMax. |
| `NexteraSelfAttention` | `attn` | 3 | Full attention with SSSMax (Stable Scalable-Softmax): logits are scaled with the number of keys, so attention stays selective on long inputs. |
| `NexteraHRA` | `hra` | 2 | Attention over mean-pooled bands of 4 tokens, a cheaper global path (also with SSSMax). |

Block order: `lily lily swa lily swa lily attn lily swa lily swa attn hra lily swa attn hra lily`.

| | |
|---|---|
| Blocks / width / MLP width | 18 / 1,024 / 2,304 |
| Query heads / key-value heads | 16 / 8 (grouped-query attention, head width 64) |
| Window attention | window 256 (\|i − j\| ≤ 128) |
| HRA | band size 4 |
| SSSMax (initial values) | s = 0.43, b = 0.1, ε = 0.1 |
| RoPE base | 100,000 (full attention, HRA); 10,000 (window attention) |
| Vocabulary | 50,368 (ModernBERT tokenizer) |
| Parameters | 212.6M (161.1M non-embedding) |

The released model is the `mezzoforte` preset of `NexteraBERTConfig`
(`src/nexterabert/configuration_nexterabert.py`). The other presets (`pianissimo`,
`piano`, `mezzopiano`, `forte`, `fortissimo`) share the layout at other sizes; only
`mezzoforte` has been released.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra train --extra eval
```

`uv run <cmd>` then runs a command inside the environment. If the default torch wheel
of your platform has no CUDA support (for example on Windows), install the CUDA build
that matches your driver from [pytorch.org](https://pytorch.org/get-started/locally/).

| Extra | Contents |
|---|---|
| `train` | Weights & Biases logging |
| `eval` | MTEB (with sentence-transformers), accelerate, scikit-learn, SciPy |
| `dev` | ruff, and the optional fast tokenizer backend used by `scripts/prepare_data.py` |

The speed and length benchmarks (`model_speedbench.py`, `model_pplbench.py`) also need
matplotlib (`uv pip install matplotlib`).

## Quick start

The Hub repositories bundle the model code, so the models load with `transformers`
and `trust_remote_code=True`.

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

repo = "RikkaBotan/NexteraBERT-Mezzoforte-220M-en"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForMaskedLM.from_pretrained(repo, trust_remote_code=True)

fill = pipeline("fill-mask", model=model, tokenizer=tokenizer)
print(fill("The capital of France is [MASK]."))
```

Hidden states and a mean-pooled embedding:

```python
import torch
from transformers import AutoModel, AutoTokenizer

repo = "RikkaBotan/NexteraBERT-Mezzoforte-220M-en"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval()

batch = tokenizer(["hello world", "a second sentence"], padding=True, return_tensors="pt")
with torch.no_grad():
    hidden = model(**batch).last_hidden_state          # (batch, length, 1024)

mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
embedding = (hidden * mask).sum(1) / mask.sum(1)       # mean pooling
```

This is a pretrained encoder, not a sentence-embedding model. Fine-tune it (for example
with the contrastive stages below) before using it for retrieval or similarity.
`AutoModelForSequenceClassification` is also available; its classification head is
newly initialized and must be trained.

### Unpadded inference

A padded batch of short texts spends most of its compute on padding. With unpadding
on, the encoder packs the real tokens of every row into one stream, runs the stack on
it and scatters the result back to `(batch, length)`:

```python
model = AutoModel.from_pretrained(repo, trust_remote_code=True, unpadding=True)
# or, with the local package: model.encoder.set_unpadding(True)
```

Each row is then encoded exactly as if it were alone in the batch: attention uses a
document mask, RoPE positions, the SSSMax length and HRA's bands restart at every row,
and the convolutions stop at row edges. This also removes a padding leak in HRA, whose
convolutions otherwise read the pad slots after a row's last token, so a head fine-tuned
on padded batches sees slightly different features; that is why unpadding is opt-in.
Calls without an `attention_mask` keep the padded path.

## Pretraining

### Recipe

Masked language modeling with the 80/10/10 replacement rule, in two phases:

| | Phase 1 | Phase 2 |
|---|---|---|
| Config | `configs/pretrain_bert_phase1.yaml` | `configs/pretrain_bert_phase2.yaml` |
| Inputs | up to 1,024 tokens, one document per example | 8,192-token packed blocks; one-document 1,024-token batches on 40% of steps |
| Masking rate | 40% → 20% (linear) | 20% |
| Tokens (130B model) | 120B | 10B |
| Peak learning rate | 8e-4 | 8e-5 |
| Warm-up | 10% of steps | 400 steps |
| Initialization | random | phase-1 weights, new optimizer state |

Both phases use AdamW with β = (0.9, 0.95), weight decay 0.1, gradient clipping 1.0,
cosine decay to 10% of the peak learning rate, and bf16. The budget is set with
`train_tokens`; `max_steps` is derived from the shard at startup. The 13B model uses
12B + 1B tokens and the 1.3B model 1.2B + 0.1B tokens (`--train_tokens` on the command
line overrides the config).

**Effective batch.** `batch_size` and `grad_accum` in the configs are per GPU. The
released models were trained on one GPU: 1,024 sequences per optimizer step in phase 1
(128 × 8) and 128 blocks of 8,192 tokens in phase 2 (8 × 16). With N GPUs, divide
`grad_accum` by N to keep this recipe; otherwise the effective batch grows N-fold.

### Data

The mix is 90.25% FineWeb-Edu (`sample-350BT`), 4.75% DCLM and 5% StarCoderData
(Python, Java, C++, JavaScript), tokenized with the ModernBERT tokenizer:

| Shard | Content |
|---|---|
| `data/mix_1024.bin` (+ `.lens`, `.meta`) | phase 1: 120B tokens, 179M documents, one document per example |
| `data/mix_8192.bin` (+ `.meta`) | phase 2: 10B tokens in 8,192-token blocks; documents aligned to 8 tokens so HRA bands never straddle two documents |
| `data/mix_eval_1024.bin`, `data/mix_eval_8192.bin` | disjoint held-out slices with the same mix |

`scripts/run_pipeline_bert.sh` downloads these shards from
[RikkaBotan/NexteraBERT-data-mix](https://huggingface.co/datasets/RikkaBotan/NexteraBERT-data-mix).
To build them yourself:

```bash
export HF_TOKEN=...   # StarCoderData is gated: accept its terms on the Hub first
bash scripts/build_data.sh                                          # shards stay in data/
DATA_REPO_ID=<user>/NexteraBERT-data-mix bash scripts/build_data.sh  # ... and upload them
```

The build is resumable: an interrupted run continues from its checkpoint when the same
command is run again.

### One-command pipeline

```bash
NPROC=1 bash scripts/run_pipeline_bert.sh
```

The pipeline downloads the shards, trains both phases, exports the backbone to
`checkpoints/mezzoforte_bert/phase2/backbone`, runs GLUE and the SimCSE + MTEB(eng, v2)
stage, and, when `HF_REPO_ID` is set, uploads the backbone and verifies that it loads
with `trust_remote_code`. Its settings are read from the environment (`NPROC`,
`HF_REPO_ID`, `DATA_REPO_ID`, `GLUE_TASKS`, `GLUE_SEEDS`, `RUN_MTEB`, `MTEB_*`, ...;
see the top of the script). Metrics go to Weights & Biases: set `WANDB_API_KEY` or run
`wandb login`, or set `WANDB_MODE=offline` to keep the logs local.

`NPROC` sets the number of GPUs for both pretraining and GLUE. With more than one GPU,
lower `grad_accum` in the two configs as described above to keep the released recipe.

### Step by step

```bash
# phase 1 (1,024 tokens) -> checkpoints/mezzoforte_bert/phase1/final.pt
uv run python scripts/pretrain.py --config configs/pretrain_bert_phase1.yaml

# phase 2 (8,192 tokens), warm-started from the phase-1 final.pt
uv run python scripts/pretrain.py --config configs/pretrain_bert_phase2.yaml

# the same on 8 GPUs of one node, with the released effective batch
NPROC=8 uv run bash scripts/train_ddp.sh --grad_accum 1
CONFIG=configs/pretrain_bert_phase2.yaml NPROC=8 uv run bash scripts/train_ddp.sh --grad_accum 2
```

Each phase writes `step_N.pt` / `final.pt` to its `output_dir` and exports the backbone
(`config.json`, `model.safetensors`, `pytorch_model.bin`, `mlm_head.safetensors` and the
tokenizer) to `<output_dir>/backbone`.

To resume an interrupted run, point `--resume` at its latest checkpoint. For phase 2,
also pass `--no-resume_weights_only`, so that the optimizer, the learning-rate schedule
and the step are restored rather than only the weights:

```bash
uv run python scripts/pretrain.py --config configs/pretrain_bert_phase2.yaml \
    --resume checkpoints/mezzoforte_bert/phase2/step_6000.pt --no-resume_weights_only
```

Keep the GPU count, batch size, gradient accumulation and data identical to the
interrupted run; changing them changes the token budget and the schedule.

### Export and upload

```bash
# training checkpoint -> backbone directory (encoder + MLM head + tokenizer)
uv run python scripts/save_model.py --checkpoint checkpoints/mezzoforte_bert/phase2/final.pt \
    --output checkpoints/mezzoforte_bert/phase2/backbone --tokenizer answerdotai/ModernBERT-base

# push to the Hub (weights, tokenizer, model code with auto_map, generated model card)
uv run python scripts/upload_to_hub.py --model checkpoints/mezzoforte_bert/phase2/backbone \
    --repo_id <user>/NexteraBERT-Mezzoforte-220M-en --size mezzoforte --private

# check that the uploaded repo loads with trust_remote_code alone
uv run python scripts/verify_hub_model.py --repo_id <user>/NexteraBERT-Mezzoforte-220M-en \
    --expect mlm --local checkpoints/mezzoforte_bert/phase2/backbone
```

If the verification reports a missing `mlm_head.safetensors`, `scripts/upload_mlm_head.py`
uploads the masked-LM head on its own.

## Evaluation

Every evaluation script reads `MODEL`, a Hub id or an exported backbone directory
(default: `RikkaBotan/NexteraBERT-Mezzoforte-220M-en`), and writes its results to
`eval_results/<model name>/`. A stage whose results file already exists is skipped, so
an interrupted run resumes; `FORCE=1` reruns it. Shared settings (`NPROC`, `OUT_DIR`,
`TOKENIZER`, retrieval lengths and precision, ...) are documented in
`scripts/eval_common.sh`.

| Result | Protocol | Command |
|---|---|---|
| GLUE | ModernBERT-base fine-tuning recipe (per-task table in `evaluate_glue.py`); layer-wise LR decay 0.9; MRPC, STS-B, RTE and QNLI start from the MNLI-tuned encoder; 5 seeds (MRPC, STS-B, RTE), 4 (CoLA), 3 (SST-2), 1 (QQP, MNLI, QNLI) | `bash scripts/eval_nlu.sh` |
| MTEB v2 (41 tasks) | supervised SimCSE on 312,663 NLI triplets with attentive pooling (maximum length 64), then MTEB(eng, v2) at length 512 | `bash scripts/eval_mteb.sh` |
| BEIR (15 datasets), MLDR | MS MARCO stage (1.25M triplets, mean pooling; a sentence-transformers port of ModernBERT's `train_st.py`), BEIR at length 512, MLDR at 8,192 | `bash scripts/eval_dpr.sh` |
| CodeSearchNet, StackOverflowQA | the same MS MARCO checkpoint at length 8,192 | `bash scripts/eval_code.sh` |
| NanoBEIR (13 subsets) | the MS MARCO stage on 250k triplets, length 512 | `bash scripts/eval_nanobeir.sh` |
| Throughput | forward pass, 65,536 tokens per batch, bf16, `torch.compile`, 5 rounds of 5 timed passes | `uv run python src/nexterabert/model_speedbench.py` |
| Masked-token loss vs. length | 15% masking, 25 draws of up to 32 windows per length | `uv run python src/nexterabert/model_pplbench.py` |

**Baselines under the same protocol.** Every stage also accepts a Hugging Face encoder
as `MODEL`, which goes through the same fine-tuning, head and scorer
(`src/nexterabert/hf_baselines.py`, including the loading repairs NeoBERT and
LFM2.5-Encoder need). The `*_baselines.sh` scripts run NexteraBERT, ModernBERT-base,
NeoBERT and LFM2.5-Encoder-230M one model per GPU and write a comparison table:

```bash
bash scripts/eval_mteb_baselines.sh
bash scripts/eval_nanobeir_baselines.sh
MODEL=answerdotai/ModernBERT-base bash scripts/eval_nlu.sh
MODEL=answerdotai/ModernBERT-base bash scripts/eval_dpr.sh
```

The Python entry points can also be called directly, for example:

```bash
uv run python scripts/evaluate_glue.py --model checkpoints/mezzoforte_bert/phase2/backbone \
    --tasks all --output glue_results.json
uv run python scripts/finetune_contrastive.py --protocol mteb-nli \
    --model checkpoints/mezzoforte_bert/phase2/backbone --output_dir checkpoints/mezzoforte_bert/simcse
uv run python scripts/evaluate_mteb.py --model checkpoints/mezzoforte_bert/simcse \
    --benchmark "MTEB(eng, v2)" --output mteb_results.json
```

## Repository layout

```
src/nexterabert/
  configuration_nexterabert.py   NexteraBERTConfig and the size presets
  modeling_nexterabert.py        token mixers, encoder, task heads and the pretraining trainers
  modeling_nexterabert_hf.py     transformers classes used by trust_remote_code
  tokenization_nexterabert.py    inference tokenizer with the training-time dynamic padding
  data.py, streaming.py          pretokenized datasets, collators, resumable streaming
  optim.py, training_utils.py    optimizer, DDP setup, LR schedules, checkpointing
  loading.py, export.py          rebuild heads from a backbone; save / push to the Hub
  simcse.py, mteb_encoder.py     pooling heads and InfoNCE loss; MTEB encoder wrapper
  hf_baselines.py                ModernBERT / NeoBERT / LFM2.5-Encoder through the same stages
  model_speedbench.py            throughput vs. input length
  model_pplbench.py              masked-token loss vs. input length
configs/
  pretrain_bert_phase1.yaml      phase 1: 1,024 tokens, 120B tokens
  pretrain_bert_phase2.yaml      phase 2: 8,192 tokens, 10B tokens, 40% short replay
scripts/
  build_data.sh, prepare_data.py            tokenize the corpus into shards
  pretrain.py, train_ddp.sh                 pretraining (single GPU or DDP)
  run_pipeline_bert.sh                      data -> phase 1 -> phase 2 -> GLUE -> MTEB -> Hub
  save_model.py                             training checkpoint -> backbone directory
  upload_to_hub.py, upload_mlm_head.py      push a backbone (or only its MLM head) to the Hub
  verify_hub_model.py                       check that a Hub repo loads with trust_remote_code
  eval_common.sh                            shared settings of the eval_*.sh scripts
  eval_nlu.sh, evaluate_glue.py                            GLUE
  eval_mteb.sh, finetune_contrastive.py, evaluate_mteb.py  SimCSE stage and MTEB
  eval_dpr.sh, train_st_dpr.py, evaluate_retrieval.py      MS MARCO stage, BEIR, MLDR
  eval_code.sh                              CodeSearchNet, StackOverflowQA
  eval_nanobeir.sh                          NanoBEIR
  eval_*_baselines.sh                       the same stages for the baseline encoders
  merge_retrieval_results.py                merge per-task retrieval results
  summarize_*.py                            result tables
```

## License

The code is released under the [MIT License](LICENSE). The pretraining data keeps the
licenses and terms of its sources (FineWeb-Edu, DCLM, StarCoderData); see the
[dataset card](https://huggingface.co/datasets/RikkaBotan/NexteraBERT-data-mix).
