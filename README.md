# Retrieval-Augmented QA on ARC

CSE 151B (Deep Learning) course project. Separates *retrieval quality* from
*reader utilization* in retrieval-augmented QA: end-to-end accuracy alone
can't tell you whether a gain comes from the retriever finding better
evidence or the reader just doing well regardless of what it's given, so
this project measures both independently.

**Dataset:** AI2 Reasoning Challenge (ARC) — ~7,800 grade-school multiple-choice
science questions split into an Easy Set and a harder Challenge Set (the
Challenge Set was specifically filtered to exclude anything a simple
retrieval/co-occurrence baseline could already solve), plus the ~14.6M-sentence
ARC Corpus as the retrieval source.

**Design:** one DeBERTa-v3 multiple-choice reader, trained once and reused
unchanged, scored under four conditions — closed-book, and three retrieval
strategies (BM25, a dense bi-encoder, and a BM25→dense-rerank hybrid) feeding
it retrieved passages. Retrieval quality itself is measured separately via
Recall@10/MRR against a relevance-labeled sample, since ARC doesn't ship gold
relevance labels.

## Repo structure

```
├── data/
│   ├── raw/                # ARC Easy/Challenge + ARC Corpus (downloaded, gitignored)
│   └── relevance_labels/   # sample.json -- labeled sample for Recall@10/MRR
├── src/
│   ├── data/loader.py       # ARC Easy/Challenge/Corpus loading
│   ├── retrievers/
│   │   ├── bm25.py           # sparse term-document matrix BM25
│   │   ├── dense.py          # Sentence-BERT bi-encoder + precomputed embeddings
│   │   └── hybrid.py         # BM25 candidates -> on-the-fly dense rerank
│   ├── reader/
│   │   ├── model.py          # multiple-choice preprocessing + model loading
│   │   └── train.py          # closed-book fine-tuning (HF Trainer)
│   └── eval/
│       ├── qa_metrics.py           # end-to-end accuracy, split by Easy/Challenge
│       ├── retrieval_metrics.py    # Recall@10, MRR, split by Easy/Challenge
│       └── relevance_heuristic.py  # lexical-overlap silver-labeling heuristic
├── scripts/                 # download, benchmark, and eval entry points (see below)
├── configs/                 # per-run hyperparameters (reader_closed_book.yaml)
└── results/                 # metrics JSON + reader checkpoint pointer (see Results)
```

`src/pipeline.py` (question → retriever → reader → answer, wired end-to-end)
is the one piece still an empty stub.

## Setup

```bash
conda create -n arc-qa python=3.11
conda activate arc-qa
pip install -r requirements.txt
```

**MPS note:** the reader (`microsoft/deberta-v3-base`) hangs / produces a
zero grad norm on its backward pass under PyTorch's MPS backend — a real
incompatibility with DeBERTa's disentangled attention, not a config issue.
CPU works for small smoke tests (`src/reader/model.py` also force-loads
fp32 — the checkpoint otherwise defaults to fp16, which underflows to NaN
gradients when trained without loss scaling). Do real training and the
dense embedding build on DataHub's CUDA GPU, not locally — MPS also measured
~7x slower than DataHub's A30 for passage embedding.

## Data

```bash
python scripts/download_data.py
```

Downloads and extracts AI2's official release (ARC-Easy, ARC-Challenge, and
the ARC Corpus) into `data/raw/`. Source:
`https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip`
(the arXiv page for the paper, 1803.05457, is not a dataset host — this S3
link, referenced from AI2's [ARC-Solvers](https://github.com/allenai/ARC-Solvers)
repo, is the actual download).

**License:** AI2's terms are non-commercial, research-only, and explicitly
prohibit redistribution. `data/raw/` is gitignored so it never gets pushed —
re-run the download script on each machine (local, DataHub, etc.) rather
than copying the data over.

Sanity-check: `python scripts/inspect_data.py`. Loading helpers live in
[src/data/loader.py](src/data/loader.py) — `load_all_questions()`,
`load_questions(split, subset)`, `iter_corpus()` (streams the corpus line by
line, ~1.4GB if you `list()` it), and `load_corpus_subset()`.

**Corpus subset:** all three retrievers default to `CORPUS_SUBSET_SIZE`
(~7.3M passages, half the full corpus) rather than the full 14.6M. AI2's
README states the corpus is already randomly shuffled, so a prefix is a
valid random subsample. This halves the dense-embedding build time and is
a single shared constant so BM25/dense/hybrid are always compared over the
*same* corpus — never pass a custom `--max-passages` to one retriever
without matching it across all three, or dense retrieval will misalign
against its precomputed embeddings.

## Running on DataHub (UCSD GPU)

DataHub is behind UCSD SSO, so this part is manual. A few hard-won notes
before the steps:

- **Home directory quota is tiny (~2GB)**, enforced separately from what
  `df -h` reports (that shows the whole NFS export, not your quota). The
  Python env, downloaded data, and dense embeddings (~11GB) all need to live
  in `/tmp` instead — `pip install` with `PIP_CACHE_DIR=/tmp/pip-cache` set,
  and symlink `data/raw` to a `/tmp` directory.
- **`/tmp` is ephemeral and tied to the specific pod.** If DataHub reschedules
  your pod (it happened once during this project, wiping ~25min of an
  in-progress embedding build), `/tmp` is gone — venv, data, and any
  in-progress build included, though your home-directory-backed git repo is
  untouched. `build_passage_embeddings()` checkpoints progress to a sidecar
  file so restarting an *interrupted process* resumes rather than restarting
  from scratch, but this doesn't survive a full pod reschedule (no storage
  tier here is both large enough and durable across that).
- If JupyterLab shows a "Server unavailable, restart?" dialog, click
  **Dismiss**, not Restart — it's often a transient websocket hiccup, not an
  actual crash, and Restart can force a pod recreation that kills whatever's
  running. Check `nvidia-smi` / the job's own terminal output before assuming
  the worst.

Steps:

```bash
# 1. Clone + environment (entirely in /tmp, see notes above)
git clone https://github.com/ikumar55/arc-retrieval-qa.git
cd arc-retrieval-qa
python3 -m venv /tmp/arc-qa-venv
source /tmp/arc-qa-venv/bin/activate
export PIP_CACHE_DIR=/tmp/pip-cache
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2. Data (symlinked into /tmp, not home)
mkdir -p /tmp/arc_data
ln -s /tmp/arc_data data/raw
python scripts/download_data.py

# 3. Reader baseline (~5min on an A30)
python src/reader/train.py --config configs/reader_closed_book.yaml 2>&1 | tee train.log
mkdir -p /tmp/arc_results
mv results/reader_closed_book/final_model results/reader_closed_book/checkpoint-* /tmp/arc_results/ 2>/dev/null

# 4. Dense passage embeddings (~90min on an A30 over the half-corpus -- use tmux)
tmux new -s dense
python scripts/bench_dense.py 2>&1 | tee dense_bench.log
# detach: Ctrl-b then d — reattach later with: tmux attach -t dense

# 5. Relevance-labeled sample (~1min)
python scripts/build_relevance_labels.py 2>&1 | tee labels.log

# 6. Final eval
python scripts/eval_retrieval.py 2>&1 | tee eval_retrieval.log
python scripts/eval_qa_accuracy.py --model-path /tmp/arc_results/final_model 2>&1 | tee eval_qa_accuracy.log
```

Long-running steps (3, 4) print periodic `[progress] ... elapsed ... eta ...`
lines rather than raw tqdm bars, specifically so they're readable via
`tail -f` over a remote/piped session.

## Results

**End-to-end QA accuracy** (DeBERTa-v3-base reader, trained closed-book only
and reused unchanged across all conditions; top-3 retrieved passages
concatenated for the retrieval-augmented conditions):

| Condition   | Easy  | Challenge | Overall |
|-------------|-------|-----------|---------|
| Closed-book | 71.7% | 51.2%     | **64.9%** |
| BM25        | 61.2% | 43.3%     | 55.3%   |
| Dense       | 64.9% | 44.0%     | **58.0%** |
| Hybrid      | 64.7% | 44.0%     | 57.8%   |

**Retrieval quality** (Recall@10 / MRR against a 200-question relevance-labeled
sample — 100 Easy + 100 Challenge, pooled top-20 candidates from all three
retrievers, auto-labeled via lexical-overlap against question+correct-answer):

| Retriever | Recall@10 (E/C/Overall) | MRR (E/C/Overall) |
|-----------|-------------------------|--------------------|
| BM25      | 0.58 / 0.33 / **0.455**  | 0.52 / 0.26 / **0.391** |
| Dense     | 0.47 / 0.23 / 0.350      | 0.31 / 0.17 / 0.242 |
| Hybrid    | 0.56 / 0.27 / 0.415      | 0.40 / 0.20 / 0.302 |

**Two findings worth flagging:**

1. **All three retrieval-augmented conditions underperform closed-book on
   accuracy.** The reader was fine-tuned purely on `[question + choice]`
   pairs and never saw retrieved passages during training, so at inference
   time it's facing an input distribution (and often partially-irrelevant
   context — BM25's own Recall@10 is only ~45%) it never learned to use.
   This is a legitimate finding rather than a bug: it suggests retrieval-aware
   training (not just closed-book) is likely necessary to see accuracy gains
   from retrieval augmentation here.
2. **The accuracy ranking (dense ≳ hybrid > BM25) doesn't match the
   Recall@10/MRR ranking (BM25 > hybrid > dense).** This is exactly the kind
   of retrieval-quality-vs-reader-utilization dissociation the project set
   out to measure. Some of BM25's Recall@10/MRR advantage is likely inflated
   by the relevance labels themselves being built from a *lexical*
   overlap heuristic (see `src/eval/relevance_heuristic.py`), which
   structurally favors a lexical retriever like BM25 over passages that are
   semantically relevant but differently worded — a known limitation of
   automated (non-manually-verified) silver relevance labels.

Reproducing these: results land in `results/qa_accuracy_*.json` and
`results/retrieval_metrics.json`. The fine-tuned reader checkpoint itself is
not committed (large binary, regenerable in ~5min via step 3 above) — only
`results/reader_closed_book/closed_book_results.json` (the accuracy numbers)
is tracked.
