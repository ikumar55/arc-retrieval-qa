# Retrieval-Augmented QA on ARC

CSE 151B project. See [PROJECT_BRIEF.md](PROJECT_BRIEF.md) for full research design.

## Setup

```bash
conda create -n arc-qa python=3.11
conda activate arc-qa
pip install -r requirements.txt
```

**Note on MPS:** the reader (`microsoft/deberta-v3-base`) hangs / produces a
zero grad norm on its backward pass under PyTorch's MPS backend — a real
incompatibility with DeBERTa's disentangled attention, not a config issue.
CPU works fine for small smoke tests (`src/reader/model.py` forces fp32
loading — the checkpoint otherwise loads in fp16 by default, which underflows
to NaN gradients). Do real training runs on DataHub's CUDA GPU, not locally.

## Data

```bash
python scripts/download_data.py
```

Downloads and extracts AI2's official release (ARC-Easy, ARC-Challenge, and the
ARC Corpus) into `data/raw/`. Source: `https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip`
(the arXiv page for the paper, 1803.05457, is not a dataset host — this S3 link,
referenced from AI2's [ARC-Solvers](https://github.com/allenai/ARC-Solvers) repo,
is the actual download).

**License note:** AI2's terms for this dataset are non-commercial, research-only,
and explicitly prohibit redistribution. `data/raw/` is gitignored so it never
gets pushed — when you set up GitHub and/or DataHub, re-run the download script
there rather than copying the data over.

Sanity-check the data once downloaded:

```bash
python scripts/inspect_data.py
```

Loading helpers live in [src/data/loader.py](src/data/loader.py):
`load_all_questions()` returns a dict of `easy_train`/`easy_dev`/`easy_test`/
`challenge_train`/`challenge_dev`/`challenge_test` lists of `ArcQuestion`;
`iter_corpus()` streams the 14.6M-sentence ARC Corpus line by line (don't
`list()` it — it's ~1.4GB in memory).

## Running on DataHub (UCSD GPU)

DataHub is behind UCSD SSO, so this part is manual:

1. Go to https://datahub.ucsd.edu, launch a server with a GPU option for this course.
2. Open a terminal in the JupyterHub session.
3. Clone and set up:
   ```bash
   git clone https://github.com/ikumar55/arc-retrieval-qa.git
   cd arc-retrieval-qa
   conda create -n arc-qa python=3.11 -y && conda activate arc-qa
   pip install -r requirements.txt
   nvidia-smi   # confirm a GPU + driver are visible
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   If `torch.cuda.is_available()` is False, the plain `pip install torch` pulled a
   CPU-only wheel — reinstall using the command from
   https://pytorch.org/get-started/locally/ for the CUDA version `nvidia-smi` reports.
4. Get the data (not committed, per the license note above):
   ```bash
   python scripts/download_data.py
   ```
5. Run training in `tmux` (or `nohup ... &`) so it survives a dropped connection,
   and log to a file so you can `tail -f` it:
   ```bash
   tmux new -s reader
   python src/reader/train.py --config configs/reader_closed_book.yaml 2>&1 | tee train.log
   # detach: Ctrl-b then d — reattach later with: tmux attach -t reader
   ```
   `train.py` prints periodic `[progress] step X/Y epoch ... elapsed ... eta ...`
   lines (not raw tqdm bars) specifically so this is readable via `tail -f train.log`
   over a remote session.
6. Results land in `results/reader_closed_book/`: `closed_book_results.json`
   (accuracy by Easy/Challenge/overall) and `final_model/` (the fine-tuned
   checkpoint, reused unchanged by the retrieval-augmented conditions later).

## Repo structure

See [PROJECT_BRIEF.md](PROJECT_BRIEF.md#suggested-repo-structure). `src/retrievers/`,
`src/reader/`, `src/eval/`, and `src/pipeline.py` are currently empty stubs —
not implemented yet.
