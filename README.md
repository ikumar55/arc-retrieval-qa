# Retrieval-Augmented QA on ARC

CSE 151B project. See [PROJECT_BRIEF.md](PROJECT_BRIEF.md) for full research design.

## Setup

```bash
conda create -n arc-qa python=3.11
conda activate arc-qa
pip install -r requirements.txt
```

Torch will use MPS (Apple Silicon GPU) automatically on the Mac for quick local
iteration; larger reader fine-tuning runs should go on the DataHub GPU instead.

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

## Repo structure

See [PROJECT_BRIEF.md](PROJECT_BRIEF.md#suggested-repo-structure). `src/retrievers/`,
`src/reader/`, `src/eval/`, and `src/pipeline.py` are currently empty stubs —
not implemented yet.
