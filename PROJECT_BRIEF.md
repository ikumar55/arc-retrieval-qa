# Project Brief: Retrieval-Augmented QA on ARC

## Course context
CSE 151B (Deep Learning), quarter-long project. Proposal submitted Week 2, checkpoint Week 4, final code + paper + 3-minute lightning talk due Week 5.

## Dataset
AI2 Reasoning Challenge (ARC) — arXiv 1803.05457. ~7,800 grade-school multiple-choice science questions (4 answer choices each), split into an Easy Set and a Challenge Set. The Challenge Set was constructed by filtering out any question a simple retrieval/co-occurrence baseline could already solve, so it specifically requires reasoning plus the correct supporting fact.

Retrieval corpus: the ARC Corpus (large unstructured science text released alongside the dataset).

## Research question
Most retrieval-augmented QA work reports only end-to-end accuracy, which can't distinguish two different explanations for an accuracy gain: (1) the retriever found better evidence, vs. (2) the reader simply performs well regardless of retrieval quality. This project separates retrieval quality from reader utilization by measuring both independently.

## System design

**Reader (fixed across all experiments):**
- DeBERTa-style encoder, fine-tuned for multiple-choice scoring.
- For each of the 4 answer choices: concatenate [question + retrieved passage(s) + that choice] → single compatibility score.
- Softmax over the 4 scores selects the predicted answer.
- Trained once; reused unchanged across closed-book and all retrieval-augmented conditions, so the reader is not a confound between experiments.

**Retrievers (the independent variable — 3 strategies, all feeding the same reader):**
1. **BM25** — classical sparse/lexical scoring (term frequency / inverse document frequency). No learned embeddings.
2. **Dense bi-encoder** — question and passages embedded independently into the same vector space via a Sentence-BERT-style model; ranked by cosine similarity. Passage embeddings precomputed once for speed.
3. **Hybrid** — BM25 produces a high-recall candidate pool, dense model reranks it.

## Evaluation (this is the core contribution — implement all of it, not just accuracy)

- **Recall@10** — does the true relevant passage appear anywhere in the top-10 retrieved results? Requires a labeled relevance sample (ARC does not ship gold relevance labels — need a labeling strategy, e.g. semi-automated matching or a manual pass on a subset). Report overall and split by Easy vs. Challenge.
- **MRR (Mean Reciprocal Rank)** — 1 / (rank of first relevant passage), averaged across questions. Rewards ranking the right passage early. Report overall and split by Easy vs. Challenge.
- **End-to-end QA accuracy** — scored separately on Easy Set and Challenge Set, for 4 configurations: closed-book, BM25, dense, hybrid.

A hybrid win on QA accuracy alone is necessary but not sufficient to prove better retrieval — it must be corroborated by Recall@10/MRR, since the reader could simply be using hybrid's passages more effectively regardless of relevance.

Splitting Recall@10/MRR by Easy vs. Challenge (not just pooled) is what actually adjudicates H1 vs. H2: if dense already matches hybrid's Recall@10/MRR on the Challenge subset specifically, that's evidence for H2; if hybrid's retrieval-quality edge over dense is concentrated in Challenge, that's evidence for H1. Pooled retrieval metrics alone can't distinguish these.

## Hypotheses (structured abstract, already submitted)
- H1: hybrid retrieval outperforms both single-method baselines, since BM25 and dense fail on different question subtypes (lexical vs. semantic gaps).
- H2 (competing): dense retrieval alone matches hybrid, if ARC Challenge questions mostly demand semantic rather than lexical matching.

## Suggested repo structure
```
arc-retrieval-qa/
├── data/
│   ├── raw/                # ARC Easy/Challenge + ARC Corpus (downloaded, not committed)
│   └── relevance_labels/   # labeled sample for Recall@10 / MRR
├── src/
│   ├── retrievers/
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   └── hybrid.py
│   ├── reader/
│   │   ├── model.py        # DeBERTa multiple-choice scoring head
│   │   └── train.py
│   ├── eval/
│   │   ├── retrieval_metrics.py   # Recall@10, MRR
│   │   └── qa_metrics.py          # accuracy, Easy vs Challenge split
│   └── pipeline.py         # end-to-end: question -> retriever -> reader -> answer
├── notebooks/               # exploration only, not the source of truth
├── configs/                 # one config per experimental condition
├── results/
└── paper/                   # final write-up
```

## Immediate next steps (Week 4 checkpoint target)
1. Get ARC Easy/Challenge + ARC Corpus loaded and preprocessed.
2. Fine-tune the closed-book DeBERTa baseline reader; establish baseline accuracy.
3. Implement BM25 retrieval, verify it returns sane top-k passages.
4. Implement dense bi-encoder retrieval.
5. Implement hybrid (BM25 candidates → dense rerank).
6. Build a relevance-labeled sample sufficient for Recall@10/MRR.
7. Run retrieval-level and end-to-end eval on the held-out sample.
