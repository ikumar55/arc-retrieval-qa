"""Dense bi-encoder retrieval: embed questions and passages into a shared space
via a Sentence-BERT-style model, rank by cosine similarity.

Passage embeddings are precomputed once (build_passage_embeddings) and written
to a disk-backed fp16 memmap, since holding the full ARC Corpus's ~14.6M
embeddings in RAM as float32 would need ~22GB. DenseRetriever.load then puts
the flat array on the GPU once (fp16, ~11GB for the full corpus, fits a 24GB
card) for fast repeated top-k search via matrix multiplication.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_passage_embeddings(
    passages: list[str],
    output_path: str,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    device: str | None = None,
    log_every_seconds: float = 120,
    resume: bool = True,
) -> None:
    """Assumes `passages` is the same list, in the same order, across resumed
    runs (true for a fixed corpus file read in order) -- resuming just skips
    re-embedding whatever prefix a progress sidecar file says is already done."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)
    dim = model.get_embedding_dimension()

    n = len(passages)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(".progress.txt")

    already_done = 0
    if resume and output_path.exists() and progress_path.exists():
        try:
            already_done = int(progress_path.read_text().strip())
        except ValueError:
            already_done = 0
        if already_done > 0:
            print(f"[progress] resuming from {already_done:,}/{n:,} passages already embedded", flush=True)

    mode = "r+" if already_done > 0 else "w+"
    mm = np.memmap(output_path, dtype=np.float16, mode=mode, shape=(n, dim))

    start_time = time.time()
    last_log = start_time
    for start in range(already_done, n, batch_size):
        batch = passages[start : start + batch_size]
        emb = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        mm[start : start + len(batch)] = emb.astype(np.float16)

        now = time.time()
        if now - last_log >= log_every_seconds:
            done = start + len(batch)
            mm.flush()
            progress_path.write_text(str(done))
            elapsed = now - start_time
            rate = (done - already_done) / elapsed
            eta_min = (n - done) / rate / 60
            print(
                f"[progress] {done:,}/{n:,} passages ({100 * done / n:.1f}%) "
                f"rate {rate:.0f}/s elapsed {elapsed / 60:.1f}m eta {eta_min:.1f}m",
                flush=True,
            )
            last_log = now

    mm.flush()
    del mm
    progress_path.write_text(str(n))
    np.save(output_path.with_suffix(".shape.npy"), np.array([n, dim]))
    total_min = (time.time() - start_time) / 60
    print(f"[progress] done: {n:,} passages embedded ({n - already_done:,} this run) in {total_min:.1f}m", flush=True)


@dataclass
class DenseRetriever:
    passages: list[str]
    embeddings: torch.Tensor  # (n_passages, dim), fp16, L2-normalized, on `device`
    model: SentenceTransformer
    device: str

    @classmethod
    def load(
        cls,
        passages: list[str],
        embeddings_path: str,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> "DenseRetriever":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        embeddings_path = Path(embeddings_path)
        n, dim = np.load(embeddings_path.with_suffix(".shape.npy"))
        mm = np.memmap(embeddings_path, dtype=np.float16, mode="r", shape=(int(n), int(dim)))
        embeddings = torch.from_numpy(np.asarray(mm)).to(device)
        model = SentenceTransformer(model_name, device=device)
        return cls(passages=passages, embeddings=embeddings, model=model, device=device)

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        q = torch.from_numpy(q_emb).to(self.device, dtype=self.embeddings.dtype)
        scores = self.embeddings @ q
        k = min(top_k, scores.shape[0])
        top_scores, top_idx = torch.topk(scores, k)
        return [(self.passages[i], float(s)) for i, s in zip(top_idx.tolist(), top_scores.tolist())]
