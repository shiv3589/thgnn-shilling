from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizerFast

logger = logging.getLogger(__name__)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TextEncoder:
    """CLS-pooled encoder backed by frozen roberta-base.

    Model and tokenizer are loaded lazily on the first ``encode`` call.
    Embeddings are optionally cached to ``.npy`` files keyed by dataset name
    so that repeated runs skip expensive forward passes.

    Parameters
    ----------
    cache_dir : str or None
        Root cache directory.  Embeddings are stored at
        ``<cache_dir>/<dataset_name>/embeddings/<suffix>.npy``.
        Pass ``None`` to disable caching.
    batch_size : int
        Texts per forward pass.  Reduce if you hit GPU OOM.
    device : str or None
        Torch device string.  Defaults to CUDA > MPS > CPU.
    """

    MODEL_NAME: str = "roberta-base"
    EMBED_DIM: int = 768
    MAX_LENGTH: int = 128
    _SUFFIX: str = "embeddings"  # subclasses set "reviews" or "items"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self.device = device or _auto_device()
        self._tokenizer: Optional[RobertaTokenizerFast] = None
        self._model: Optional[RobertaModel] = None

    # ------------------------------------------------------------------
    # Lazy-loaded model / tokenizer
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> RobertaTokenizerFast:
        if self._tokenizer is None:
            logger.info("Loading tokenizer: %s", self.MODEL_NAME)
            self._tokenizer = RobertaTokenizerFast.from_pretrained(self.MODEL_NAME)
        return self._tokenizer

    @property
    def model(self) -> RobertaModel:
        if self._model is None:
            logger.info("Loading model %s → %s", self.MODEL_NAME, self.device)
            m = RobertaModel.from_pretrained(self.MODEL_NAME)
            m.eval()
            m.requires_grad_(False)  # freeze all weights
            self._model = m.to(self.device)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: List[str],
        dataset_name: Optional[str] = None,
        *,
        show_progress: bool = True,
        force: bool = False,
    ) -> np.ndarray:
        """Encode *texts* with CLS-token pooling, returning shape ``(N, 768)``.

        Parameters
        ----------
        texts : List[str]
            Input strings.  Empty strings are replaced with a single ``[PAD]``
            token so RoBERTa still produces a valid CLS embedding.
        dataset_name : str or None
            When provided alongside a ``cache_dir``, results are read from / written
            to ``<cache_dir>/<dataset_name>/embeddings/<suffix>.npy``.
        show_progress : bool
            Display a tqdm bar over batches.
        force : bool
            Re-encode and overwrite even when a valid cache file is present.
        """
        if len(texts) == 0:
            return np.empty((0, self.EMBED_DIM), dtype=np.float32)

        cache_path = self._cache_path(dataset_name)

        if cache_path is not None and cache_path.exists() and not force:
            cached = np.load(cache_path)
            if cached.shape[0] == len(texts):
                logger.info("Cache hit (%d × %d): %s", *cached.shape, cache_path)
                return cached
            logger.warning(
                "Cache shape mismatch: expected %d rows, got %d — re-encoding.",
                len(texts), cached.shape[0],
            )

        embeddings = self._encode_all(texts, show_progress=show_progress)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
            logger.info("Saved %s  shape=%s", cache_path, embeddings.shape)

        return embeddings

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cache_path(self, dataset_name: Optional[str]) -> Optional[Path]:
        if self.cache_dir is None or dataset_name is None:
            return None
        return self.cache_dir / dataset_name / "embeddings" / f"{self._SUFFIX}.npy"

    def _encode_all(self, texts: List[str], show_progress: bool) -> np.ndarray:
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        starts = range(0, len(texts), self.batch_size)
        if show_progress:
            starts = tqdm(
                starts,
                total=n_batches,
                desc=self._SUFFIX,
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
        chunks = [self._encode_batch(texts[s: s + self.batch_size]) for s in starts]
        return np.concatenate(chunks, axis=0)  # (N, 768) float32

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        # RoBERTa needs at least one real token; replace blank strings
        texts = [t.strip() or "[PAD]" for t in texts]

        enc = self.tokenizer(
            texts,
            max_length=self.MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model(**enc)

        # RoBERTa: CLS token is at sequence position 0
        cls: np.ndarray = out.last_hidden_state[:, 0, :].cpu().float().numpy()
        return cls  # (B, 768)


# ---------------------------------------------------------------------------
# Named subclasses — separate cache files, same implementation
# ---------------------------------------------------------------------------

class ReviewEncoder(TextEncoder):
    """Encodes review texts.  Cached as ``<dataset>/embeddings/reviews.npy``."""

    _SUFFIX = "reviews"


class ItemEncoder(TextEncoder):
    """Encodes item description / category strings.  Cached as ``items.npy``."""

    _SUFFIX = "items"
