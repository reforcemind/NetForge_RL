import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from netforge_rl.siem.event_templates import (
    evid_4624,
    evid_4625,
    evid_4648,
    evid_4688,
    evid_4768,
    evid_4776,
    sysmon_1,
    sysmon_3,
    sysmon_10,
    sysmon_22,
)

import importlib.util

# Probe availability without importing: sentence-transformers pulls in torch and
# transformers, a slow multi-GB import we must not trigger on the default tfidf path.
HAS_TRANSFORMERS = (
    importlib.util.find_spec('torch') is not None
    and importlib.util.find_spec('sentence_transformers') is not None
)

EMBEDDING_DIM = 128

_tfidf_pipeline: 'Pipeline | None' = None


class LogEncoder:
    """Encode SIEM log strings to fixed-dim float32 vectors."""

    def __init__(
        self,
        backend: Literal['tfidf', 'transformer'] = 'tfidf',
        cache_size: int = 512,
    ):
        self.backend = backend
        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self._encoder = (
            self._build_transformer()
            if backend == 'transformer'
            else self._build_tfidf()
        )

    def encode(self, text: str) -> np.ndarray:
        """Encode one log string."""
        if not text or not text.strip():
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        cache_key = hashlib.md5(text[:256].encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        vec = self._encoder(text)
        self._evict_if_full()
        self._cache[cache_key] = vec
        return vec

    def encode_buffer(self, log_lines: list, agg: str = 'mean') -> np.ndarray:
        """Encode a batch of log lines."""
        if not log_lines:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        str_lines = [s if isinstance(s, str) else str(s) for s in log_lines]
        vecs = np.stack([self.encode(s) for s in str_lines])
        if agg == 'max':
            return vecs.max(axis=0).astype(np.float32)
        return vecs.mean(axis=0).astype(np.float32)

    def _build_tfidf(self):
        global _tfidf_pipeline
        if _tfidf_pipeline is None:
            corpus = self._build_training_corpus()
            pipeline = Pipeline(
                [
                    (
                        'tfidf',
                        TfidfVectorizer(
                            analyzer='char_wb',
                            ngram_range=(3, 5),
                            max_features=4096,
                            sublinear_tf=True,
                        ),
                    ),
                    ('svd', TruncatedSVD(n_components=EMBEDDING_DIM, random_state=42)),
                    ('norm', Normalizer(norm='l2')),
                ]
            )
            pipeline.fit(corpus)
            _tfidf_pipeline = pipeline
            logger.info(
                'LogEncoder[tfidf]: fitted on %d docs -> %d-dim LSA.',
                len(corpus),
                EMBEDDING_DIM,
            )

        pipeline = _tfidf_pipeline

        def encode_fn(text: str) -> np.ndarray:
            vec = pipeline.transform([text])[0]
            if vec.shape[0] < EMBEDDING_DIM:
                padded = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                padded[: vec.shape[0]] = vec
                return padded
            return vec.astype(np.float32)

        return encode_fn

    def _build_transformer(self):
        if not HAS_TRANSFORMERS:
            logger.warning(
                'sentence-transformers not installed; falling back to tfidf.'
            )
            return self._build_tfidf()

        import torch
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('all-MiniLM-L6-v2')
        model.eval()
        rng = np.random.default_rng(42)
        proj = rng.standard_normal((384, EMBEDDING_DIM)).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8

        def encode_fn(text: str) -> np.ndarray:
            with torch.no_grad():
                emb = model.encode(text, convert_to_numpy=True)
            vec = (emb @ proj).astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-8) if norm > 0 else vec

        return encode_fn

    def _build_training_corpus(self) -> list:
        corpus: list = []

        lib_path = (
            Path(__file__).parent.parent / 'docker_bridge' / 'payload_library.json'
        )
        if lib_path.exists():
            lib = json.loads(lib_path.read_text())
            for action_data in lib.values():
                for outcomes in action_data.values():
                    corpus.extend(outcomes)

        sample_ips = ['10.0.0.1', '10.0.1.2', '192.168.1.5', '10.0.0.7', '10.0.1.9']
        for src, tgt in zip(sample_ips, reversed(sample_ips)):
            for fn in (evid_4624, evid_4625, evid_4648, evid_4776):
                corpus.append(fn(src, tgt))
            for proc in (
                'cmd.exe',
                'powershell.exe',
                'mimikatz.exe',
                'procdump.exe',
                'net.exe',
            ):
                corpus.append(evid_4688(src, process=proc))
                corpus.append(sysmon_1(src, process=proc))
            corpus.append(evid_4768(src, tgt))
            corpus.append(sysmon_3(src, tgt, dst_port=445))
            corpus.append(sysmon_3(src, tgt, dst_port=3389))
            corpus.append(sysmon_10(src))
            corpus.append(sysmon_22(src))

        for i in range(50):
            corpus.append(
                f'Synthetic noise event {i} for dimension stability - '
                f'{random.Random(i).random()}'
            )

        return corpus

    def _evict_if_full(self):
        if len(self._cache) >= self._cache_size:
            for k in list(self._cache.keys())[: self._cache_size // 4]:
                del self._cache[k]
