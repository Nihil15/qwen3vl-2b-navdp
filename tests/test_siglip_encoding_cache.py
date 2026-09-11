"""Encode-once behaviour + persistence for nav_pipeline.siglip_encoding_cache.

Pure Python -- a fake embedder that counts calls stands in for SigLIP2. Run:

    python -m pytest tests/test_siglip_encoding_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nav_pipeline.siglip_encoding_cache import SiglipEncodingCache  # noqa: E402


class FakeEmbedder:
    def __init__(self, dim: int = 8):
        self.dim = dim
        self.text_calls = 0
        self.image_calls = 0

    def _vec(self, seed: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(seed)) % (2 ** 32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def encode_text(self, texts):
        self.text_calls += 1
        return np.stack([self._vec("t:" + t) for t in texts])

    def embed(self, rgb, box):
        self.image_calls += 1
        return self._vec(f"i:{int(np.asarray(rgb).sum())}:{tuple(np.asarray(box).tolist())}")


def test_normalize_key_folds_prefixes_articles_and_case():
    n = SiglipEncodingCache.normalize_key
    assert n("Go to the Cabinet") == "cabinet"
    assert n("the  cabinet ") == "cabinet"
    assert n("go back to the blue box") == "blue box"
    assert n("behind::the Big Cabinet") == "behind::big cabinet"


def test_text_encoding_is_computed_once_then_reused():
    emb = FakeEmbedder()
    cache = SiglipEncodingCache(embedder=emb)

    v1, cached1 = cache.ensure_text("go to the cabinet")
    assert cached1 is False and emb.text_calls == 1

    v2, cached2 = cache.ensure_text("the CABINET")     # same after normalization
    assert cached2 is True and emb.text_calls == 1
    assert np.allclose(v1, v2)

    _, cached3 = cache.ensure_text("a different landmark")
    assert cached3 is False and emb.text_calls == 2
    assert cache.stats == {"text_hit": 1, "text_miss": 2, "image_hit": 0, "image_miss": 0}


def test_image_encoding_cached_by_caller_key():
    emb = FakeEmbedder()
    cache = SiglipEncodingCache(embedder=emb)
    rgb = np.ones((16, 16, 3), dtype=np.uint8)
    box = [1, 2, 9, 14]

    v1, c1 = cache.ensure_image("obstacle::behind::cabinet", rgb, box)
    v2, c2 = cache.ensure_image("obstacle::behind::cabinet", rgb, box)
    assert c1 is False and c2 is True and emb.image_calls == 1
    assert np.allclose(v1, v2)


def test_persistence_round_trip_serves_hits_without_an_embedder(tmp_path):
    emb = FakeEmbedder()
    cache = SiglipEncodingCache(embedder=emb)
    tv, _ = cache.ensure_text("behind::cabinet")
    iv, _ = cache.ensure_image("obstacle::behind::cabinet",
                               np.ones((8, 8, 3), dtype=np.uint8), [0, 0, 8, 8])
    p = cache.save(tmp_path / "enc_cache.npz")

    reloaded = SiglipEncodingCache(embedder=None, path=p)
    got, cached = reloaded.ensure_text("behind::the cabinet")   # -> stored key "behind::cabinet"
    assert cached is True and np.allclose(got, tv)
    assert reloaded.has_image("obstacle::behind::cabinet")
    assert np.allclose(reloaded._image["obstacle::behind::cabinet"], iv)


def test_miss_without_embedder_raises():
    cache = SiglipEncodingCache(embedder=None)
    try:
        cache.ensure_text("never encoded")
    except RuntimeError as e:
        assert "not cached" in str(e)
    else:
        raise AssertionError("expected RuntimeError on an uncached miss with no embedder")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
