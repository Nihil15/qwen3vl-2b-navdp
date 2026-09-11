"""Encode-once SigLIP cache: ask before every goal-set whether this thing has
already been encoded; encode fresh on a miss, reuse the stored vector on a hit.

The pipeline re-embeds a goal's text (and an obstacle's crop) every time a goal
is set -- once on the outbound leg, again on recall, again for each waypoint,
again for a directional anchor. That is wasted compute AND a source of drift:
two encodings of "the cabinet" taken seconds apart are not bit-identical, so a
similarity gate can flip between legs. This memoizes both directions of
``Siglip2Embedder`` (``encode_text`` / ``embed``) behind a normalized key, and
persists to a ``.npz`` so a later recall *process* reuses the outbound run's
vectors too, not just a later call in the same process.

Keys are normalized (lowercase, whitespace-collapsed, leading article and
navigation prefix stripped) so "go to the cabinet", "the Cabinet" and
"cabinet" are one entry. A directional key keeps its ``<dir>::<anchor>``
shape, each side normalized independently.

The embedder is optional: constructed with ``embedder=None`` the cache still
serves anything already on disk (and raises on a genuine miss), which is what
the unit tests and an offline replay use.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_NAV_PREFIXES = (
    "go back to ", "go to ", "return to ", "navigate to ", "head to ",
    "walk to ", "drive to ", "back to ", "go towards ", "go toward ",
    "move to ", "proceed to ",
)


class SiglipEncodingCache:
    def __init__(self, embedder=None, path: Optional[str | Path] = None):
        # `embedder`: any object exposing encode_text(list[str]) -> (N, D) and
        # embed(rgb, box) -> (D,) | None -- Siglip2Embedder satisfies both.
        self.embedder = embedder
        self.path = Path(path) if path else None
        self._text: dict[str, np.ndarray] = {}
        self._image: dict[str, np.ndarray] = {}
        self.stats = {"text_hit": 0, "text_miss": 0, "image_hit": 0, "image_miss": 0}
        if self.path and self.path.exists():
            self.load(self.path)

    # -- key normalization ------------------------------------------- #
    @staticmethod
    def _norm_one(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[^a-z0-9 :_-]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        for pre in _NAV_PREFIXES:
            if s.startswith(pre):
                s = s[len(pre):]
                break
        s = re.sub(r"^(the|a|an) ", "", s).strip()
        return s

    @classmethod
    def normalize_key(cls, key: str) -> str:
        if "::" in key:
            return "::".join(cls._norm_one(p) for p in key.split("::"))
        return cls._norm_one(key)

    # -- text ------------------------------------------------------- #
    def has_text(self, key: str) -> bool:
        return self.normalize_key(key) in self._text

    def ensure_text(self, key: str) -> Tuple[np.ndarray, bool]:
        """Return ``(embedding, was_cached)`` for `key`'s text encoding."""
        nk = self.normalize_key(key)
        cached = self._text.get(nk)
        if cached is not None:
            self.stats["text_hit"] += 1
            return cached, True
        if self.embedder is None:
            raise RuntimeError(f"SiglipEncodingCache: no embedder and text key not cached: {nk!r}")
        emb = np.asarray(self.embedder.encode_text([nk]), dtype=np.float32)[0]
        self._text[nk] = emb
        self.stats["text_miss"] += 1
        return emb, False

    # -- image ---------------------------------------------------- #
    def has_image(self, key: str) -> bool:
        return self.normalize_key(key) in self._image

    def ensure_image(self, key: str, rgb: np.ndarray, box) -> Tuple[Optional[np.ndarray], bool]:
        """Return ``(embedding|None, was_cached)`` for `key`'s image-crop
        encoding. `key` is a caller-chosen stable id (e.g.
        ``f"obstacle::{goal_key}"``), not derived from pixels. A degenerate
        crop (embedder returns None) is not cached and comes back as
        ``(None, False)``."""
        nk = self.normalize_key(key)
        cached = self._image.get(nk)
        if cached is not None:
            self.stats["image_hit"] += 1
            return cached, True
        if self.embedder is None:
            raise RuntimeError(f"SiglipEncodingCache: no embedder and image key not cached: {nk!r}")
        emb = self.embedder.embed(rgb, np.asarray(box, dtype=float))
        if emb is None:
            return None, False
        emb = np.asarray(emb, dtype=np.float32)
        self._image[nk] = emb
        self.stats["image_miss"] += 1
        return emb, False

    # -- persistence ------------------------------------------------ #
    def save(self, path: Optional[str | Path] = None) -> Path:
        p = Path(path or self.path)
        if p is None:
            raise ValueError("SiglipEncodingCache.save needs a path")
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {}
        for i, (k, v) in enumerate(self._text.items()):
            payload[f"tk{i}"] = np.frombuffer(k.encode("utf-8"), dtype=np.uint8)
            payload[f"tv{i}"] = np.asarray(v, dtype=np.float32)
        for i, (k, v) in enumerate(self._image.items()):
            payload[f"ik{i}"] = np.frombuffer(k.encode("utf-8"), dtype=np.uint8)
            payload[f"iv{i}"] = np.asarray(v, dtype=np.float32)
        payload["meta"] = np.array([len(self._text), len(self._image)], dtype=np.int64)
        np.savez(p, **payload)
        return p if p.suffix else p.with_suffix(".npz")

    def load(self, path: Optional[str | Path] = None) -> None:
        p = Path(path or self.path)
        if not p.exists() and p.with_suffix(".npz").exists():
            p = p.with_suffix(".npz")
        with np.load(p) as z:
            n_text, n_image = (int(x) for x in z["meta"])
            for i in range(n_text):
                k = bytes(z[f"tk{i}"].tolist()).decode("utf-8")
                self._text.setdefault(k, np.asarray(z[f"tv{i}"], dtype=np.float32))
            for i in range(n_image):
                k = bytes(z[f"ik{i}"].tolist()).decode("utf-8")
                self._image.setdefault(k, np.asarray(z[f"iv{i}"], dtype=np.float32))

    def summary(self) -> str:
        s = self.stats
        return (f"{len(self._text)} text + {len(self._image)} image encodings "
                f"(text {s['text_hit']} hit / {s['text_miss']} miss, "
                f"image {s['image_hit']} hit / {s['image_miss']} miss)")
