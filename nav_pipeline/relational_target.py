"""Compound target phrases -> a specific instance among same-class candidates.

Two distinct patterns, both handled the same way: split what DINO is bad at
(compositional/positional language) from what it's good at (plain category
detection), and resolve the rest with explicit geometry over already-good
detections, rather than trusting one grounding call to do it all jointly.

- RELATIONAL ("chair near the door"): two named objects. Detect both classes
  in one multi-class call, pick the target-class box nearest any anchor-class
  box.
- POSITIONAL ("rightmost person", "person in the middle"): one class, ranked
  by position among its own same-class candidates. Detect just that class,
  rank by pixel x-center.

Deliberately a hardcoded keyword list, not a parser: zero new dependencies,
zero added latency, deterministic -- consistent with the rest of this
pipeline's text handling (DINO prompt normalization, scene_tagger's vocab
list are all plain string ops too). Anything not matching a known keyword
falls through untouched to the caller's existing single-phrase detect() path.
"""

from typing import List, Optional, Tuple

import numpy as np

from .dino_detector import Detection

# Longest-first so "next to" / "close to" match before any shorter keyword
# that might be a substring of a longer one would.
RELATION_KEYWORDS = ["next to", "close to", "near", "by", "beside"]

_ARTICLES = ("the ", "a ", "an ")


def _strip_article(s: str) -> str:
    for art in _ARTICLES:
        if s.startswith(art):
            return s[len(art):]
    return s


def clean_label(label: str) -> str:
    """Normalize a DINO-returned label for exact-match comparison against a
    parsed target_class/anchor_class -- same cleanup scene_tagger.py already
    applies to labels from this same multi-class detect() call shape."""
    return _strip_article(label.strip().lower().rstrip("."))


def parse_relational_target(text: str) -> Optional[Tuple[str, str]]:
    """"chair near the door" -> ("chair", "door"), or None if no relation
    keyword is present (caller should fall back to plain single-phrase
    detection in that case)."""
    t = text.strip().lower().rstrip(".")
    for kw in RELATION_KEYWORDS:
        marker = f" {kw} "
        idx = t.find(marker)
        if idx == -1:
            continue
        target = _strip_article(t[:idx].strip())
        anchor = _strip_article(t[idx + len(marker):].strip())
        if target and anchor:
            return target, anchor
    return None


def _box_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def select_by_relation(target_dets: List[Detection], anchor_dets: List[Detection]) -> Optional[Detection]:
    """Among target_dets, pick whichever sits closest (pixel-center distance)
    to ANY anchor_dets box. Falls back to top detector score (target_dets[0],
    matching detect()'s score-sorted contract) if no anchor instance was
    detected this tick -- can't resolve "near the door" without a door."""
    if not target_dets:
        return None
    if not anchor_dets:
        return target_dets[0]
    anchor_centers = [_box_center(a.box) for a in anchor_dets]
    best, best_dist = None, float("inf")
    for d in target_dets:
        c = _box_center(d.box)
        dist = min(float(np.linalg.norm(c - ac)) for ac in anchor_centers)
        if dist < best_dist:
            best, best_dist = d, dist
    return best


# Longest-variant-first per keyword so e.g. "right most" (two words, as
# typed) matches before any shorter overlapping substring would.
POSITION_SYNONYMS = {
    "rightmost": ["rightmost", "right most", "right-most"],
    "leftmost": ["leftmost", "left most", "left-most"],
    "middle": ["in the middle", "in the center", "middle", "center"],
}


def parse_positional_target(text: str) -> Optional[Tuple[str, str]]:
    """"rightmost person" -> ("person", "rightmost"); "person in the middle"
    -> ("person", "middle"). Position word may come before OR after the noun
    -- whatever text remains once it's removed (minus articles/leftover
    prepositions) is the target class. Returns None if no position keyword
    is present (caller should fall back to plain single-phrase detection, or
    try parse_relational_target first)."""
    t = text.strip().lower().rstrip(".")
    for canonical, variants in POSITION_SYNONYMS.items():
        for kw in sorted(variants, key=len, reverse=True):
            idx = t.find(kw)
            if idx == -1:
                continue
            remainder = _trim_stray_words(t[:idx] + " " + t[idx + len(kw):])
            if remainder:
                return remainder, canonical
    return None


_STRAY_WORDS = ("in", "on", "at", "the", "a", "an")


def _trim_stray_words(s: str) -> str:
    """Strip leading/trailing filler words left over once a multi-word
    position keyword is removed -- e.g. "chair in middle" only matches the
    bare "middle" keyword (no "the"), leaving a dangling "in" that a plain
    article-prefix strip wouldn't catch."""
    words = s.split()
    while words and words[0] in _STRAY_WORDS:
        words.pop(0)
    while words and words[-1] in _STRAY_WORDS:
        words.pop()
    return " ".join(words)


def select_by_position(dets: List[Detection], position: str) -> Optional[Detection]:
    """Rank same-class dets by pixel x-center (0=left edge of image) and pick
    the requested extreme/middle one. Falls back to top detector score if
    `position` isn't recognized (shouldn't happen -- parse_positional_target
    only ever returns a key from POSITION_SYNONYMS)."""
    if not dets:
        return None
    ordered = sorted(dets, key=lambda d: _box_center(d.box)[0])
    if position == "leftmost":
        return ordered[0]
    if position == "rightmost":
        return ordered[-1]
    if position == "middle":
        return ordered[len(ordered) // 2]
    return dets[0]
