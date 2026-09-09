"""LiveEntity.position weighting checks for nav_pipeline.fact3r_live_memory.

Pure arithmetic on the dataclass -- no SAM2 / SigLIP / torch. Run:

    python -m pytest tests/test_fact3r_live_memory.py
    # or plain:
    python tests/test_fact3r_live_memory.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nav_pipeline.fact3r_live_memory import (  # noqa: E402
    Fact3rLiveMemory,
    LiveEntity,
    _enforce_reason_decision_consistency,
)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_weighted_position_favours_close_high_quality_observation():
    """The measured real-world failure this fix targets: a couple of far,
    partial glimpses badly disagreeing with one close, clear one. The plain
    median (pre-fix behaviour) lands near the bad far cluster; the
    quality-weighted mean should land near the close, trustworthy one."""
    true_pos = (1.24, -4.98)
    ent = LiveEntity(
        entity_id="e",
        positions=[(1.10, -4.90), (-2.5, -8.0), (-2.0, -9.5)],
        qualities=[3000.0, 60.0, 45.0],   # valid-depth pixel counts
        depths=[1.4, 6.5, 7.8],           # metres
    )
    median = (float(np.median([1.10, -2.5, -2.0])), float(np.median([-4.90, -8.0, -9.5])))
    weighted = ent.position
    assert dist(weighted, true_pos) < dist(median, true_pos) / 2, (
        f"weighted {weighted} should be much closer to {true_pos} than the "
        f"plain median {median}")
    assert dist(weighted, true_pos) < 0.5


def test_position_falls_back_to_median_without_quality_data():
    """Entities restored without qualities/depths (e.g. an older run's
    checkpoint) must not crash or silently drop observations."""
    ent = LiveEntity(entity_id="e", positions=[(1.0, 2.0), (3.0, 2.0), (2.0, 2.0)])
    assert ent.position == (2.0, 2.0)  # plain median, unaffected by the fix


def test_single_observation_returns_itself():
    ent = LiveEntity(entity_id="e", positions=[(4.0, -1.0)], qualities=[500.0], depths=[2.0])
    assert ent.position == (4.0, -1.0)


def test_equal_quality_observations_average_like_a_mean():
    ent = LiveEntity(
        entity_id="e",
        positions=[(0.0, 0.0), (2.0, 0.0)],
        qualities=[100.0, 100.0], depths=[2.0, 2.0],
    )
    x, y = ent.position
    assert abs(x - 1.0) < 1e-9 and abs(y - 0.0) < 1e-9


def test_position_spread_stays_unweighted():
    """position_spread is a raw-scatter sanity check, deliberately not
    quality-weighted (see its docstring) -- it should not shrink just
    because the bad observations would be down-weighted in `position`."""
    ent = LiveEntity(
        entity_id="e",
        positions=[(0.0, 0.0), (0.0, 0.0), (10.0, 0.0)],
        qualities=[1000.0, 1000.0, 1.0], depths=[1.0, 1.0, 8.0],
    )
    assert ent.position_spread > 3.0  # the far outlier still shows up in spread


def test_weighted_spread_forgives_low_weight_outliers():
    """The measured real case: one good close observation + a couple of far
    low-quality outliers gives an accurate weighted position but a large
    raw spread. Weighted spread should read much lower than raw spread
    here -- it's measuring deviation from a mean the outliers barely moved."""
    ent = LiveEntity(
        entity_id="e",
        positions=[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-8.0, 3.0), (7.0, -4.0)],
        qualities=[2000.0, 1800.0, 1900.0, 40.0, 35.0],
        depths=[1.5, 1.6, 1.5, 7.0, 7.5],
    )
    assert ent.position_spread_weighted < ent.position_spread / 3


def test_weighted_spread_still_flags_genuine_fragmentation():
    """Two comparable-weight clusters far apart (a genuinely fragmented /
    mis-associated entity) should NOT be forgiven -- weighted spread must
    stay high here too, since the weighted mean sits between two real
    clusters and both are far from it."""
    ent = LiveEntity(
        entity_id="e",
        positions=[(0.0, 0.0), (0.2, 0.0), (8.0, 0.0), (8.2, 0.0)],
        qualities=[500.0, 500.0, 500.0, 500.0],
        depths=[2.0, 2.0, 2.0, 2.0],
    )
    assert ent.position_spread_weighted > 2.0


def test_reason_decision_contradiction_is_overridden_to_no():
    """The measured real failure: query 'the cabinet', Qwen's own reason said
    'the context shows it is part of a bed ... indicating it is a bed
    headboard' and then decision='yes' anyway. The override should catch
    this exact contradiction."""
    verdict = {"decision": "yes", "confidence": 0.95, "predicted_object": "cabinet",
               "reason": "The object is a wooden cabinet with a visible front panel. "
                         "The context shows it is part of a bed, with a mattress and "
                         "blanket visible, indicating it is a bed headboard."}
    fixed = _enforce_reason_decision_consistency("the cabinet", verdict)
    assert fixed["decision"] == "no"
    assert "overridden" in fixed["reason"]


def test_consistent_yes_is_left_alone():
    verdict = {"decision": "yes", "confidence": 0.95, "predicted_object": "cabinet",
               "reason": "A freestanding dark wood cabinet with two doors, no bed nearby."}
    fixed = _enforce_reason_decision_consistency("the cabinet", verdict)
    assert fixed == verdict


def test_no_decision_is_not_touched_by_override():
    verdict = {"decision": "no", "confidence": 0.9, "predicted_object": "headboard",
               "reason": "This is a bed headboard, not a cabinet."}
    fixed = _enforce_reason_decision_consistency("the cabinet", verdict)
    assert fixed == verdict


def test_reason_decision_contradiction_is_overridden_to_yes():
    """The mirror measured real failure: query 'a table', Qwen's own reason
    said '...the entity is clearly a table... no better-fitting lookalike
    exists in the context' and then decision='no' anyway."""
    verdict = {"decision": "no", "confidence": 1.0, "predicted_object": "",
               "reason": "The entity is a large, rectangular, flat surface with a dark "
                         "finish. It is not a bed, headboard, or cabinet. The entity is "
                         "clearly a table, not a bed or other furniture. The object is a "
                         "table, and no better-fitting lookalike exists in the context."}
    fixed = _enforce_reason_decision_consistency("a table", verdict)
    assert fixed["decision"] == "yes"
    assert "overridden" in fixed["reason"]


def test_consistent_no_is_left_alone():
    verdict = {"decision": "no", "confidence": 0.99, "predicted_object": "nightstand",
               "reason": "This is a nightstand beside the bed, not a dining table."}
    fixed = _enforce_reason_decision_consistency("a table", verdict)
    assert fixed == verdict


def _unit_embedding(seed_vec):
    v = np.asarray(seed_vec, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_association_ignores_position_with_few_observations():
    """Early on, an entity's position is still noisy -- appearance alone has
    to carry confidence (can't gate on position yet). A same-looking new
    proposal far away should still associate while the entity has fewer
    than position_gate_min_obs observations."""
    ent = LiveEntity("live-0000", embeddings=[_unit_embedding([1.0, 0.0])],
                     positions=[(0.0, 0.0)], qualities=[100.0], depths=[2.0])
    emb = _unit_embedding([1.0, 0.01])
    got = Fact3rLiveMemory.select_association_target(
        [ent], emb, ox=10.0, oy=10.0, threshold=0.90,
        position_gate_m=3.0, position_gate_min_obs=5)
    assert got is ent


def test_association_position_gate_blocks_far_lookalike_once_settled():
    """Real bug (2026-09-05, live run): a real house has several visually-
    similar flat/reflective surfaces (tables, counters, shelves) -- pure
    appearance association let one entity absorb 495 observations across a
    3.57m spread. Once an entity has enough observations to trust its
    position, a same-looking proposal far outside the gate radius must NOT
    associate to it, even though it would win on appearance alone."""
    ent = LiveEntity("live-0000",
                     embeddings=[_unit_embedding([1.0, 0.0])] * 5,
                     positions=[(0.0, 0.0)] * 5,
                     qualities=[100.0] * 5, depths=[2.0] * 5)
    emb = _unit_embedding([1.0, 0.01])   # near-identical appearance
    got = Fact3rLiveMemory.select_association_target(
        [ent], emb, ox=5.0, oy=5.0, threshold=0.90,   # ~7m away
        position_gate_m=3.0, position_gate_min_obs=5)
    assert got is None


def test_association_position_gate_allows_nearby_settled_match():
    """The gate must not block genuine re-observations of the same settled
    entity from a plausible nearby position."""
    ent = LiveEntity("live-0000",
                     embeddings=[_unit_embedding([1.0, 0.0])] * 5,
                     positions=[(0.0, 0.0)] * 5,
                     qualities=[100.0] * 5, depths=[2.0] * 5)
    emb = _unit_embedding([1.0, 0.01])
    got = Fact3rLiveMemory.select_association_target(
        [ent], emb, ox=0.5, oy=0.5, threshold=0.90,
        position_gate_m=3.0, position_gate_min_obs=5)
    assert got is ent


def test_association_falls_back_to_none_when_only_candidate_is_gated_out():
    """A far, gated-out entity must not be returned just because it's the
    only candidate -- association should refuse (caller starts a new
    entity), not silently pick a spatially implausible match."""
    ent = LiveEntity("live-0000",
                     embeddings=[_unit_embedding([1.0, 0.0])] * 5,
                     positions=[(0.0, 0.0)] * 5,
                     qualities=[100.0] * 5, depths=[2.0] * 5)
    emb = _unit_embedding([1.0, 0.0])   # perfect appearance match
    got = Fact3rLiveMemory.select_association_target(
        [ent], emb, ox=8.0, oy=0.0, threshold=0.90,
        position_gate_m=3.0, position_gate_min_obs=5)
    assert got is None


def test_merge_fuses_close_similar_fragments():
    """The measured real case: one physical object fragmented into several
    entities by the live per-tick threshold -- close positions, similar
    appearance. Should merge into one, losing the others."""
    mem = Fact3rLiveMemory()
    e_dir = _unit_embedding([1.0, 0.02, 0.0])
    mem.entities = [
        LiveEntity("live-0000", embeddings=[_unit_embedding([1.0, 0.00, 0.0])],
                   positions=[(0.0, 0.0)], qualities=[500.0], depths=[2.0]),
        LiveEntity("live-0001", embeddings=[_unit_embedding([1.0, 0.01, 0.0])],
                   positions=[(0.2, 0.1)], qualities=[400.0], depths=[2.2]),
        LiveEntity("live-0002", embeddings=[_unit_embedding([1.0, 0.03, 0.0])],
                   positions=[(-0.1, 0.3)], qualities=[300.0], depths=[2.5]),
    ]
    n_merged = mem.merge_nearby_duplicates(position_radius=0.6, appearance_threshold=0.75)
    assert n_merged == 2
    assert len(mem.entities) == 1
    assert mem.entities[0].observations == 3  # all 3 fragments' positions kept


def test_merge_leaves_close_but_different_looking_objects_separate():
    """A nightstand standing right next to a bed: close in space, NOT the
    same object -- appearance similarity should block the merge."""
    mem = Fact3rLiveMemory()
    mem.entities = [
        LiveEntity("live-0000", embeddings=[_unit_embedding([1.0, 0.0, 0.0])],
                   positions=[(0.0, 0.0)], qualities=[500.0], depths=[2.0]),
        LiveEntity("live-0001", embeddings=[_unit_embedding([0.0, 1.0, 0.0])],  # orthogonal = 0 similarity
                   positions=[(0.3, 0.0)], qualities=[500.0], depths=[2.0]),
    ]
    n_merged = mem.merge_nearby_duplicates(position_radius=0.6, appearance_threshold=0.75)
    assert n_merged == 0
    assert len(mem.entities) == 2


def test_merge_leaves_similar_looking_far_apart_objects_separate():
    """Two near-identical chairs in different rooms: appearance matches,
    position doesn't -- should NOT merge (that would be a mis-association,
    not a fragment repair)."""
    mem = Fact3rLiveMemory()
    mem.entities = [
        LiveEntity("live-0000", embeddings=[_unit_embedding([1.0, 0.0, 0.0])],
                   positions=[(0.0, 0.0)], qualities=[500.0], depths=[2.0]),
        LiveEntity("live-0001", embeddings=[_unit_embedding([1.0, 0.0, 0.0])],
                   positions=[(8.0, 3.0)], qualities=[500.0], depths=[2.0]),
    ]
    n_merged = mem.merge_nearby_duplicates(position_radius=0.6, appearance_threshold=0.75)
    assert n_merged == 0
    assert len(mem.entities) == 2


def test_merge_transitive_chain_fuses_into_one_group():
    """A close+similar to B, B close+similar to C, but A and C not directly
    compared as close -- union-find should still fuse all three."""
    mem = Fact3rLiveMemory()
    mem.entities = [
        LiveEntity("A", embeddings=[_unit_embedding([1.0, 0.0])], positions=[(0.0, 0.0)],
                   qualities=[100.0], depths=[2.0]),
        LiveEntity("B", embeddings=[_unit_embedding([1.0, 0.01])], positions=[(0.5, 0.0)],
                   qualities=[100.0], depths=[2.0]),
        LiveEntity("C", embeddings=[_unit_embedding([1.0, 0.02])], positions=[(1.0, 0.0)],
                   qualities=[100.0], depths=[2.0]),
    ]
    n_merged = mem.merge_nearby_duplicates(position_radius=0.6, appearance_threshold=0.75)
    assert n_merged == 2
    assert len(mem.entities) == 1


def test_merge_long_chain_does_not_blow_past_diameter_cap():
    """Real bug (2026-09-05, live run): 5 fragments each 0.5m from the next
    (every single hop well under position_radius=0.6) chain-merged via plain
    union-find into ONE entity spanning 2.0m end to end -- several genuinely
    different real objects folded together, not one physical table. With
    max_cluster_diameter=1.2 (default), the chain must NOT fully fuse: no
    resulting group may span more than 1.2m internally, even though every
    adjacent pair individually cleared position_radius."""
    mem = Fact3rLiveMemory()
    mem.entities = [
        LiveEntity(chr(65 + i), embeddings=[_unit_embedding([1.0, 0.01 * i])],
                   positions=[(0.5 * i, 0.0)], qualities=[100.0], depths=[2.0])
        for i in range(5)   # positions 0.0, 0.5, 1.0, 1.5, 2.0 -- span 2.0m
    ]
    mem.merge_nearby_duplicates(position_radius=0.6, appearance_threshold=0.75,
                                max_cluster_diameter=1.2)
    for ent in mem.entities:
        pts = ent.positions
        span = max(dist(pts[a], pts[b]) for a in range(len(pts)) for b in range(len(pts)))
        assert span <= 1.2 + 1e-9, f"{ent.entity_id} spans {span}m internally, over the cap"
    assert len(mem.entities) > 1, "the whole chain fully fused despite exceeding the diameter cap"


def test_merge_caps_views_at_max_views_per_entity():
    mem = Fact3rLiveMemory(max_views_per_entity=3)
    make_crop = lambda v: np.zeros((4, 4, 3), dtype=np.uint8) + v
    mem.entities = [
        LiveEntity("A", embeddings=[_unit_embedding([1.0, 0.0])] * 3,
                   positions=[(0.0, 0.0)], qualities=[100.0], depths=[2.0],
                   crops=[make_crop(1)] * 3, context_crops=[make_crop(1)] * 3),
        LiveEntity("B", embeddings=[_unit_embedding([1.0, 0.01])] * 3,
                   positions=[(0.1, 0.0)], qualities=[100.0], depths=[2.0],
                   crops=[make_crop(2)] * 3, context_crops=[make_crop(2)] * 3),
    ]
    mem.merge_nearby_duplicates(position_radius=0.6, appearance_threshold=0.75)
    assert len(mem.entities) == 1
    assert len(mem.entities[0].embeddings) <= 3
    assert len(mem.entities[0].crops) <= 3


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
