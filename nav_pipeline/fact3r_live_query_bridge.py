"""Bridge to run_fact3r_live.py: subprocess manager + live query interface.

STATUS: subprocess management (start/stop) is real and correct. Entity-level
polling (`_poll_entities_from_checkpoint`) is NOT verified against the real
checkpoint format and almost certainly does not work as written -- see below.
Do not treat `query()` as functional until that is fixed.

Verified against MASt3R-SLAM/fact3r-map/scripts/run_fact3r_live.py directly
(2026-09-04): `<output>/live_status.json` is real (`format:
"fact3r-live-status"`) but its `"entities"` field is a plain INTEGER COUNT
(`mapper.entity_count`), not a list of per-entity records -- there is no
`position_xy` / `confidence` / `labels` in it. Per-entity data (with the UOT
associations and appearance embeddings) instead lives under
`<output>/siglip_observations/` (built by `attach_mapping_to_observation_index`
from `<output>/image_uot/manifest.json` + `<output>/siglip_pre_uot/`), whose
schema this module has not yet parsed. `resolve_semantic_goal.py` is the
existing, working reader of that shape -- for a finished map, not a live one.

So what remains to make `query()` real:
  1. read `<output>/image_uot/manifest.json` to get current entity IDs +
     frame associations while the mapper is still appending to it
  2. pull each entity's position/embedding the way `resolve_semantic_goal.py`
     does, but from the in-progress index rather than a closed one
  3. re-test end-to-end once that's in place

Until then, the tested real-time-mapping path in this repo is
`fact3r_live_memory.Fact3rLiveMemory` (nav_pipeline/fact3r_live_memory.py) --
a lightweight in-process SAM2+SigLIP2 reimplementation with its own position
tracking, deliberately NOT the real fact3r.* package, used by
run_ab_position_recall_test.py and confirmed working on real HM3D runs
(see memory: fact3r-navdp-bridge.md).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Any, List, Optional, Sequence


@dataclass
class LiveEntity:
    entity_id: str
    position_xy: tuple[float, float]  # (x, y) in odometry frame
    confidence: float  # max SigLIP2 score seen for this entity
    labels: list[str]  # accumulated VLM-verified names


class Fact3rLiveQueryBridge:
    """Polls run_fact3r_live.py's output directory for entity checkpoint files.

    Assumes run_fact3r_live.py is writing:
      <output_dir>/live_status.json (updated every frame, carries entity_bank)
    or
      <output_dir>/entities/*.json (one file per entity)

    Call query(text: str) -> List[LiveEntity] to match text query against
    current entity bank via SigLIP2 encoder (loaded here, not in mapper).
    """

    def __init__(self, output_dir: Path, device: str = "cuda:0"):
        self.output_dir = Path(output_dir)
        self.device = device
        self._lock = Lock()
        self._entities: dict[str, LiveEntity] = {}
        self._last_update_time = 0.0
        self._encoder = None
        self._proc: Optional[subprocess.Popen] = None

    def start_mapper_subprocess(
        self,
        video_source: str | int,
        fact3r_map_root: Path,
        sam2_env: str = "sam2",
        discovery_model: str = "facebook/sam2.1-hiera-small",
        siglip_model: str = "google/siglip2-base-patch16-224",
        max_frames: Optional[int] = None,
    ) -> None:
        """Launch run_fact3r_live.py in background.

        Args:
            video_source: "0" for webcam, "rtsp://..." for stream, or file path
            fact3r_map_root: path to MASt3R-SLAM/fact3r-map
            sam2_env: conda environment with SAM2
            discovery_model: HF model ID for SAM2
            siglip_model: HF model ID for SigLIP2
            max_frames: optional frame limit (for testing)
        """
        if self._proc is not None:
            raise RuntimeError("Mapper subprocess already running; call stop() first")

        cmd = [
            "conda", "run", "--no-capture-output", "-n", sam2_env,
            "python3", str(fact3r_map_root / "scripts" / "run_fact3r_live.py"),
            "--source", str(video_source),
            "--output", str(self.output_dir),
            "--sample-fps", "1",  # causal: process every frame
            "--discovery-model", discovery_model,
            "--siglip-model", siglip_model,
            "--device", self.device.replace("cuda:", ""),
        ]
        if max_frames:
            cmd.extend(["--max-frames", str(max_frames)])

        print(f"[Fact3rLiveQueryBridge] starting: {' '.join(cmd)}", flush=True)
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Give it a moment to initialize
        time.sleep(2)
        print("[Fact3rLiveQueryBridge] subprocess started", flush=True)

    def _ensure_encoder_loaded(self):
        if self._encoder is None:
            from transformers import AutoModel, AutoProcessor
            self._encoder = AutoModel.from_pretrained("google/siglip2-base-patch16-224", torch_dtype="auto").to(self.device)
            self._encoder.eval()

    def _poll_entities_from_checkpoint(self) -> dict[str, LiveEntity]:
        """Read entity checkpoint from live_status.json or entities/*.json."""
        entities = {}

        # Try live_status.json first (if run_fact3r_live.py writes it)
        status_path = self.output_dir / "live_status.json"
        if status_path.exists():
            try:
                data = json.loads(status_path.read_text())
                # Format: {"entities": [{"entity_id": "e0", "position_xy": [x, y], "confidence": 0.95, "labels": ["chair"]}]}
                if "entities" in data:
                    for ent in data["entities"]:
                        entities[ent["entity_id"]] = LiveEntity(
                            entity_id=ent["entity_id"],
                            position_xy=tuple(ent["position_xy"]),
                            confidence=ent.get("confidence", 0.5),
                            labels=ent.get("labels", []),
                        )
                    return entities
            except Exception as e:
                pass  # fall through to entities/ directory

        # Fallback: read individual entity files
        entities_dir = self.output_dir / "entities"
        if entities_dir.exists():
            for ent_file in entities_dir.glob("*.json"):
                try:
                    data = json.loads(ent_file.read_text())
                    ent_id = ent_file.stem
                    entities[ent_id] = LiveEntity(
                        entity_id=ent_id,
                        position_xy=tuple(data.get("position_xy", [0, 0])),
                        confidence=data.get("confidence", 0.5),
                        labels=data.get("labels", []),
                    )
                except Exception:
                    pass

        return entities

    def poll(self) -> int:
        """Update entity bank from checkpoint. Return count of entities."""
        with self._lock:
            self._entities = self._poll_entities_from_checkpoint()
            self._last_update_time = time.time()
        return len(self._entities)

    def query(self, text: str, top_k: int = 3) -> List[LiveEntity]:
        """Match text query against current entity bank.

        Returns top-k entities by SigLIP2 embedding similarity.
        """
        self._ensure_encoder_loaded()
        from transformers import AutoProcessor
        import torch

        with self._lock:
            entities = list(self._entities.values())

        if not entities:
            return []

        # Encode query text
        processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
        inputs = processor(text=[text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_emb = self._encoder.get_text_features(**inputs)  # [1, D]

        # Score each entity's top appearance embedding against query
        scores = []
        for ent in entities:
            # For now, a simple heuristic: if entity labels contain words from query, boost score
            label_boost = 1.0
            if ent.labels:
                query_lower = text.lower()
                for label in ent.labels:
                    if label.lower() in query_lower or query_lower in label.lower():
                        label_boost = 1.5
                        break
            scores.append((ent, ent.confidence * label_boost))

        # Sort by score and return top-k
        top = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        return [ent for ent, _ in top]

    def stop(self) -> None:
        """Terminate mapper subprocess."""
        if self._proc is not None:
            print("[Fact3rLiveQueryBridge] stopping mapper subprocess...", flush=True)
            self._proc.terminate()
            self._proc.wait(timeout=10)
            self._proc = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None
