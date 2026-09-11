#!/usr/bin/env bash
# ============================================================
#  Nav_new -- real-rover OCCLUDED GO-BEHIND + RECALL GUI
#    ./LAUNCH/launch_rover_occluded_recall_gui.sh [--rover|--hiwonder] [PI_IP]
#
#  Same viewer as launch_rover_multileg_gui.sh (live camera+detection
#  overlay, NavDP top-down, live BEV occupancy, plan/subtask status feed,
#  Recall & remember panel) but the Run button spawns
#  run_rover_occluded_recall.py --serve instead of run_rover_multileg.py.
#
#  That runner adds two clause kinds the instruction box understands:
#    * "go behind|beyond|around|left of|right of the <X>"  -> directional leg:
#      DINO locates X, the rover drives to a point past X's occlusion
#      shadow, recording a visual-subgoal spine (a forced keyframe at every
#      sharp turn) and SigLIP-encoding the opaque obstacle it rounds.
#    * "recall <B>" / "go back to <B>" / "return to <B>"   -> recall leg:
#      follow the reversed subgoal spine (A* only for local repair),
#      re-identify the obstacle from its SigLIP signature en route, then
#      hand the final approach to DINO + the Qwen supervisor.
#
#  Type ONE combined instruction and hit Run, e.g.
#     go behind the black curtain, then recall the computer monitor
#  (pre-filled below). Artifacts land in qwen3vl-2b-navdp/logs/occluded_recall/
#  : subgoals.json, obstacles.jsonl, encoding_cache.npz, result.json, topdown.png.
#
#  Supervisor defaults to the 2B model (the GUI's own combobox / on_run args);
#  --max-angular 0.25 and the goal-memory flags are baked into the Run button
#  exactly as in launch_rover_multileg_gui.sh.
#
#  Examples:
#    ./LAUNCH/launch_rover_occluded_recall_gui.sh 10.146.234.125
#    ./LAUNCH/launch_rover_occluded_recall_gui.sh --rover 10.146.234.125
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

source LAUNCH/_backend.sh
backend_parse_args "$@"
set -- "${BACKEND_ARGS[@]}"

backend_bringup camera

set +u
if [[ -n "${MULTILEG_CONDA_SH:-}" ]]; then
    source "$MULTILEG_CONDA_SH"; conda activate "${MULTILEG_ENV:-navdp}"
elif [[ -f /home/gpu/miniconda3/etc/profile.d/conda.sh ]]; then
    source /home/gpu/miniconda3/etc/profile.d/conda.sh; conda activate "${MULTILEG_ENV:-navdp}"
elif [[ -f /home/i3d/exit/etc/profile.d/conda.sh ]]; then
    source /home/i3d/exit/etc/profile.d/conda.sh; conda activate "${MULTILEG_ENV:-internnav}"
else
    warn "no known conda.sh found -- set MULTILEG_CONDA_SH / MULTILEG_ENV; using current python"
fi
set -u
export HF_HOME=${HF_HOME:-/home/gpu/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/home/gpu/.cache/huggingface/hub}
# Do NOT force HF_HUB_OFFLINE=1 -- it breaks ClipVerifier's safetensors load
# (see launch_rover_multileg_gui.sh's note). Only pass it through if the
# caller already set it.
[[ -n "${HF_HUB_OFFLINE:-}" ]] && export HF_HUB_OFFLINE

# Kill any stale GUI/server from a previous launch (either runner) -- --serve
# extra_args only apply to a FRESH spawn.
pkill -f "multileg_gui.py" 2>/dev/null && sleep 1
pkill -f "run_rover_multileg.py" 2>/dev/null && sleep 1
pkill -f "run_rover_occluded_recall.py" 2>/dev/null && sleep 1

if [[ "$BACKEND" == "hiwonder" ]]; then
    warn "the Run button spawns run_rover_occluded_recall.py with ESP32-tuned defaults -- re-check on this chassis"
fi

info "Starting occluded go-behind + recall GUI [$BACKEND] (pi-ip=$PI_IP)..."
exec python -u multileg_gui.py --pi-ip "$PI_IP" \
    --runner run_rover_occluded_recall.py \
    --default-instruction "go behind the black curtain, then recall the computer monitor" \
    "$@"
