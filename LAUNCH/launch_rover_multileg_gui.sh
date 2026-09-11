#!/usr/bin/env bash
# ============================================================
#  Nav_new — real-rover Qwen+Grounding-DINO+NavDP multileg GUI
#  Run on the GPU machine:
#    ./LAUNCH/launch_rover_multileg_gui.sh [--rover|--hiwonder] [PI_IP]
#
#  Same Pi bring-up / conda env as launch_rover_multileg.sh, but opens
#  multileg_gui.py instead of running headless: live camera+detection
#  overlay, NavDP top-down trajectory, live BEV occupancy, the plan/
#  subtask status feed, and the "Recall & remember" goal-memory panel.
#  The instruction box starts blank -- type a free-text instruction and
#  hit Run; it resolves into a move:/turn:/object: plan and either spawns
#  the persistent --serve backend (first click -- models load once) or
#  sends the instruction live to an already-running one.
#
#  Everything this session's work landed on is baked into the Run
#  button's own launch args (see multileg_gui.py's on_run, not this
#  script): --goal-memory-approach, --goal-memory-appearance,
#  --supervisor-history-frames 2, --max-angular 0.25, auto-encoder-only
#  IMU detection. Nothing extra to pass here for those.
#
#  Examples:
#    ./LAUNCH/launch_rover_multileg_gui.sh 10.86.180.125
#    ./LAUNCH/launch_rover_multileg_gui.sh --rover 10.86.180.125
#    PI_PASS=... ./LAUNCH/launch_rover_multileg_gui.sh 192.168.0.42
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

source LAUNCH/_backend.sh
backend_parse_args "$@"
set -- "${BACKEND_ARGS[@]}"

backend_bringup camera

# Conda base + env differ per machine -- see launch_rover_multileg.sh's
# identical block for the full rationale; kept in sync with it.
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
# NOT forced to 1 by default (unlike some older launchers here) --
# live-hit 2026-09-09: openai/clip-vit-base-patch32's local cache carries a
# stale huggingface_hub negative-cache marker (.no_exist/<rev>/model.
# safetensors) that offline-mode resolution trusts even though the file is
# genuinely present in the snapshot dir, and offline mode appears to keep
# re-writing that stale marker rather than ever correcting it -- so forcing
# HF_HUB_OFFLINE=1 here made ClipVerifier's from_pretrained(use_safetensors=
# True) fail outright with "does not appear to have a file named model.
# safetensors" despite the file being right there. Only export it if the
# caller already had it set in their own environment.
[[ -n "${HF_HUB_OFFLINE:-}" ]] && export HF_HUB_OFFLINE

# Kill any stale GUI/server from a previous launch -- --serve's extra_args
# (goal-memory flags, --max-angular, etc.) only apply to a FRESH server
# spawn, so a leftover one from before a code/flag change silently keeps
# running the old behavior.
pkill -f "multileg_gui.py" 2>/dev/null && sleep 1
pkill -f "run_rover_multileg.py" 2>/dev/null && sleep 1

if [[ "$BACKEND" == "hiwonder" ]]; then
    warn "multileg_gui.py's Run button launches run_rover_multileg.py with ESP32-tuned defaults -- re-check on this chassis"
fi

info "Starting multileg GUI [$BACKEND] (pi-ip=$PI_IP)..."
exec python -u multileg_gui.py --pi-ip "$PI_IP" "$@"
