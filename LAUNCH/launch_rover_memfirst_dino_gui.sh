#!/usr/bin/env bash
# ============================================================
#  Nav_new -- real-rover MEMORY-FIRST + MULTI-CANDIDATE-VERIFY multileg GUI
#    ./LAUNCH/launch_rover_memfirst_dino_gui.sh [--rover|--hiwonder] [PI_IP]
#
#  Same viewer as launch_rover_multileg_gui.sh (live camera+detection
#  overlay, NavDP top-down, live BEV occupancy, plan/subtask status feed,
#  Recall & remember panel), but the Run button spawns
#  run_rover_memfirst_dino.py --serve instead of run_rover_multileg.py.
#  launch_rover_multileg_gui.sh and run_rover_multileg.py are NOT modified.
#
#  What that runner changes (all ON by default there):
#    1. MEMORY FIRST, AUTHORITATIVE -- every "object:<phrase>" leg checks
#       memory first. On a hit it drives to the remembered point and accepts
#       arrival there as success -- no fresh Grounding-DINO search. Detection
#       only runs when memory has nothing (or the blind drive fell short).
#    2. MANY DINO BOXES, QWEN PICKS -- before the first lock, if Grounding-DINO
#       returns >=2 comparably-scored boxes for the phrase, all their crops go
#       to the Qwen supervisor, which picks the one the instruction means;
#       that both refines the phrase and seeds the tracker's first lock.
#    3. FACT3R LIVE ENTITY MAP -- while driving ANY leg, SAM2 proposes every
#       object in the frame, SigLIP2 embeds each, and it's stored with a
#       metric position. A later "go to X" recalls X's position even if the
#       rover only saw it in passing (never navigated to it). Persisted to
#       logs/live_fact3r_memory.json across restarts.  Needs SAM2 (SAM2_ROOT).
#
#  The Run button's own baked-in args (--supervisor-model-id, --max-angular
#  0.25, --goal-memory-approach, --goal-memory-appearance,
#  --supervisor-history-frames 2, the lowered DINO score floors) are passed
#  through unchanged -- run_rover_memfirst_dino.py inherits the same CLI.
#
#  Examples:
#    ./LAUNCH/launch_rover_memfirst_dino_gui.sh 10.146.234.125
#    ./LAUNCH/launch_rover_memfirst_dino_gui.sh --rover 10.146.234.125
#    PI_PASS=... ./LAUNCH/launch_rover_memfirst_dino_gui.sh 192.168.0.42
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

source LAUNCH/_backend.sh
backend_parse_args "$@"
set -- "${BACKEND_ARGS[@]}"

backend_bringup camera

# Conda base + env differ per machine -- see launch_rover_multileg_gui.sh's
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
# fact3r live entity map (default ON in run_rover_memfirst_dino.py) discovers
# objects with SAM2 -- nav_pipeline/fact3r_live_memory.py adds this dir to
# sys.path to import the `sam2` package. Default matches its own hardcoded
# fallback; override by exporting SAM2_ROOT before calling this script.
export SAM2_ROOT=${SAM2_ROOT:-/home/gpu/Desktop/pineapple/packages/sam2}
# Do NOT force HF_HUB_OFFLINE=1 -- it breaks ClipVerifier's safetensors load
# (see launch_rover_multileg_gui.sh's note). Only pass it through if the
# caller already set it.
[[ -n "${HF_HUB_OFFLINE:-}" ]] && export HF_HUB_OFFLINE

# Kill any stale GUI/server from a previous launch (either runner) -- --serve
# extra_args only apply to a FRESH spawn.
pkill -f "multileg_gui.py" 2>/dev/null && sleep 1
pkill -f "run_rover_multileg.py" 2>/dev/null && sleep 1
pkill -f "run_rover_memfirst_dino.py" 2>/dev/null && sleep 1

if [[ "$BACKEND" == "hiwonder" ]]; then
    warn "the Run button spawns run_rover_memfirst_dino.py with ESP32-tuned defaults -- re-check on this chassis"
fi

info "Starting memory-first + candidate-verify GUI [$BACKEND] (pi-ip=$PI_IP)..."
# 8B supervisor by default here: it is the crop verifier for fact3r recall
# (query_verified) and the 2B is too weak a judge -- live-observed 2026-09-10,
# every fact3r recall failed at "nothing passed Qwen verification". 8B is
# fully cached (the 7B cache on this box is a broken partial download).
# Still switchable in the GUI combobox.
exec python -u multileg_gui.py --pi-ip "$PI_IP" \
    --runner run_rover_memfirst_dino.py \
    --supervisor-model Qwen/Qwen3-VL-8B-Instruct \
    "$@"
