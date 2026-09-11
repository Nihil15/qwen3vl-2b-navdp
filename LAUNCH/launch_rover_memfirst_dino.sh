#!/usr/bin/env bash
# ============================================================
#  Nav_new -- headless MEMORY-FIRST + MULTI-CANDIDATE-VERIFY multileg runner
#    ./LAUNCH/launch_rover_memfirst_dino.sh [--rover|--hiwonder] [PI_IP] <plan args...>
#
#  Same Pi bring-up / conda env / real-rover caps as
#  LAUNCH/launch_rover_multileg.sh, but runs run_rover_memfirst_dino.py
#  instead of run_rover_multileg.py (neither the stock runner nor its
#  launchers are modified).
#
#  Behaviour changes, both ON by default (disable with the --no-* flags):
#    --memory-first-authoritative   object leg checks goal memory first; a hit
#                                   drives to the remembered point and counts
#                                   as arrival -- no fresh DINO search.
#    --dino-candidate-verify        >=2 comparable DINO boxes -> Qwen picks the
#                                   one the instruction means, seeds the lock.
#
#  Examples:
#    ./LAUNCH/launch_rover_memfirst_dino.sh 10.146.234.125 \
#        --subtask "object:red bucket" --subtask turn:-90 --subtask "object:red bucket" --yes
#
#    ./LAUNCH/launch_rover_memfirst_dino.sh --instruction \
#        "go to the air cooler, then turn around, then go back to the air cooler" --yes
#
#    # opt back out of one behaviour:
#    ./LAUNCH/launch_rover_memfirst_dino.sh 10.146.234.125 --subtask "object:chair" \
#        --no-memory-first-authoritative --yes
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
[[ -n "${HF_HUB_OFFLINE:-}" ]] && export HF_HUB_OFFLINE

pkill -f "run_rover_multileg.py" 2>/dev/null && sleep 1
pkill -f "run_rover_memfirst_dino.py" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.instruction_gui" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.zenoh_node" 2>/dev/null && sleep 1

if [[ "$BACKEND" == "hiwonder" ]]; then
    warn "run_rover_memfirst_dino.py's turn/move deadband floors are ESP32-tuned -- re-check on this chassis"
fi

info "Starting memory-first + candidate-verify runner [$BACKEND] (pi-ip=$PI_IP, caps 0.15 m/s / $BACKEND_MAX_ANGULAR rad/s, fov $BACKEND_FOV, slew $BACKEND_ANGULAR_SLEW_MAX rad/s/tick)..."
exec python -u run_rover_memfirst_dino.py \
    --pi-ip "$PI_IP" \
    --max-linear 0.15 --max-angular "$BACKEND_MAX_ANGULAR" --fov "$BACKEND_FOV" \
    --angular-slew-max "$BACKEND_ANGULAR_SLEW_MAX" \
    "$@"
