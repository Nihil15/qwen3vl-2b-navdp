#!/usr/bin/env bash
# ============================================================
#  Nav_new — headless multi-subtask real-rover runner
#  Run on the GPU machine:
#    ./LAUNCH/launch_rover_multileg.sh [--rover|--hiwonder] [PI_IP] <plan args...>
#
#  Same Pi bring-up / conda env / real-rover caps as
#  LAUNCH/launch_instruction_gui.sh, but instead of the Tk instruction GUI
#  it runs run_rover_multileg.py: an ordered sequence of MIXED subtasks --
#    move:<m>      straight-line drive, closed-loop on encoder+IMU odometry
#    turn:<deg>    in-place rotation, closed-loop on IMU heading (neg = right)
#    object:<text> Qwen3-VL-2B pixel-goal -> depth -> NavDP, driven to STOP
#  -- with a 1 s rest between each, plus an optional 1 m out-and-back
#  odometry self-check before the run.  Defaults to --rover (ESP32).
#
#  Examples:
#    ./LAUNCH/launch_rover_multileg.sh 10.86.180.125 \
#        --subtask move:2.5 --subtask turn:-90 --subtask "object:the air cooler"
#
#    ./LAUNCH/launch_rover_multileg.sh --instruction \
#        "go straight up to 2.5 meter, turn right, go to the air cooler"
#
#    # skip the confirm prompt / the self-check:
#    ./LAUNCH/launch_rover_multileg.sh --subtask move:2.5 --yes --no-odom-selfcheck
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

source LAUNCH/_backend.sh
backend_parse_args "$@"
set -- "${BACKEND_ARGS[@]}"

backend_bringup camera

# Conda base + env differ per machine: blackwell (this repo's current host)
# is /home/gpu/miniconda3 + `navdp` (transformers 5.x, zenoh, torch cu128);
# the older i3d box was /home/i3d/exit + `internnav`. Auto-detect, override
# with MULTILEG_CONDA_SH / MULTILEG_ENV.
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
# NOT forced to 1 by default -- live-hit 2026-09-09 (see
# launch_rover_multileg_gui.sh's identical comment): a stale huggingface_hub
# negative-cache marker made ClipVerifier's from_pretrained fail outright
# under offline mode even though the actual file is present locally. Only
# export it if the caller already had it set in their own environment.
[[ -n "${HF_HUB_OFFLINE:-}" ]] && export HF_HUB_OFFLINE

pkill -f "run_rover_multileg.py" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.instruction_gui" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.zenoh_node" 2>/dev/null && sleep 1

# The metric-controller floors/gains in run_rover_multileg.py (--turn-w-floor
# 0.20, --move-v-floor 0.05) are derived from the ESP32 rover's
# VEL_DEADBAND_MS / TRACK_WIDTH_M (see LAUNCH/_backend.sh's BACKEND_SEARCH_
# ANGULAR comment). They are NOT validated for the Hiwonder chassis.
if [[ "$BACKEND" == "hiwonder" ]]; then
    warn "run_rover_multileg.py's turn/move deadband floors are ESP32-tuned -- re-check on this chassis"
fi

info "Starting multi-subtask runner [$BACKEND] (pi-ip=$PI_IP, caps 0.15 m/s / $BACKEND_MAX_ANGULAR rad/s, fov $BACKEND_FOV, slew $BACKEND_ANGULAR_SLEW_MAX rad/s/tick)..."
exec python -u run_rover_multileg.py \
    --pi-ip "$PI_IP" \
    --max-linear 0.15 --max-angular "$BACKEND_MAX_ANGULAR" --fov "$BACKEND_FOV" \
    --angular-slew-max "$BACKEND_ANGULAR_SLEW_MAX" \
    "$@"
