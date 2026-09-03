#!/usr/bin/env bash
# ============================================================
#  Nav_new — Qwen instruction + NavDP GUI launcher (real rover / LanderPi)
#  Run on the GPU machine:
#    ./launch_instruction_gui.sh [--rover|--hiwonder] [PI_IP] [gui args...]
#
#  Brings up the Pi (camera + ESP32 micro-ROS + Zenoh bridge, see
#  LAUNCH/_backend.sh), then starts nav_pipeline.instruction_gui: camera
#  view, top-down NavDP trajectory plot, and a free-text INSTRUCTION box
#  (Send/STOP) -- no DINO target phrase, no manual-drive pad. Qwen2.5-VL
#  grounds the instruction to a pixel goal every tick (throttled), NavDP
#  samples/follows a trajectory toward it -- see nav_pipeline/
#  qwen_pixel_goal.py. Defaults to --rover (same as launch_rover.sh).
#
#  Examples:
#    ./launch_instruction_gui.sh                          # default rover, default Pi IP
#    ./launch_instruction_gui.sh 172.22.217.125 --instruction "go to the door"
#    ./launch_instruction_gui.sh --hiwonder --instruction "go to the door"
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

source LAUNCH/_backend.sh
backend_parse_args "$@"
set -- "${BACKEND_ARGS[@]}"

backend_bringup camera

set +u
source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav
set -u
export HF_HOME=${HF_HOME:-/mnt/bigdisk/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/bigdisk/hf_cache/transformers}

pkill -f "nav_pipeline.instruction_gui" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.isaac_gui" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.zenoh_node" 2>/dev/null && sleep 1

# Same real-rover caps/fov/ramp/slew reasoning as launch_rover.sh -- see
# that script's comments for the ESP32 deadband/servo-ramp history. This
# GUI has no --search-angular (no DINO SEARCH state in instruction mode).
EXTRA_ARGS=()
if [[ "$BACKEND" == "hiwonder" ]]; then
    EXTRA_ARGS+=(--footprint-length "$BACKEND_FOOTPRINT_LENGTH" --footprint-width "$BACKEND_FOOTPRINT_WIDTH")
else
    # ESP32 6WD rover only -- see pipeline.py's clear_wheel_deadband.
    EXTRA_ARGS+=(--wheel-deadband-correction)
fi
info "Starting Nav_new instruction GUI [$BACKEND] (pi-ip=$PI_IP, caps 0.15 m/s / $BACKEND_MAX_ANGULAR rad/s, fov $BACKEND_FOV, ramp 70deg, slew $BACKEND_ANGULAR_SLEW_MAX rad/s/tick)..."
exec python -u -m nav_pipeline.instruction_gui \
    --pi-ip "$PI_IP" \
    --max-linear 0.15 --max-angular "$BACKEND_MAX_ANGULAR" --fov "$BACKEND_FOV" \
    --servo-ramp-deg 70 --angular-slew-max "$BACKEND_ANGULAR_SLEW_MAX" \
    --compressed-only \
    "${EXTRA_ARGS[@]}" \
    "$@"
