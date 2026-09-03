#!/usr/bin/env bash
# ============================================================
#  Nav_new — Qwen3-VL-2B (fp16) instruction + NavDP GUI
#  Run on the GPU machine:
#    ./launch_instruction_gui_qwen3vl.sh [--rover|--hiwonder] [PI_IP] [gui args...]
#
#  Identical to LAUNCH/launch_instruction_gui.sh (Pi bring-up, conda env,
#  real-rover caps/fov/ramp/slew, the free-text INSTRUCTION box, Qwen ->
#  pixel goal -> depth -> NavDP trajectory) EXCEPT the instruction grounder
#  is Qwen/Qwen3-VL-2B-Instruct loaded in fp16 (~5GB VRAM, no bitsandbytes)
#  instead of the 4-bit Qwen2.5-VL-7B default. The model-class-agnostic
#  loader in nav_pipeline/qwen_pixel_goal.py (AutoModelForImageTextToText)
#  picks Qwen3VLForConditionalGeneration from the checkpoint config.
#
#  First run downloads Qwen3-VL-2B (~4-5GB) into $HF_HOME.
#
#  Examples:
#    ./launch_instruction_gui_qwen3vl.sh 10.76.200.125
#    ./launch_instruction_gui_qwen3vl.sh 10.76.200.125 --instruction "go to the door"
#    ./launch_instruction_gui_qwen3vl.sh --hiwonder --instruction "..."
# ============================================================
set -uo pipefail
cd "$(dirname "$0")"

# Thin wrapper: launch_instruction_gui.sh forwards its trailing "$@"
# straight to `python -m nav_pipeline.instruction_gui`, so appending the
# grounder flags here is all that's needed. Backend selection + positional
# PI_IP are still parsed from the leading args by _backend.sh.
exec ./launch_instruction_gui.sh "$@" \
    --qwen-model-id Qwen/Qwen3-VL-2B-Instruct \
    --qwen-fp16 \
    --stop-distance 0.5
