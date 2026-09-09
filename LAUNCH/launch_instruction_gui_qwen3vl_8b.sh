#!/usr/bin/env bash
# Stronger perception configuration for cluttered object-goal navigation.
# Uses the locally cached Qwen3-VL-8B-Instruct checkpoint with the same
# NavDP, depth guard, multi-view appearance lock, and focused visual context
# as the 2B launcher.  It is intentionally a separate entry point: 8B needs
# substantially more VRAM, while existing Qwen3-VL-2B workflows stay intact.
set -uo pipefail
cd "$(dirname "$0")"

exec ./launch_instruction_gui.sh "$@" \
    --qwen-model-id Qwen/Qwen3-VL-8B-Instruct \
    --qwen-fp16 \
    --qwen-appearance-lock \
    --stop-distance 0.5
