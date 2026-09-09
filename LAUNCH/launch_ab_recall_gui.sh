#!/usr/bin/env bash
# ============================================================
#  Manual-instruction GUI for the HM3D A/B recall test.
#
#    ./LAUNCH/launch_ab_recall_gui.sh [--scene wcojb4TFT35] [--cam-height 0.9]
#
#  Opens a window: walk the scanned HM3D house (arrow keys / WASD), read off
#  the scene's real semantic categories, then type the two instructions --
#     A = initial navigation instruction (leg 1)
#     B = recall instruction            (leg 2)
#  -- and hit "Run A/B episode from this pose". That shells out to
#  run_ab_position_recall_test.py with --instruction-a / --query-b set to
#  your text and --start-xy set to wherever you are standing.
#
#  Needs an X display and the `habitat` conda env (habitat_sim + tkinter +
#  pillow). Override CONDA_SH / GUI_ENV per machine.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
GUI_ENV="${GUI_ENV:-habitat}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$GUI_ENV"
else
  echo "[launch_ab_recall_gui] no conda.sh at $CONDA_SH -- set CONDA_SH; using current python" >&2
fi

exec python -u ab_recall_gui.py "$@"
