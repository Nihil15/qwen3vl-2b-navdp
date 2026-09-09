#!/usr/bin/env bash
# ============================================================
#  Nav_new — offline Fact3R-Map goal -> NavDP "GOTO" rollout
#  Run on the GPU machine (no Pi, no Zenoh, no rover):
#    ./launch_offline_fact3r_nav.sh --frames <video|dir> [driver args...]
#
#  Drives a recorded RGB(-D) walk through DinoNavDPPipeline with the target
#  supplied every tick by nav_pipeline/fact3r_goal_source.py as
#  `external_goal` -- the pipeline's blind navigate-back path (state
#  "GOTO"). Fact3R-Map (../MASt3R-SLAM/fact3r-map) was built earlier from a
#  walk through the scene; a text query located an entity in it; the rover
#  now drives back to that point. See run_offline_fact3r_nav.py's docstring
#  and FACT3R_INTEGRATION.md.
#
#  This launcher only fixes the interpreter/env + points the NavDP and
#  Depth-Anything checkpoints at the repo's `checkpoints/` (symlink them in
#  first, or set NAVDP_CKPT_DIR-style paths -- see FACT3R_INTEGRATION.md).
#  Every run_offline_fact3r_nav.py flag passes straight through.
#
#  Examples:
#    # goal request already produced by fact3r-map/resolve_semantic_goal.py
#    ./launch_offline_fact3r_nav.sh --frames walk/ \
#        --goal-request goal_request.json --align identity \
#        --integrate-cmd --out logs/fact3r_nav/run1
#
#    # resolve the query here (shells out to the SigLIP env)
#    ./launch_offline_fact3r_nav.sh --frames walk.mp4 \
#        --fact3r-map-root ../MASt3R-SLAM/fact3r-map \
#        --semantic-map ../MASt3R-SLAM/fact3r-map/logs/fact3r_video/room-walk/room-walk_semantic.json \
#        --query "the black chair" --siglip-env SAM2 --align anchor \
#        --integrate-cmd --out logs/fact3r_nav/run1
#
#    # no map -- just exercise the GOTO loop toward a fixed odom point
#    ./launch_offline_fact3r_nav.sh --frames walk/ --goal-odom-xy "4 -1.5" \
#        --out logs/fact3r_nav/smoke
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

# NAV_ENV needs torch + cv2 + transformers>=4.57 + the Depth-Anything-V2 deps
# that nav_pipeline/ imports. The instruction-GUI launchers use `internnav`;
# override NAV_ENV / CONDA_SH for your machine.
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
NAV_ENV="${NAV_ENV:-internnav}"

if [[ -f "$CONDA_SH" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$NAV_ENV"
    set -u
else
    echo "[launch_offline_fact3r_nav] no conda.sh at $CONDA_SH -- set CONDA_SH; using current python" >&2
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

if [[ ! -e checkpoints/navdp_extracted.pth && ! -e navdp-cross-modal.ckpt ]]; then
    echo "[launch_offline_fact3r_nav] WARNING: no NavDP checkpoint found." >&2
    echo "  Expected navdp-cross-modal.ckpt (--policy crossmodal, default) or" >&2
    echo "  checkpoints/navdp_extracted.pth (--policy extracted). See FACT3R_INTEGRATION.md." >&2
fi

pkill -f "run_offline_fact3r_nav" 2>/dev/null && sleep 1
true

exec python -u run_offline_fact3r_nav.py "$@"
