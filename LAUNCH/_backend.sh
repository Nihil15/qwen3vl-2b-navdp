#!/usr/bin/env bash
# ============================================================
#  Shared --rover / --hiwonder backend selection + Pi bring-up.
#  Sourced by the other LAUNCH/*.sh scripts -- not meant to be run directly.
#
#  Usage in a launch script:
#    source "$(dirname "$0")/_backend.sh"
#    backend_parse_args "$@"
#    set -- "${BACKEND_ARGS[@]}"          # positional PI_IP + --rover/--hiwonder stripped
#    backend_bringup camera               # or: backend_bringup nocamera
#    ...
#    exec python -u -m nav_pipeline.xxx --pi-ip "$PI_IP" --fov "$BACKEND_FOV" ...
#
#  Defaults to --rover when neither flag is given, so every existing call
#  site (nobody passes --rover/--hiwonder today) is unaffected.
#
#  --hiwonder notes:
#    - Pi bring-up is landerpi/deploy_bridge.sh (bridge.py inside the
#      existing armpi_pro container), not systemd services -- see
#      landerpi/README.md.
#    - $BACKEND_FOV/$BACKEND_FOOTPRINT_LENGTH/$BACKEND_FOOTPRINT_WIDTH are
#      this robot's real measured/spec values (see landerpi/README.md).
#    - Steering-shape constants tuned specifically for the OLD rover's
#      camera/motor response (search-angular, servo-ramp-deg, the 1.2
#      max-angular normalization) are NOT re-validated for this chassis --
#      each launch script still passes its own tuned values verbatim, and
#      backend_bringup prints an explicit warning for --hiwonder callers
#      that pass them through unchanged, since they're a carried-over
#      starting point, not a measurement (see landerpi/README.md's "Known
#      caveats").
# ============================================================

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
info() { echo -e "${CYAN}  → $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $*${NC}"; }
err()  { echo -e "${RED}  ✗ $*${NC}"; }

backend_parse_args() {
    BACKEND=""
    for arg in "$@"; do
        case "$arg" in
            --rover) BACKEND=rover ;;
            --hiwonder) BACKEND=hiwonder ;;
        esac
    done
    [[ -z "$BACKEND" ]] && BACKEND=rover

    BACKEND_ARGS=()
    for arg in "$@"; do
        [[ "$arg" == "--rover" || "$arg" == "--hiwonder" ]] || BACKEND_ARGS+=("$arg")
    done

    if [[ "$BACKEND" == "rover" ]]; then
        BACKEND_DEFAULT_IP=10.86.180.125   # Pi IP churns -- pass the real IP as the first arg
        BACKEND_PI_PASS_DEFAULT=""          # scrubbed for public repo -- export PI_PASS=... before running
        # Camera is the Intel RealSense D435i (scripts/realsense_only_
        # bringup.launch.py) as of 2026-08-13 -- the earlier 2026-08-11
        # attempt failed on a USB2-only cable link-negotiated at 480M in a
        # USB3 port; re-plugged with a proper cable it now negotiates full
        # USB3 (5000M) and image_raw is solid at 15fps.
        #
        # This is now only a BOOTSTRAP fallback: nav_pipeline/pipeline.py
        # prefers real intrinsics from /image_raw/camera_info (Zenoh
        # CAMERA_INFO_KEYS, see zenoh_node.py's parse_camera_info) the
        # instant a fresh one arrives, and only falls back to this
        # FOV-derived approximation before the first camera_info shows up
        # or if that subscription ever goes stale. 55.5 = 2*atan(320/fx)
        # with fx=607.79, measured directly from this D435i's own
        # camera_info at 640x480 (2026-08-13) -- re-measure the same way if
        # this ever drifts (e.g. resolution change).
        BACKEND_FOV=55.5
        # Matches the ESP32 firmware's own angular normalization (see
        # launch_rover.sh's comment) -- real, measured tuning for this bot.
        BACKEND_MAX_ANGULAR=1.2
        BACKEND_ANGULAR_SLEW_MAX=0.10   # pipeline.py's own default, unchanged
        # Set to 0.174 on 2026-08-14 at explicit user request -- this is the
        # theoretical minimum, not a safety-margined value. The ESP32
        # firmware (esp32/rover_6wd_complete.ino) hard-zeros (strict `<`,
        # line 437) any per-wheel target below VEL_DEADBAND_MS=0.03 m/s, and
        # SEARCH's pure rotation converts to a per-wheel target of
        # angular_z * TRACK_WIDTH_M/2 (0.345m track width). 0.174 * 0.1725 =
        # 0.030015 m/s -- clears the deadband by only ~0.000015 m/s
        # (~0.05%), i.e. essentially zero margin against float rounding,
        # encoder noise, or friction variance. If SEARCH doesn't move (or
        # moves inconsistently) on this rover, this near-zero margin is the
        # first thing to suspect -- 0.18 (0.031 m/s/wheel, ~3.5% margin) is
        # the last value actually validated as reliably working; 0.15/0.13/
        # 0.14 are all confirmed NOT to work (2026-08-12). See
        # LAUNCH/launch_rover.sh's comment for the full derivation.
        BACKEND_SEARCH_ANGULAR=0.174
    else
        BACKEND_DEFAULT_IP=10.47.234.228
        BACKEND_PI_PASS_DEFAULT=""          # scrubbed for public repo -- export PI_PASS=... before running
        BACKEND_FOV=64.6   # from /usb_cam/camera_info: fx=507.2, width=640 -> 2*atan(320/507.2)
        BACKEND_FOOTPRINT_LENGTH=0.298
        BACKEND_FOOTPRINT_WIDTH=0.256
        # 1.2 was only ever the ROVER's ESP32-normalization value, carried
        # over unvalidated (see warning below) -- reduced 2026-08-07 to the
        # same conservative cap already live-validated on this exact chassis
        # via launch_bot.sh/home_gui.py (README: "Velocity caps start at the
        # same conservative real-rover defaults"). Bump this specifically
        # (not the rover's) if 0.5 turns out too slow once re-tuned properly.
        BACKEND_MAX_ANGULAR=0.5
        # Unchanged (still the old rover's carried-over 0.13, see
        # launch_rover.sh's comment) -- the 0.18 rover fix above is derived
        # from this exact ESP32 firmware's VEL_DEADBAND_MS/TRACK_WIDTH_M,
        # which don't apply to this chassis's own bridge/firmware. Not
        # re-derived or validated for the Hiwonder; check landerpi's actual velocity
        # deadband before bumping this one too.
        BACKEND_SEARCH_ANGULAR=0.13
        # Reduced 2026-08-10 from pipeline.py's 0.10 default: a fresh
        # SEARCH->TRACK correction (bearing error is large right after a
        # glimpse at frame edge) commands a real turn, and that turn's own
        # viewpoint change was breaking the very re-identification continuity
        # (IoU + DINOv2 appearance similarity, see PipelineConfig's
        # appearance_min_similarity/appearance_iou_override comments) needed
        # to confirm the lock -- symptom: bot "gets excited" toward the
        # object, swings, loses it mid-swing, reverts to SEARCH, repeats.
        # Halving the slew rate spreads the same correction over ~2x the
        # ticks, so each tick's viewpoint change is smaller and more likely
        # to stay within the IoU-override/appearance-similarity bounds.
        BACKEND_ANGULAR_SLEW_MAX=0.05
    fi

    if [[ "${BACKEND_ARGS[0]:-}" == -* || -z "${BACKEND_ARGS[0]:-}" ]]; then
        PI_IP=$BACKEND_DEFAULT_IP
    else
        PI_IP=${BACKEND_ARGS[0]}
        BACKEND_ARGS=("${BACKEND_ARGS[@]:1}")
    fi
    PI_USER=pi
    PI_PASS=${PI_PASS:-$BACKEND_PI_PASS_DEFAULT}
    SSH="sshpass -p $PI_PASS ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new $PI_USER@$PI_IP"
}

# backend_bringup camera|nocamera
backend_bringup() {
    local need_camera=$1

    info "[$BACKEND] Pinging Pi at $PI_IP ..."
    ping -c1 -W2 "$PI_IP" >/dev/null || { warn "Pi unreachable — is it ON and on this network?"; exit 1; }
    $SSH 'echo ssh_ok' | grep -q ssh_ok || { warn "SSH failed (user=$PI_USER pass=\$PI_PASS)"; exit 1; }
    ok "Pi reachable"

    if [[ "$BACKEND" == "rover" ]]; then
        local services n
        if [[ "$need_camera" == "camera" ]]; then
            services="rover-camera rover-agent rover-zenoh"; n=3
        else
            services="rover-agent rover-zenoh"; n=2
        fi
        info "Restarting rover services on Pi..."
        $SSH "echo $PI_PASS | sudo -S systemctl restart $services 2>/dev/null; sleep 4; systemctl is-active $services" \
            | grep -c active | grep -q "$n" && ok "services active: $services" \
            || warn "services not all active — run: bash scripts/pi_install_services.sh on the Pi"

        if [[ "$need_camera" == "camera" ]]; then
            info "Waiting for camera topic (up to 25 s)..."
            local cam_ok=false
            for i in $(seq 1 25); do
                sleep 1
                $SSH 'bash -lc "source /opt/ros/humble/setup.bash; ros2 topic list 2>/dev/null"' 2>/dev/null \
                    | grep -q "/image_raw" && { cam_ok=true; ok "camera live [${i}s]"; break; }
            done
            $cam_ok || warn "camera topic missing — check: ssh $PI_USER@$PI_IP 'journalctl -u rover-camera -n 20'"
        fi

        info "Checking ESP32 heartbeat (/rover/rpm, up to 25 s)..."
        if $SSH 'bash -lc "source /opt/ros/humble/setup.bash; timeout 25 ros2 topic echo /rover/rpm --once 2>/dev/null"' 2>/dev/null \
            | grep -q "layout\|data"; then
            ok "ESP32 alive (/rover/rpm publishing)"
        else
            warn "No /rover/rpm — check: ssh $PI_USER@$PI_IP 'journalctl -u rover-agent -n 20'"
        fi
    else
        info "Deploying/restarting the Nav_new bridge on the LanderPi..."
        PI_PASS=$PI_PASS "$(dirname "${BASH_SOURCE[0]}")/../landerpi/deploy_bridge.sh" "$PI_IP" \
            || { warn "bridge deploy failed — see landerpi/README.md"; exit 1; }
        warn "steering-shape constants below (search-angular/servo-ramp-deg/max-angular normalization) were tuned for the OLD rover's camera/motor response, NOT re-validated on this chassis -- see landerpi/README.md"
    fi
}
