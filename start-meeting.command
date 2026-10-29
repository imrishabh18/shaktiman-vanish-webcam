#!/bin/bash
# Send Shaktimaan video directly to the installed OBS virtual camera.
exec "$(dirname "$0")/start.command" --virtual-camera --virtual-backend obs --width 1280 --height 720 --fps 30 "$@"
