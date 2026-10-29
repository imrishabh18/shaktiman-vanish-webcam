#!/bin/bash
# Double-click on macOS, or run this script from a terminal with extra options.
set -u
cd "$(dirname "$0")" || exit 1

fail() {
    printf '\n[Shaktimaan] %s\n' "$1" >&2
    printf 'Press Return to close this window. '
    read -r _
    exit 1
}

export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export PIP_DISABLE_PIP_VERSION_CHECK=1
VENV_PY=".venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    command -v python3 >/dev/null 2>&1 || fail "Install Python 3.10 or newer from python.org, then try again."
    python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' || fail "Python 3.10 or newer is required. Install it from python.org, then try again."
    printf '[Shaktimaan] Creating a local Python environment...\n'
    python3 -m venv .venv || fail "Could not create .venv. Check that this folder is writable and Python includes venv."
fi

"$VENV_PY" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' || fail "The existing .venv needs Python 3.10 or newer. Rename .venv and run this launcher again."

if ! cmp -s requirements.txt .venv/.vanish-requirements ||
   ! "$VENV_PY" -c 'import cv2, mediapipe, numpy, pyvirtualcam, AVFoundation' >/dev/null 2>&1; then
    printf '[Shaktimaan] Installing dependencies. Internet is required for this setup.\n'
    "$VENV_PY" -m pip install -r requirements.txt || fail "Dependency setup failed. Check the error above and your internet connection, then rerun this launcher."
    cp requirements.txt .venv/.vanish-requirements || fail "Could not save the setup marker in .venv. Check folder permissions."
fi

printf '[Shaktimaan] Starting webcam. Step fully out of view and wait for BACKGROUND READY.\n'
"$VENV_PY" -m shaktiman "$@" || fail "The webcam app stopped with an error. Read the message above; see README.md for help."
