# Shaktiman Vanish

Point your index finger up. Your silhouette becomes a spinning whirlwind and flies out of frame.
Video is processed locally; nothing is recorded or uploaded.

## Start

1. Install [Python 3.10+](https://www.python.org/downloads/), then [download and extract this repo](https://github.com/imrishabh18/shaktiman-vanish/archive/refs/heads/main.zip).
2. Double-click **`start.command`** on Mac or **`start.bat`** on Windows. First launch installs dependencies and downloads the models.
3. Keep the camera still. **Step completely out of view** until the preview says **BACKGROUND READY — STEP BACK IN**.
4. Return, point one index finger upward with the other fingers curled, and hold for half a second. Either hand works.

**Controls:** `R` reappear · `V` test the effect · `Space` recapture background · `F` fullscreen · `Q` quit.

## Music

Put **`music.wav`** or **`music.mp3`** beside the launchers. A three-second cue plays once when the whirlwind starts. `R`, `Space`, and quitting stop it. Custom music stays local.
Use `--no-music` to mute it, or `--music-volume 0.5` to set the volume (0–1). For another file: `--music /path/to/clip.mp3 --music-start 4`.

## Google Meet, Zoom, and other calls

1. Install [OBS Studio](https://obsproject.com/download) (30+ on Mac). Open OBS, click **Start Virtual Camera**, allow the setup prompts, then **Stop Virtual Camera** and quit OBS.
2. Close the regular preview and run **`start-meeting.command`** on Mac or **`start-meeting.bat`** on Windows.
3. Select **OBS Virtual Camera** in your meeting app's camera settings. Keep Shaktiman running and OBS closed. Use your normal microphone.

**Share music + voice on Mac:** install [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole#installation-instructions), run `./start-meeting.command --meeting-audio`, and select **BlackHole 2ch** as the call's microphone. Keep the call's speaker output on your headphones/speakers. If music is filtered out, disable noise cancellation or enable Zoom's original sound.

## Quick fixes

- **Still visible after spinning?** Press `Space`, leave the frame completely, and wait for the green **BACKGROUND READY** message.
- **Wrong camera?** Mac prefers the built-in webcam. To choose another, run `./start.command --camera 1` (Windows: `start.bat --camera 1`).
- **Camera blocked?** Allow Terminal/Python camera access in system privacy settings, then reopen the app.
- **Mac won't open the launcher?** Run `bash start.command` from this folder in Terminal.
- **Virtual camera missing?** Finish OBS's camera setup, then fully quit and reopen your meeting app/browser.

Tests: `python -m unittest discover -s tests` (after setup, use the Python inside `.venv`).

Based on [Nicovic17/vanish_webcam](https://github.com/Nicovic17/vanish_webcam). Hand tracking and segmentation use [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide).
