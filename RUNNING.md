# Running the pipeline

All commands assume you're inside the devcontainer (`Dev Containers: Reopen in
Container`), working directory `/workspace` (repo root). Build once:

```bash
make clean && make        # → build/face_recon
```

The pretrained landmark models (LBF 68-point + YuNet face detector) are baked
into the devcontainer image at `/opt/models/`; the app also checks `./models`
first, so nothing to download on a normal container run.

---

## 1. Every mode at a glance

The only dataset is **Biwi** (Kinect RGB-D). Everything is selected with
`--mode`; there is no `--dataset` flag. The two primary modes reconstruct a
**video sequence**; the geometry modes operate on a single frame.

| `--mode` | What it does | Depth | Landmarks |
|---|---|---|---|
| **`rgb`** | Video: personalise frame 0, then track pose+expression. Landmarks + jaw contour + photometric (lighting/albedo). | no | auto (YuNet) |
| **`rgbd`** | Same, **plus the metric Kinect depth ICP term** (the full fit). | yes | auto (YuNet) |
| **`live-cpu`** | Realtime from the Mac camera, CPU tracker (§4b). **HOST only.** | no | auto (YuNet) |
| **`live-gpu`** | Realtime on the GPU — **stub**, under development by the GPU team. | — | — |
| `dense` | Single frame: depth-only ICP (geometry, no photometric). | yes | none |
| `full` | Single frame: sparse landmark fit → dense ICP. | yes | file (Stage 1) |

**auto (YuNet)** = landmarks are detected in-process by the C++
`LandmarkDetector` (YuNet CNN face box + pose-robust 5 points + LBF jaw contour)
— no Python step, no landmark file. Switch backend with `--detector` (§2).
`full`'s Stage-1 sparse fit still reads a pre-generated `landmarks_XXXXX.txt`
(see §2).

---

## 2. Landmarks

Three landmark backends, selected with **`--detector yunet|lbf|mediapipe`**
(default `yunet`):

| `--detector` | Source | Points | Notes |
|---|---|---|---|
| `yunet` (default) | in-process C++ | 5 pose-robust interior (YuNet CNN) + 8 LBF jaw contour | best pose robustness, no pre-pass |
| `lbf` | in-process C++ | 9 interior + 8 jaw, all from Haar+LBF | the legacy detector; frontal-biased |
| `mediapipe` | **offline pre-pass files** | 8 interior (incl. upper/lower lip + chin) + 8 jaw contour | densest; MediaPipe is Bazel-built so it runs as a Python pre-pass, not in the binary |

For `mediapipe`, run the pre-pass once per sequence/image (writes
`landmarks_mp_XXXXX.txt` next to the frames; mediapipe is preinstalled in the
devcontainer image):

```bash
python3 python/gen_landmarks_mediapipe.py --biwi-dir data/BK-1/01 [--max N]
python3 python/gen_landmarks_mediapipe.py --iphone-dir data/iphone/default
```

`yunet`/`lbf` need **no landmark file and no Python step** for: iPhone
(`sparse`/`photometric`), Biwi `photometric` (incl. `--depth`), and `video`.

Only two modes still read the old pre-generated `landmarks_XXXXX.txt` file:
`--mode sparse --dataset biwi` and `--mode full --dataset biwi` (its Stage-1
sparse fit). For those, generate the file once with the in-repo Python tool
(one line per point: `bfm_vertex_index u v`; `-1` = jaw-contour point):

```bash
# Biwi frame — filename must match the frame NUMBER in the folder, e.g. frame_00003
python3 python/gen_landmarks.py --set small --contour \
    data/BK-1/01/frame_00003_rgb.png \
    data/BK-1/01/landmarks_00003.txt
```

To find which frame numbers exist in a Biwi subject folder:

```bash
ls data/BK-1/01/*_rgb.png | sed -E 's/.*frame_([0-9]+)_rgb.*/\1/' | sort -n
```

---

## 3. The full RGB-D reconstruction (`rgbd`)

The complete analysis-by-synthesis pipeline on a Biwi sequence, per frame:
**sparse landmarks → jaw-contour + expression + depth ICP (jointly) →
photometric (lighting + albedo)**, with identity + albedo personalised on
frame 0 and frozen for tracking. Landmarks are detected in-process (YuNet) —
**no landmark file needed**.

```bash
# subject 01, first 200 frames, with the depth term
./build/face_recon --mode rgbd --biwi-dir data/BK-1/01 --frames 200 --sparse-reg 30
```

**Outputs** land in `data/out/biwi_video_full/`: `tracking.mp4` plus per-frame
3-panel PNGs (overlay | reconstruction @ pose | reconstruction frontal) in
`frames/`.

Use `--mode rgb` for the same pipeline **without** the depth term (RGB-only);
outputs go to `data/out/biwi_video_rgb/`.

---

## 4. Other common runs

```bash
# RGB-only video (no depth)
./build/face_recon --mode rgb --biwi-dir data/BK-1/01 --frames 200 --sparse-reg 30

# RGB-D video with the MediaPipe detector (run the pre-pass first — see §2)
./build/face_recon --mode rgbd --biwi-dir data/BK-1/01 --frames 200 \
    --sparse-reg 30 --detector mediapipe

# Depth-only ICP on a single frame (geometry, no photometric)
./build/face_recon --mode dense --biwi-dir data/BK-1/01 --icp-iters 30

# Single-frame sparse landmark init → dense ICP (Stage 1 needs a file — see §2)
./build/face_recon --mode full --biwi-dir data/BK-1/01 --icp-iters 30

# Any mode with a different detector (see §2) or per-stage timings:
./build/face_recon --mode photometric --dataset biwi --biwi-dir data/BK-1/01 \
    --depth --sparse-reg 30 --detector mediapipe --timers

# Focal-estimation test on the iPhone photo (known fx=1930): replace K with a
```

---

## 4b. Realtime camera mode (`--mode live-cpu`) — HOST ONLY

Live tracking from the MacBook camera: personalises on the first detected face
(identity + albedo + lighting), then tracks pose + expression per frame with a
live overlay. **Must run on the host** (Docker has no camera). From the repo
root in a normal terminal:

```bash
# build for the host (Homebrew paths), then run
CPATH=/opt/homebrew/include LIBRARY_PATH=/opt/homebrew/lib make
./build/face_recon --mode live-cpu --sparse-reg 30

# with the pyramid photometric pose refinement (better silhouette, ~9 fps)
./build/face_recon --mode live-cpu --sparse-reg 30 --photo-refine

# pick a specific camera (0 is sometimes the iPhone Continuity Camera)
./build/face_recon --mode live-cpu --sparse-reg 30 --camera 1

# headless smoke test without a camera (Biwi frames as a fake camera)
./build/face_recon --mode live-cpu --live-source data/BK-1/01/frame_00003_rgb.png \
    --live-frames 12 --live-nodisplay --sparse-reg 30 --timers
```

(`--mode live-gpu` is a stub for the GPU team — see PLAN_REALTIME.md.)

- Keys: `q` quit · `p` re-personalise · `s` snapshot → `data/out/live/`.
- HUD shows fps and the current focal. Intrinsics start from a 60°-HFOV guess;
  `--optimize-focal` additionally solves the focal during personalisation
  (experimental — single-frame estimation is biased long, see PLAN_REALTIME.md).
- `--photo-refine` is auto-rejected on frames where it would worsen the
  landmark reprojection (safeguard).

**Black screen / camera troubleshooting.** The app now warms each camera up
(~2.5 s), skips devices that only deliver black frames (a plugged-in iPhone
Continuity Camera at index 0 is the usual culprit), retries with a fresh open
(a stream opened *before* the permission grant stays black until re-opened),
and falls back over indices 0–2 automatically. If it still reports no usable
frames:
1. System Settings → Privacy & Security → **Camera** → enable your terminal
   app, then **re-run** (the grant only applies to new sessions).
2. Try `--camera 1` (or 2) explicitly.
3. Make sure no other app (Zoom/FaceTime) is holding the camera.

Measured on an M2 (640 px): detect ~9 ms, track ~15 ms, render ~14 ms →
~25 fps steady-state; ~9 fps with `--photo-refine`.

---

## 5. CLI flag reference

| Flag | Meaning | Default |
|---|---|---|
| `--mode <rgb\|rgbd\|live-cpu\|live-gpu\|dense\|full>` | Which pipeline to run | `rgb` |
| `--sparse-reg <λ>` | Identity/expression regulariser weight | `100.0` |
| `--biwi-seq <NN>` | Shortcut for `data/biwi/NN` | — |
| `--biwi-dir <path>` | Biwi subject folder (e.g. `data/BK-1/01`) | `data/biwi/01` |
| `--frames <n>` | Number of frames for `rgb`/`rgbd` | `30` |
| `--icp-iters <n>` | Outer ICP rounds for `dense`/`full` | `30` |
| `--detector <yunet\|lbf\|mediapipe>` | Landmark backend (§2); `mediapipe` needs the pre-pass | `yunet` |
| `--camera <idx>` / `--live-source <path>` | Live input: device index, or a video file / image sequence | `0` |
| `--live-width <px>` / `--live-frames <n>` / `--live-nodisplay` | Live processing width; headless test run | `640` / `0` / off |
| `--photo-refine` | Live: pyramid photometric pose refinement (safeguarded) | off |
| `--optimize-focal` | Solve fx=fy during personalisation (experimental, see §4b) | off |
| `--timers` | Print per-stage timings | off |

---

## 6. Output layout

Everything writes into `data/out/<tag>/` so modes never clobber each other:

```
data/out/
  biwi_video_rgb/      # --mode rgb
  biwi_video_full/     # --mode rgbd
  biwi_dense/          # --mode dense
  biwi_sparse/         # full's Stage-1 sparse fit
  live/                # --mode live-cpu snapshots / headless frames
  debug/               # mean-face sanity render, current_face.obj
```

`rgb`/`rgbd` write one 3-panel composite **per frame** into
`<tag>/frames/frame_XXXXX.png`, plus the whole sequence as `<tag>/tracking.mp4`.
Each panel is: **overlay** (render blended on the frame) | **reconstruction @
pose** (mask alone on black, at the tracked pose) | **reconstruction frontal**
(mask alone, straight-on).

The single-frame geometry modes (`dense`/`full`) write `fitted_face*.obj`,
wireframe/render overlays, and a `mask_panels*.png` strip into `biwi_dense/` /
`biwi_sparse/`.
