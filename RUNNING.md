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

| `--mode` | `--dataset` | What it does | Landmarks | Needs depth? |
|---|---|---|---|---|
| `sparse` (default) | `iphone` | Pose + identity from 2D landmarks only | **auto (YuNet)** | no |
| `sparse` | `biwi` | Same, on a Biwi RGB frame | file | no |
| `dense` | `biwi` | Depth-only ICP (geometry, no photo term) | none | yes |
| `full` | `biwi` | Stage 1 sparse landmark fit → Stage 2 dense ICP (no photometric) | file | yes |
| `photometric` | `iphone` | Sparse+contour+expression → photometric (lighting+albedo). RGB only. | **auto (YuNet)** | no |
| `photometric` | `biwi` | Same, on a Biwi RGB frame. Add `--depth` for the **full depth+photo+sparse fit** (see §3) | **auto (YuNet)** | optional |
| `video` | `biwi` | Personalise on frame 0, then track a sequence (pose+expression only) | **auto (YuNet)** | optional (`--depth`) |

**auto (YuNet)** = landmarks are detected in-process by the C++ `LandmarkDetector`
(YuNet CNN face box + pose-robust 5 points, LBF jaw contour) — no Python step,
no landmark file. **file** = still reads a pre-generated `landmarks_XXXXX.txt`
(see §2).

`--dataset iphone` only ever runs RGB-only (no depth exists for the iPhone
selfie); `--dataset biwi` is where the depth term is available.

---

## 2. Landmarks

Most modes now detect landmarks **in-process** (C++ `LandmarkDetector`: YuNet
face box + pose-robust 5 points + LBF jaw contour) — **no landmark file and no
Python step**. This covers: iPhone (`sparse`/`photometric`), Biwi
`photometric` (incl. `--depth`), and `video`.

Only two modes still read a pre-generated `landmarks_XXXXX.txt` file:
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

## 3. Full Biwi fit (depth + photo + sparse) for a given person and frame

This is the complete analysis-by-synthesis pipeline in one run: **sparse
landmarks → jaw-contour + expression + depth ICP (jointly) → photometric
(lighting + albedo)**. Landmarks are detected in-process (YuNet) — **no landmark
file needed**. Person = the Biwi subject folder; frame = a specific
`frame_XXXXX` in it, selected by its 0-based index via `--biwi-frame`.

```bash
# 1. Pick a person (subject folder) and frame number, e.g. subject 01, frame 00201
PERSON=data/BK-1/01
FRAME=00201

# 2. Find that frame's INDEX in the sorted frame list (0-based). --biwi-frame
#    selects by index, not by raw frame number. grep -n gives the 1-based line
#    number, so the index is (that number − 1).
ls $PERSON/*_rgb.png | sed -E 's/.*frame_([0-9]+)_rgb.*/\1/' | sort -n | grep -n "^${FRAME}$"
# → e.g. "199:00201"  means line 199, so index = 198

# 3. Run the full fit (--depth is what adds the depth ICP term)
./build/face_recon --mode photometric --dataset biwi \
    --biwi-dir $PERSON --biwi-frame 198 --depth --sparse-reg 30
```

If you just want the **first available frame** of a person (simplest case,
`--biwi-frame` defaults to `0`):

```bash
./build/face_recon --mode photometric --dataset biwi \
    --biwi-dir data/BK-1/01 --depth --sparse-reg 30
```

**Outputs** land in `data/out/biwi_full/` (see §5), including
`mask_panels_photometric.png` — the overlay + reconstruction-only views side
by side, which is the quickest way to check the fit.

Drop `--depth` to run the same pipeline **RGB-only** (no depth term) —
outputs go to `data/out/biwi_rgb/` instead.

---

## 4. Other common runs

```bash
# iPhone: sparse pose+identity only
./build/face_recon --mode sparse --dataset iphone

# iPhone: full RGB analysis-by-synthesis (sparse+contour+expr → photometric)
./build/face_recon --mode photometric --dataset iphone --sparse-reg 30 --iphone-frame 0

# Biwi: sparse landmarks only  (needs a landmarks_XXXXX.txt file — see §2)
./build/face_recon --mode sparse --dataset biwi --biwi-dir data/BK-1/01

# Biwi: depth-only ICP (no landmarks, no photo term)
./build/face_recon --mode dense --biwi-dir data/BK-1/01 --icp-iters 30

# Biwi: sparse landmark init → dense ICP (no photometric; needs a file — see §2)
./build/face_recon --mode full --biwi-dir data/BK-1/01 --icp-iters 30

# Biwi video: depth tracking (robust to rotation; landmarks auto-detected)
./build/face_recon --mode video --dataset biwi --biwi-dir data/BK-1/01 \
    --depth --frames 200 --sparse-reg 30

# Biwi video: RGB-only tracking (in-C++ YuNet+LBF detection every frame, no depth)
./build/face_recon --mode video --dataset biwi --biwi-dir data/BK-1/01 \
    --frames 200 --sparse-reg 30
```

---

## 5. CLI flag reference

| Flag | Meaning | Default |
|---|---|---|
| `--mode <sparse\|dense\|full\|photometric\|video>` | Which pipeline to run | `sparse` |
| `--dataset <iphone\|biwi>` | Which input | `biwi` |
| `--sparse-reg <λ>` | Shape/expression regulariser weight | `100.0` |
| `--biwi-seq <NN>` | Shortcut for `data/biwi/NN` | — |
| `--biwi-dir <path>` | Explicit Biwi subject folder (e.g. `data/BK-1/01`) | `data/biwi/01` |
| `--biwi-frame <k>` | Frame **index** (0-based, sorted) within `--biwi-dir` for single-image Biwi modes | `0` |
| `--iphone-frame <k>` | iPhone frame index | `0` |
| `--icp-iters <n>` | Outer ICP rounds for `dense`/`full` | `30` |
| `--frames <n>` | Number of frames for `video` mode | `30` |
| `--depth` | Add the depth ICP term (Biwi `photometric`/`video` modes) | off |

---

## 6. Output layout

Everything writes into `data/out/<tag>/` so datasets/modes never clobber each
other:

```
data/out/
  iphone/            # --dataset iphone (sparse/photometric)
  biwi_sparse/        # --mode sparse --dataset biwi
  biwi_dense/          # --mode dense
  biwi_rgb/            # --mode photometric --dataset biwi   (no --depth)
  biwi_full/           # --mode photometric --dataset biwi   --depth
  biwi_video_rgb/      # --mode video                        (no --depth)
  biwi_video_full/     # --mode video                        --depth
  debug/               # mean-face sanity render, current_face.obj
```

Inside each single-image tag folder:

- `fitted_face*.obj` — the reconstructed mesh at each stage.
- `overlay_<stage>.png` — wireframe + landmark reprojection.
- `render_overlay_<stage>.png` / `render_photometric_appearance.png` — full
  render blended over the photo.
- **`mask_panels_<stage>.png`** — 3-panel strip: **overlay** (render blended
  on the photo) | **reconstruction @ pose** (mask alone, on black, at the
  fitted scale/pose) | **reconstruction frontal** (mask alone, straight-on).
  Present for every stage of every single-image mode (iPhone, Biwi sparse,
  Biwi RGB/full) and as `mask_panels.png` for `biwi_dense`.

`video` mode writes one 3-panel composite **per frame** into
`<tag>/frames/frame_XXXXX.png`, and the whole sequence as `<tag>/tracking.mp4`.
