# face2face — 3D Face Reconstruction

Fits the Basel Face Model 2017 to monocular RGB(-D) input by analysis-by-synthesis,
following the Face2Face approach. It runs offline on the Biwi Kinect dataset and
live from a webcam, and includes an optional CUDA renderer and a live
expression-transfer mode.

## Build

Dependencies: Eigen, OpenCV 4, HDF5, Ceres/glog, and HighFive (header-only, cloned
by the Makefile on first build). C++17. Development is done in the provided dev
container (VS Code: "Reopen in Container"), which has everything preinstalled.

Build with:

```bash
make
```

CUDA build (needs an NVIDIA GPU and toolkit); set the arch for your GPU:

```bash
make USE_CUDA=1 CUDA_ARCH=sm_86
```

The CUDA build adds the display renderer, the `live-gpu` and `verify-gpu` modes, and
the `--photo-gpu` flags. The default build needs no CUDA toolkit.

The landmark models (LBF and YuNet) live in `models/` and are shipped with the
docker image.

## Data

The BFM and Biwi datasets are not included but should live under

```
data/
  bfm/model2017-1_bfm_nomouth.h5     
  BK-1/01/*                      
```

Request the BFM from https://faces.dmi.unibas.ch/bfm/bfm2017/ and place it in
`data/bfm/`.

The Biwi Kinect Head Pose Database is found on:
https://huggingface.co/datasets/ZihanWang1029/BIWI. Place a sequence under `data/`, e.g. `data/BK-1/01`, where
each frame is `frame_NNNNN_rgb.png`, `frame_NNNNN_depth.bin` and
`frame_NNNNN_pose.txt`, with `rgb.cal` and `depth.cal` in the folder.

## Landmarks

The fitter needs 2D facial landmarks. MediaPipe gives the best results; run the
pre-pass once per sequence to write `landmarks_mp_*.txt` next to the frames:

```bash
python3 python/gen_landmarks_mediapipe.py --biwi-dir data/BK-1/01
```

Then run with `--detector mediapipe`. With `--detector yunet` (the default) or `lbf`,
landmarks are detected in-process and no pre-pass is needed. The live modes start the
MediaPipe coprocess (`python/mp_landmark_server.py`) themselves and fall back to YuNet
if mediapipe is unavailable. We recommend sticking to MediaPipe if possible.

## Main entrypoints and modes to run code

Run as `./build/face_recon --mode <MODE> [options]` (dataset: Biwi RGB-D).

| Mode | Description |
|------|-------------|
| `rgb` | Video reconstruction from RGB: landmarks and photometric (no depth term). |
| `rgbd` | Same as `rgb` plus the metric Kinect depth ICP term. |
| `live-cpu` | Realtime CPU tracker from the webcam. |
| `transfer` | Live expression transfer: personalise a target avatar, then drive its expression from the webcam. |
| `live-gpu` | Live CPU tracker with the CUDA display renderer (needs `USE_CUDA=1`). |
| `verify-gpu` | CPU vs CUDA renderer parity and speed check on the mean face (needs `USE_CUDA=1`). |
| `dense` | Single-frame depth-only ICP. |
| `full` | Single-frame sparse landmark fit followed by dense ICP. |


```bash
# offline reconstruction with RGB
./build/face_recon --mode rgb  --biwi-dir data/BK-1/01 --frames 200 --sparse-reg 30 --detector mediapipe

# offline reconstruction with RGBD
./build/face_recon --mode rgbd --biwi-dir data/BK-1/01 --frames 200 --sparse-reg 30 --detector mediapipe

# multi-keyframe identity bundle RGB
./build/face_recon --mode rgb --biwi-dir data/BK-1/01 --frames 200 --detector mediapipe --bundle --bundle-keyframes 7

# live tracking and expression transfer (host only, needs a webcam)
./build/face_recon --mode live-cpu --detector mediapipe
./build/face_recon --mode transfer --transfer-target data/BK-1/05 --detector mediapipe
```

Outputs (overlays and the tracked-video panels) are written to `data/out/`.

## Main code for losses and optimisation

The energy terms and the expression transfer are implemented in these functions.
Each fit function assembles a Ceres problem from the listed residual(s).

| Term | Residual (`src/CeresFitter.cpp`) | Assembled in |
|------|----------------------------------|--------------|
| Landmark loss | `LandmarkReprojectionResidual`, `LandmarkFocalReprojectionResidual` | `CeresFitter::fitPoseAndShapeContour` (interior landmarks + jaw contour) |
| Depth loss | `DepthPointResidual` (point-to-point + point-to-plane) | `CeresFitter::fitDense`; also added to `fitPoseAndShapeContour` for `rgbd` |
| Photometric loss | `PhotometricPixelResidual` | `CeresFitter::fitPhotometric` |
| Expression transfer | neutral-relative δ copy | `runTransferLive` (`src/main.cpp`) |

Identity bundling is in `CeresFitter::fitIdentityBundle` (geometric) and
`fitIdentityPhotometricBundle` (photometric). The per-frame personalise→track loop is
in `FaceTracker` (`src/FaceTracker.cpp`).

## Other Options

```
--biwi-dir <path>          Biwi sequence folder (default data/BK-1/01)
--frames <n>               number of frames for rgb/rgbd            (default 30)
--sparse-reg <λ>           identity/expression regulariser (default 30)
--detector <yunet|lbf|mediapipe>   landmark backend      (default yunet)
--bundle [--bundle-keyframes <k>]  multi-keyframe identity bundle (default k=7)
--transfer-target <dir>    avatar to drive in --mode transfer
--camera <i> | --live-source <path> | --live-width <px>   live input
--live-frames <n> --live-nodisplay   headless live test
--icp-iters <n>            ICP rounds for dense/full      (default 30)
--photo-gpu | --photo-gpu-analytic | --gpu-render   GPU solve/render (USE_CUDA=1)
--timers                   print per-stage timings
```

## Layout

```
src/            C++ pipeline
  main.cpp        per-mode entry points
  FaceTracker.*   personalise-then-track system
  CeresFitter.*   the energy terms and Ceres solves
  BFMLoader.*     Basel Face Model (HDF5) loading and utils
  BiwiLoader.*    Biwi RGB-D, calibration and projection utils
  LandmarkDetector.*   YuNet/LBF in-process detector
  render/         CPU renderer (G-buffer + SH) and the optional CUDA renderer
include/        headers
python/         gen_landmarks_mediapipe.py (landmark pre-pass),
                mp_landmark_server.py (live mode with MediaPipe)
.devcontainer/  Dockerfile (CPU) and Dockerfile.cuda (GPU)
```
