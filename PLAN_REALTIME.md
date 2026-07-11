# Plan: real-time RGB face tracking from the MacBook camera

Target: a `--mode live` that opens the built-in camera, personalises the face
model on the fly (identity + albedo + lighting + **camera focal length**), then
tracks pose + expression per frame with a live overlay — the Face2Face loop,
scoped to what a MacBook Pro M2 CPU can do. All existing modes (iPhone, Biwi
sparse/dense/full/video) must keep working unchanged.

Implementor notes are written for a later session with no context: files are
named, and every phase ends with a concrete verification.

---

## 0. Ground rules & constraints

- **Live mode runs on the HOST, not the devcontainer.** Docker cannot access
  the Mac camera. The host build already works
  (`CPATH=/opt/homebrew/include LIBRARY_PATH=/opt/homebrew/lib make`), host
  OpenCV 4.12 has `videoio` (AVFoundation camera) + `highgui` (`cv::imshow`),
  and `models/` already holds the YuNet + LBF files. First camera open will
  trigger the macOS permission prompt for the terminal app.
- Container workflows (Biwi/iPhone verification) stay as they are; live mode
  simply isn't available there. Guard with a friendly error if no camera.
- Detector for live mode is **YuNet** (in-process). MediaPipe stays offline
  (pre-pass files) — a live MediaPipe bridge is out of scope.

## 1. Verdict on the Gaussian-pyramid proposal

**The idea is right — it is what Face2Face does — with one adjustment.**
Face2Face runs its dense photometric Gauss-Newton on a 3-level image hierarchy,
coarse to fine, exactly as proposed: solve at the coarse level, use the result
to initialise the next finer level. Coarse levels both smooth the energy
(bigger convergence basin, survives fast motion) and cost fewer residuals.

The adjustment: **the pyramid applies to the dense photometric term only, not
the sparse term.** Landmarks are detected once per frame at detector-native
resolution and their reprojection residuals are resolution-independent — there
is nothing to gain (and detector accuracy to lose) by re-detecting on
downsampled images. So the per-frame schedule is:

```
detect landmarks (once)                                  ~5 ms
sparse solve: pose+expression, warm-started               ~few ms
photometric refine, coarse → fine (e.g. 100 → 200 px):    budgeted
    level L: render, build subsampled residuals, 1-2 GN steps
lighting update (linear 9×9, every k frames)              ~1 ms
```

Note we already run photometric at a capped 400 px — the pyramid formalises
this into explicit levels with per-level iteration counts, and during tracking
the coarse level can be skipped when the warm start is close (prediction error
small), keeping only one cheap fine-ish level.

## 2. Architecture refactor (keep old modes working)

**New class `FaceTracker`** (include/FaceTracker.h, src/FaceTracker.cpp) —
extract the personalise-then-track logic currently inlined in `fitBiwiVideo`
(src/main.cpp) so offline video and live mode share it:

- State: identity α, albedo β, lighting SH, intrinsics K, previous/prev2 pose &
  expression (velocity prediction), EMA smoothing, gating centroid — all the
  logic that already exists in `fitBiwiVideo`, moved, not rewritten.
- `personalise(frame, obs [, depthCloud])` → runs fitPose + fitPoseAndShapeContour
  + fitPhotometric (current code), stores α/β/SH.
- `track(frame, obs [, depthCloud])` → prediction → contour fit with identity
  frozen → lighting update → smoothing. Returns the current `FitParameters`.
- **Hoist the per-call precompute:** `fitPhotometric` currently rebuilds the
  BFM_TO_CAM-aligned per-vertex mean/basis (53k × 30) on every call
  (src/CeresFitter.cpp:1646). Compute once in the tracker (or add a reusable
  "PhotometricContext" the fitter accepts). Same for the renderer instance.
- `fitBiwiVideo` becomes a thin loop over `FaceTracker` — **regression: Biwi
  video RGB + full must produce the same output as before the refactor.**

**New entry `runLive()`** (src/main.cpp, `--mode live`):
- `cv::VideoCapture cap(0)` (optional `--camera <idx>`), read at 1280×720 or
  640×480 (`--live-width`).
- Loop: grab → YuNet detect → gate → `tracker.track()` → render overlay →
  `cv::imshow` + FPS HUD → `cv::waitKey(1)`; keys: `p` re-personalise,
  `q` quit, `s` save snapshot to data/out/live/.
- First N frames (or until `p`): personalisation phase (see §3).
- Frame dropping: always process the latest grabbed frame; never queue.

## 3. Camera intrinsics estimation

Webcam K is unknown. Strategy:

- **Init:** `proj::defaultIntrinsics(W, H)` (60° HFOV guess, principal point =
  image centre) — already exists in src/render/ProjectionUtils.cpp.
- **Optimise focal during personalisation only.** Add one extra Ceres parameter
  block `double focal[1]` (fx = fy = focal, cx/cy fixed at centre) to the
  landmark/contour reprojection residuals in `fitPoseAndShapeContour`, enabled
  by a new flag `optimizeFocal` (default false → all existing call sites
  unchanged). Residuals already take K; templating fx/fy on `T` is mechanical.
- **Beware the focal↔depth ambiguity:** from a single frontal view, doubling
  focal ≈ doubling distance. Mitigations (both cheap): (a) the BFM face has
  metric size, so the identity prior anchors absolute scale — keep the identity
  reg active while focal is free; (b) personalise over a **small window of ~5
  keyframes with varied head pose** (Face2Face's model-based bundling, lite):
  accumulate frames where the detected yaw differs, solve identity+focal
  jointly over all of them with per-frame pose/expression. Freeze focal (and α,
  β) for tracking.
- Bound focal to [0.4·W, 3·W]; log it on screen.
- **Sanity check:** run the same optimisation on the iPhone photo where fx is
  known (1930 @ 2316 px) — start from the 60° guess and confirm it converges
  toward the truth (±10%).

## 4. Performance plan for M2 (per-frame budget ~33-66 ms)

Measured hotspots today (order): CPU rasteriser over 105k triangles;
autodiff photometric residuals; per-call precompute (§2); brute-force
correspondence scans. Measures, in order of impact/effort:

1. **Timers first** (Phase 0): a tiny scoped-timer helper printing per-stage ms
   (detect / sparse / render / photometric / total). All later work is judged
   against this.
2. **Landmarks-only tracking is the baseline real-time path**: YuNet (≈5 ms at
   320 input) + pose/expression solve on ≤17 landmark residuals (milliseconds,
   2-3 outer iterations with warm start + velocity prediction). This alone
   should hit 30 fps and is Phase 2's deliverable.
3. **Photometric refinement inside the budget** (Phase 4):
   - pyramid levels ~100/200 px; **subsample residuals** (~2-4k pixels max,
     pixelStride per level);
   - 1-2 Gauss-Newton steps per level, skip coarse level when prediction error
     is small;
   - render at level resolution only (the display overlay renders once, at
     display size, independent of the solver);
   - reuse hoisted precompute; consider `-O3 -march=native` and OpenMP for the
     rasteriser loop (Makefile tweak) before anything exotic.
4. **If still short**: analytic Jacobians for the photometric residual
   (replace autodiff Jets — the residual is small and hand-differentiable:
   bicubic image gradient × projection Jacobian × pose/expr chain), and/or a
   decimated tracking mesh (every 4th triangle) for solver renders. **Metal/GPU
   is explicitly out of scope** — stretch goal only.

Honest expectation: 25-30 fps with landmarks-only tracking; 10-20 fps with the
photometric refinement enabled at 200 px. That is demo-quality real-time on CPU;
full 30 fps dense tracking is a GPU project.

## 5. Phases (each independently verifiable)

**Phase 0 — profiling harness.** Scoped timers around detect/sparse/photo/render
in the video path; run Biwi video RGB 15 frames in-container, record baseline
ms table in the PR/commit message. *Verify: numbers printed, no behaviour change.*

**Phase 1 — FaceTracker refactor.** Extract from `fitBiwiVideo`; hoist the
photometric precompute. *Verify: Biwi video RGB + full (15 frames, container)
visually identical to pre-refactor frames; single-frame modes untouched.*

**Phase 2 — live mode, landmarks-only.** `--mode live` on the host: camera in,
personalise on keypress/first frames (photometric personalisation included —
it's one-off), track with sparse-only, live overlay + FPS HUD. *Verify: run on
the MacBook, ≥25 fps tracking, overlay follows the face; `p` re-personalises.*

**Phase 3 — focal optimisation.** `optimizeFocal` in the contour fit + keyframe
bundling-lite during personalisation. *Verify: iPhone known-focal test (§3)
converges from the 60° guess; live overlay silhouette visibly tighter than the
fixed-guess run (screenshot both).* 

**Phase 4 — pyramid photometric refinement in tracking.** Per §1/§4. *Verify:
timers show the budget; side-by-side video with/without refinement shows better
silhouette/jaw adherence during motion; fps reported.*

**Phase 5 — polish.** Tracking-loss recovery (gate fails N frames → re-detect
without warm start), lighting update cadence, snapshot key, RUNNING.md section
for live mode (host-only, camera permission note).

## 6. Risks / open points

- **Haar/LBF quality on the webcam** is moot (YuNet), but LBF jaw points on a
  low-light webcam may jitter → the EMA + gating we already have mitigates;
  keep `kTrackExprRegWeight` stiff until MediaPipe-grade mouth points exist.
- **Focal drift**: never optimise focal during tracking (frozen after
  personalisation) — it would eat pose depth changes.
- **highgui event loop**: `cv::imshow` must run on the main thread on macOS;
  keep the whole loop single-threaded first (capture→track→show), thread only
  if profiling demands it.
- **Expression richness**: with YuNet's 5 interior points, live expression is
  effectively jaw/mouth-corner only; document as limitation (MediaPipe live
  bridge = future work).
