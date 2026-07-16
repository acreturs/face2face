<!-- Written by Claude (Anthropic Claude Code) on 2026-07-12. -->

# GPU renderer (CUDA)

A CUDA rasteriser that is a **drop-in for the CPU `Renderer`**: same constructor,
same `render(RenderInput) → RenderOutput` contract, same G-buffer semantics. The
CPU renderer stays the reference; the GPU one is verified against it.

- `include/CudaRenderer.h` — the class (no CUDA headers; safe to include from g++)
- `src/render/CudaRenderer.cpp` — host orchestration (Eigen vertex stage + copies)
- `src/render/cuda_raster.cu` — the kernels (no Eigen/OpenCV)

The two sides talk only through three `extern "C"` launchers, so **nvcc never
compiles Eigen/OpenCV and g++ never sees CUDA headers.**

---

## Why the renderer is the target (and what it does *not* fix)

Measured on the CPU baseline (`test_logs/`):

| Path | Per-frame | Renderer's share |
|---|---|---|
| Live, landmarks-only | ~55 ms (~18 fps) | `live/render` ~20 ms → **~1/3** |
| Live, `--photo-refine` | ~240 ms (~4 fps) | render is part of the 164 ms refine |
| Offline `rgb`/`rgbd` video | **~2000–2700 ms/frame** | small — see below |

Key fact about the pipeline: it uses **fixed-visibility differentiable
rendering**. The rasteriser runs **once per photometric outer iteration** to
produce the G-buffer (triangle index + barycentrics); Ceres then does ~26
Levenberg–Marquardt steps of **per-pixel autodiff reusing that fixed G-buffer,
without re-rendering**.

So in the expensive photometric solve, the split is roughly:

- rasterise: tens of ms  ← **the GPU renderer removes this**
- per-pixel autodiff Jacobian in Ceres: ~95% of the time  ← **still on the CPU**

**Consequence:** a GPU *renderer* gives a clean win on the display overlay, the
one-off personalise renders, and landmarks-only live FPS (≈18 → ≈28 fps by
dropping `live/render` off the critical path). It does **not**, by itself, make
the 2 s/frame photometric path real-time — the dominant cost is the Jacobian.
Making *that* real-time is the next milestone (see Roadmap).

---

## The CPU renderer (what the GPU version reproduces)

`Renderer::render` ([`src/render/Renderer.cpp`](src/render/Renderer.cpp)), 5 steps:

1. **project** vertices to the camera frame (`proj::toCameraFrame`, `proj::project`)
2. **normals** per vertex (`computeNormals`), rotated into the camera frame
3. **back-face cull** — front-facing when the camera-space normal `n_z < 0`
4. **z-buffer rasterise** — per triangle, walk its 2D bbox, edge-function
   barycentrics, **perspective-correct** via the `1/z`-is-linear trick, keep the
   nearest fragment per pixel. Writes the G-buffer.
5. **shade + interpolate** — SH-shade each vertex, blend the 3 vertex colours per
   pixel by the perspective-correct barycentrics.

**Output contract that must be preserved** (`RenderOutput`):
`image` (CV_32FC3 RGB), `depth` (CV_32F, +inf empty), `mask` (CV_8U),
`triIdx` (CV_32S, −1 empty), `bary` (CV_32FC3, perspective-correct). The last two
are what make the photometric term differentiable — the GPU port reproduces them
exactly, not just the visible image.

The CPU rasteriser is **single-threaded** over ~105k triangles — that loop (step
4 + 5) is exactly what moves to the GPU.

### What runs where in `CudaRenderer`

Steps 1–2 and the SH shading (all O(N) ≈ 53k, already fast and Eigen-vectorised)
run **on the CPU using the identical `proj::`/`light::`/`computeNormals` helpers**
— so the projected pixels, normals and shaded colours are byte-for-byte the same
as the CPU renderer. Only steps 4–5 (the O(triangles × pixels) hot loop) run on
the GPU:

- **kernel 1 — clearZ:** reset the packed depth buffer.
- **kernel 2 — raster:** one thread per triangle; `atomicMin` of a packed key
  `(depthBits << 32 | triangleId)` into the z-buffer. Packing makes the *same*
  fragment win as on the CPU: nearest depth first, ties broken by the smaller
  triangle id (the CPU's strict-`<` test keeps the first-written = smaller `f`).
- **kernel 3 — resolve:** one thread per pixel; decode the winning triangle,
  recompute its perspective-correct barycentrics, blend the shaded colours, write
  the full G-buffer + image.

---

## Build & run

Requires an NVIDIA GPU and the CUDA toolkit. The default build is unchanged and
needs no CUDA.

```bash
# default (CPU only) — unchanged
make

# with the CUDA renderer (set CUDA_ARCH for your GPU, e.g. sm_86 Ampere, sm_89 Ada)
make USE_CUDA=1 CUDA_ARCH=sm_86
# CUDA_HOME defaults to /usr/local/cuda; override if installed elsewhere.
```

### 1. Verify parity + speed (do this first — needs only the BFM, no Biwi)

```bash
./build/face_recon --mode verify-gpu
```

Renders the mean face on **both** renderers and prints a diff table
(mask / triIdx mismatches, image/bary/depth max|Δ|) plus a render-time speedup,
and writes `data/out/debug/gpu_check_{cpu,gpu}.png`.

**How to read it:** `image mean|Δ|` should be ~1e-6–1e-4 and `triIdx mismatches`
a small fraction of a percent, all at silhouette/shared edges — that is expected
float rounding (nvcc vs g++), not a bug. The CPU is the reference. The `.cu` is
compiled with `--fmad=false` to keep the arithmetic close; if you see large
regions differing, that's a real bug (check the triangle-flatten order and the
`R`/`t`/`K` conventions).

### 2. Live overlay on the GPU (host with a camera)

```bash
./build/face_recon --mode live-gpu --sparse-reg 30
```

Identical to `live-cpu` except the on-screen overlay is GPU-rendered. Tracking
runs on the CPU; add `--photo-gpu` to move the photometric geometry solve to the
GPU too (see below).

### 3. GPU photometric solve (`--photo-gpu`)

This is the big one — it moves the per-pixel photometric **geometry solve** (the
~95% hot spot, see above) off Ceres/CPU and onto the GPU. It is **opt-in** and
composes with any mode that runs a photometric geometry step. Two Jacobian
backends, both selectable (the CPU/Ceres path stays the reference):

- `--photo-gpu` — **finite-difference** Jacobian (central differences). Robust,
  a couple of FD step constants to tune (`H_AA`/`H_T`/`H_SHAPE` in the `.cu`).
- `--photo-gpu-analytic` — **analytic** Jacobian (image-gradient × projection ×
  pose/shape chain; pose rotation solved as a local SO(3) perturbation). Fewer
  residual evals per step, no FD tuning.

```bash
# offline RGB video, GPU photometric (finite-diff)
./build/face_recon --mode rgb --biwi-dir data/BK-1/01 --frames 30 --photo-gpu --timers
# same, analytic Jacobian
./build/face_recon --mode rgb --biwi-dir data/BK-1/01 --frames 30 --photo-gpu-analytic --timers
# live overlay + GPU pose photometric refinement (real-time path; needs a camera)
./build/face_recon --mode live-gpu --sparse-reg 30 --photo-refine --photo-gpu-analytic
# headless live smoke test (Biwi frames as a fake camera — no camera needed)
./build/face_recon --mode live-gpu --live-source data/BK-1/01/frame_00003_rgb.png \
    --live-frames 12 --live-nodisplay --photo-refine --photo-gpu-analytic --timers
```

**Photo-refine** (`--photo-refine`, the live pyramid pose refinement) runs its
solve through the same `fitPhotometric`, so it automatically uses whichever GPU
backend you selected — it's a pure pose solve (6 params), the cleanest case for
the analytic Jacobian.

**How it works** (`src/render/cuda_photometric.cu` + `solvePhotometricGpu` in
`src/CeresFitter.cpp`): the correspondence set is fixed per outer iteration, so
the per-pixel residual `r = √w·(rendered − input(project(R·S+t)))` and its
Jacobian are built on the GPU (one thread per pixel), assembled into the
`nParams×nParams` Gauss–Newton normal equations, and the tiny system is solved on
the host with Eigen inside a Levenberg–Marquardt loop. `nParams` = 6 (pose-only,
the live `--photo-refine` case) or 6+30 (pose+identity, personalisation).

**Comparing CPU vs GPU:** run the same command with and without `--photo-gpu`;
each outer iteration prints `photo N | pixels … | RMSE(before solve) …`. The RMSE
trajectory (and the final `translation`/`shapeCoeff`) should track the CPU/Ceres
run closely. The CPU path is the reference.

**Deliberate approximations** (documented so the parity gap is expected, not a
bug):
- **Jacobian:** finite-difference (`--photo-gpu`) or analytic (`--photo-gpu-analytic`)
  instead of Ceres autodiff. FD step sizes are `H_AA`/`H_T`/`H_SHAPE` at the top of
  `cuda_photometric.cu` — tune if convergence stalls or oscillates. The analytic
  path has no such tuning.
- **Bilinear** image sampling instead of Ceres' **bicubic** — sub-pixel colour
  differences, negligible for tracking. (The analytic gradient is the exact
  derivative of this bilinear interpolant.)
- A hand-rolled LM (λ up/down on accept/reject) rather than Ceres' trust region —
  converges to a very similar minimum, not bit-identical.
- Three-way comparison to sanity-check the analytic path: run the same sequence
  with `--photo-gpu` (FD) and `--photo-gpu-analytic` and compare the `photo N`
  RMSE trajectories — they should agree closely, and both should track the CPU
  Ceres run.

**Perf note:** the normal-equation accumulation uses global `atomicAdd` (one per
matrix entry per pixel). Correct but not optimal; a shared-memory block reduction
is the obvious next optimisation if profiling shows it dominating.

---

## Determinism note

`image`, `depth` and `bary` are floats computed in a different order and rounding
mode on the GPU, so exact equality is not expected. `verify-gpu` reports the
magnitudes; the pipeline is robust to sub-ULP differences (correspondences are
Huber-robust and re-solved each iteration). Triangle-index ties are resolved
identically to the CPU (smaller id wins), so `triIdx` differs only where a pixel
sits exactly on an edge and rounding flips the inside test.

---

## Roadmap

1. ✅ **GPU rasteriser** (`--mode live-gpu`, `verify-gpu`).
2. ✅ **GPU photometric residual + Jacobian** (`--photo-gpu`) — the per-pixel
   residual + finite-difference Jacobian + normal equations now run on the GPU;
   the small linear solve/LM is on the host. This is where the ~95% lived.
3. ✅ **Analytic Jacobian** (`--photo-gpu-analytic`) — image-gradient × projection
   Jacobian × pose/shape chain, pose rotation as a local SO(3) perturbation. Kept
   alongside the finite-difference path (`--photo-gpu`), which stays the default.
4. ◻ **Keep the G-buffer on the device.** `CudaRenderer` copies the G-buffer back
   to host cv::Mats; `cuda_photometric` re-uploads the derived correspondences.
   Fuse them so `triIdx`/`bary`/base-point/target stay resident between render and
   solve (copy only the final display image).
5. ◻ **Shared-memory reduction** for the normal-equation accumulation (replace the
   global `atomicAdd`s).
6. ◻ **GPU vertex stage** (project/normals/shade; normals need an atomic
   scatter-add) — removes the remaining CPU O(N) work + host→device uploads.
7. ◻ **Decimated solver mesh / pyramid levels** — orthogonal, compounds.

Suggested next step after this lands + tests green: item 3 (analytic Jacobian),
with its own parity check against the CPU photometric fit.
