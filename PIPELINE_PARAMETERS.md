# Fitting pipeline — parameters per stage

What the analysis-by-synthesis pipeline solves for, stage by stage. Everything
here is pure Markdown (ASCII UML) — no external diagram tools.

Source: [`src/CeresFitter.cpp`](src/CeresFitter.cpp), driven from
[`src/main.cpp`](src/main.cpp) (`overlayMeshOnPhoto`).

---

## The five ways to parameterise a face

```
  +===================================================================+
  |                        FACE PARAMETERS                            |
  +===============+===========+===========+=================+=========+
  | group         | symbol    | size      | controls        | basis   |
  +===============+===========+===========+=================+=========+
  | Pose          | R, t      | 3 + 3     | head placement  |  —      |
  | Geometry (id) | alpha     | 30 (/199) | face shape      | BFM id  |
  | Albedo        | beta      | 30 (/199) | skin colour     | BFM col |
  | Illumination  | gamma     | 9 x 3     | lighting (SH)   |  —      |
  | Expression    | delta     | 0 (/100)  | mouth/brows/... | BFM exp |
  +===============+===========+===========+=================+=========+
```

- **Pose** — rigid: angle-axis rotation `R` (3) + translation `t` (3) = 6 DOF.
- **Geometry `alpha`** — BFM *identity* PCA coeffs; we optimise the first 30 of 199
  (`kShapeCoefficientCount`).
- **Albedo `beta`** — BFM *colour* PCA coeffs; first 30 of 199
  (`kAlbedoCoefficientCount`).
- **Illumination `gamma`** — order-2 spherical-harmonics lighting, 9 coeffs per
  RGB channel.
- **Expression `delta`** — BFM *expression* PCA (100 coeffs). **Present in the
  model, but NOT solved by any stage** — held at neutral (`delta = 0`).

---

## Pipeline overview (activity diagram)

```
                 iPhone RGB photo + intrinsics K
                              |
                              v
        +===========================================+
        |  DETECT LANDMARKS (python, offline)        |
        |  9 interior + 8 jaw-contour 2D points      |
        +===========================================+
                              |
                              v
   +==============================================================+
   |  STAGE 1 — fitPose                                            |
   |  ----------------------------------------------------------  |
   |  solve :  Pose (R, t)                                         |
   |  fixed :  Geometry = mean,  Expression = neutral             |
   |  data  :  interior landmark reprojection                     |
   +==============================================================+
                              |  R, t
                              v
   +==============================================================+
   |  STAGE 2 — fitPoseAndShapeContour                            |
   |  ----------------------------------------------------------  |
   |  solve :  Pose (R, t)  +  Geometry (alpha, 30)               |
   |  fixed :  Expression = neutral                               |
   |  data  :  interior landmarks + sliding jaw-contour term      |
   |  prior :  L2 on alpha (Mahalanobis)                          |
   +==============================================================+
                              |  R, t, alpha
                              v
   +==============================================================+
   |  STAGE 3 — fitPhotometric   (outer loop, per iteration)     |
   |  ----------------------------------------------------------  |
   |  3a estimate Illumination gamma   (linear LS)               |
   |  3b estimate Albedo      beta     (linear LS)               |
   |  3c refine   Pose (R, t)          (Ceres, per-pixel)        |
   |  ----------------------------------------------------------  |
   |  solve :  Illumination + Albedo + Pose                      |
   |  FROZEN:  Geometry (alpha)   <- comes from Stage 2          |
   |  fixed :  Expression = neutral                              |
   |  data  :  render() G-buffer vs photo (per-pixel colour)    |
   +==============================================================+
                              |
                              v
              fitted mesh + textured overlay
```

Note Stage 3 **freezes** `alpha`: the photometric term has no silhouette signal,
so re-optimising geometry there would let its mean-pulling prior shrink the face.
Geometry belongs to Stage 2 (which has the contour evidence); appearance belongs
to Stage 3.

---

## Who owns which parameter (class diagram)

`FitParameters` is the state passed down the pipeline. Each field is written by
exactly the stage that has evidence for it.

```
   +--------------------------------------+
   |            FitParameters             |
   +--------------------------------------+
   | + pose.angleAxis   : Vec3   (R)      |<--- Stage 1, 2, 3
   | + pose.translation : Vec3   (t)      |<--- Stage 1, 2, 3
   | + shapeCoefficients: Vec30  (alpha)  |<--- Stage 2      (frozen in 3)
   | + albedoCoefficients: Vec30 (beta)   |<--- Stage 3
   | + sh               : Mat9x3 (gamma)  |<--- Stage 3
   |   (expression delta : not a field)   |      never set
   +--------------------------------------+
             ^                 ^
             | reads/writes    | reads/writes
   +------------------+   +-------------------------+
   |  CeresFitter     |   |  Renderer / Lighting    |
   +------------------+   +-------------------------+
   | fitPose          |   | render()  uses alpha,   |
   | fitPoseAndShape* |   |   beta, gamma, R, t, K  |
   | fitPhotometric   |   | shadeVertices(gamma)    |
   +------------------+   +-------------------------+
```

---

## Solve matrix (stage x parameter)

```
   legend:  #=solve   o=fixed/frozen   .=not modelled (neutral)

                 Pose   Geometry   Albedo   Illum.   Expr.
                 (R,t)  (alpha)    (beta)   (gamma)  (delta)
   -----------  ------  --------   ------   -------  -------
   Stage 1        #        o         .         .        .
   Stage 2        #        #         .         .        .
   Stage 3        #        o         #         #        .
   -----------  ------  --------   ------   -------  -------
   ever solved?  yes      yes       yes       yes      NO
```

---

## What is NOT solved (and where it would go)

```
   +-----------------------------------------------------------+
   |  Expression (delta, BFM 100-dim)                          |
   |  - available: bfm.expr_basis_raw(), bfm.expr_sigma()      |
   |  - used by  : bfm.shape(alpha, delta)  (2-arg overload)   |
   |  - solved by: NOTHING  ->  face stays neutral             |
   |  - would fit in: Stage 2 (landmarks move with expression) |
   |                  and/or Stage 3 (photometric)             |
   +-----------------------------------------------------------+
```

Adding expression means introducing a `delta` parameter block (regularised like
`alpha`) into the Stage-2 landmark residual and/or the Stage-3 photometric
residual, and rebuilding the mesh with the 2-argument `bfm.shape(alpha, delta)`.

---

## The other entry point: dense depth (Biwi)

For completeness — the RGB-D path (`--mode dense`, Biwi) is geometry-only:

```
   +==============================================================+
   |  fitDense  (depth ICP, outer loop)                          |
   |  solve :  Pose (R, t)  +  Geometry (alpha)                  |
   |  fixed :  Albedo, Illumination, Expression                  |
   |  data  :  point-to-point + point-to-plane vs depth cloud    |
   +==============================================================+
```

It recovers the same `Pose + Geometry` as Stage 2, but from a metric depth cloud
instead of 2D landmarks — no appearance terms.
