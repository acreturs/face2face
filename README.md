<!-- Rewritten by Claude (Anthropic Claude Code) on 2026-07-04. -->

# face2face — Face Reconstruction (3DSMC)

Fit the Basel Face Model 2017 to Biwi RGB-D frames: sparse landmark fit (pose)
→ dense depth ICP (identity) → project the photo texture onto the mesh.

Build & run inside the dev container (VS Code → "Reopen in Container").
Needs the BFM in `data/bfm/` (see SETUP.md) and Biwi (below).

## Get the Biwi dataset

The container already has cv2/numpy/h5py; the downloader only needs two extra
packages. Inside the container Python is PEP-668-managed, so pass the override:

```bash
pip install --break-system-packages remotezip requests   # container
# (on the host with a normal/conda env: pip install -r requirements.txt)
python3 python/get_biwi.py --list     # list sequences (24 total)
python3 python/get_biwi.py 01 07      # download some (~200-450 MB each)
python3 python/get_biwi.py --all      # or all (~7.2 GB)
python3 python/check_biwi.py 01       # verify → data/out/biwi_check_01.png
```

The official ETH link is dead; `get_biwi.py` pulls each sequence from a Hugging
Face mirror via HTTP range requests (CRC-checked, resumable). Genders per the
dataset's `females.txt`/`males.txt`: female 01-06,15,18,21 · male the rest.

## Generate things

```bash
# landmarks (once per frame; C++ reads the .txt). 9-point set by default,
# 25-point set (--set dense) for the Biwi/dense fit:
python3 python/gen_landmarks.py data/iphone/default/RGB/000000_RGB.png \
        data/iphone/default/landmarks_000000.txt
python3 python/gen_landmarks.py --set dense data/biwi/01/frame_00003_rgb.png \
        data/biwi/01/landmarks_00003.txt

# fit (writes meshes + overlays to data/out/):
make run                              # sparse fit, Biwi seq 01
make run MODE=full                    # sparse → dense → textured (the pipeline)
make run MODE=dense                   # dense only
make run MODE=sparse DATASET=iphone   # iPhone photo (RGB only)

# report figures (needs the two runs in its header):
python3 report_figures/week4/make_figs.py
```

## Flags

```
--biwi-seq NN     Biwi sequence 01..24                     (default 01)
--icp-iters N     outer ICP rounds of the dense fit        (default 30)
--sparse-reg L    shape regulariser λ of the iPhone fit    (default 100)
```

## Key outputs (`data/out/`)

- `biwi_rgb_overlay_textured.png` / `_100.png` — the fitted, photo-textured face
  rendered back onto the photo (70 % blend / fully replacing the mask)
- `biwi_face_render_textured.png` — frontal portrait of the fitted identity
- `fitted_face_biwi_dense.obj` (+ `_camframe.obj`, `biwi_head_cloud.obj`) — meshes
- `biwi_icp_progress/icp_XX.png` — one overlay per ICP iteration

## Datasets

- **BFM 2017** (`data/bfm/model2017-1_bfm_nomouth.h5`): 53,149 vertices, mm;
  face = `mean + pcaBasis·(α ⊙ σ)`, `σ = √pcaVariance`.
- **Biwi** (`data/biwi/NN/`): RGB + depth (640×480) from one Kinect, both cameras
  calibrated (`rgb.cal`/`depth.cal`, `p_rgb = R·p_depth + t`), GT head pose per
  frame. Formats decoded/verified in `python/biwi.py` + `check_biwi.py`.

Both are research/education-licensed → `data/` is gitignored.
