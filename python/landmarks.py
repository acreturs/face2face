# detects 2D face landmarks with OpenCV and maps them to BFM vertices
# also has small drawing helpers to check the detections
import json
import os
import urllib.request

import cv2
import numpy as np

# url for the pretrained OpenCV LBF landmark model (raw file on GitHub)
LBF_MODEL_URL = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lbfmodel.yaml")

# Correspondence from OpenCV 68-point LBF indices to BFM named landmark points.
# Indices are 0-based, following the dlib/OpenCV 68-landmark convention.
# LBF68_TO_BFM = {
#     30: "center.nose.tip",
#     # OpenCV/dlib 68-point indices are anatomical with respect to the subject's
#     # face, while the BFM landmark names are also anatomical. Avoid swapping
#     # left/right based on viewer-side image layout.
#     36: "left.eye.corner_outer",
#     39: "left.eye.corner_inner",
#     42: "right.eye.corner_inner",
#     45: "right.eye.corner_outer",
#     48: "left.lips.corner",
#     54: "right.lips.corner",
#     51: "center.lips.upper.inner",
#     57: "center.lips.lower.inner",
# }


# maps dlib/LBF 68-point indices to BFM landmark names
# left/right is from the subject's point of view, so dlib 36 (image-left eye)
# is the subject's right eye, which matches the BFM names
#
# there are two sets, picked per scenario (gen_landmarks.py --set small|dense):
#
# LBF68_TO_BFM_SMALL is the default, 9 solid points (nose tip, 4 eye corners,
#   2 lip corners, 2 lip centres). use it for the iphone / sparse-only fit. on
#   a high-res photo (face ~2300 px) more landmarks make the identity overfit
#   the depth (2D points carry no depth, a 0.9 mm shift becomes ~12 mm), so
#   keep it small, the 2D landmarks pin the pose not the identity
#
# LBF68_TO_BFM_DENSE has 25 points, for the Biwi / dense case. there the depth
#   term gives the identity, so the extra landmarks only help the stage-1 pose
#   and the overfitting is no longer a problem
#   the jawline 0-16 stays unmapped in both, contour points have no fixed vertex
LBF68_TO_BFM_SMALL = {
    30: "center.nose.tip",

    36: "right.eye.corner_outer",
    39: "right.eye.corner_inner",
    42: "left.eye.corner_inner",
    45: "left.eye.corner_outer",

    48: "right.lips.corner",
    54: "left.lips.corner",

    62: "center.lips.upper.inner",
    66: "center.lips.lower.inner",
}

LBF68_TO_BFM_DENSE = {
    8:  "center.chin.tip",     # chin tip is steady when the face is roughly frontal

    30: "center.nose.tip",
    31: "right.nose.wing.outer",
    33: "center.nose.attachement_to_philtrum",
    35: "left.nose.wing.outer",

    19: "right.eyebrow.bend.upper",
    21: "right.eyebrow.inner_upper",
    22: "left.eyebrow.inner_upper",
    24: "left.eyebrow.bend.upper",

    36: "right.eye.corner_outer",
    38: "right.eye.top",
    39: "right.eye.corner_inner",
    40: "right.eye.bottom",
    42: "left.eye.corner_inner",
    43: "left.eye.top",
    45: "left.eye.corner_outer",
    47: "left.eye.bottom",

    48: "right.lips.corner",
    50: "right.lips.philtrum_ridge",
    51: "center.lips.upper.outer",
    52: "left.lips.philtrum_ridge",
    54: "left.lips.corner",
    57: "center.lips.lower.outer",
    62: "center.lips.upper.inner",
    66: "center.lips.lower.inner",
}

# default mapping, kept under the old name
LBF68_TO_BFM = LBF68_TO_BFM_SMALL


def ensure_lbf_model(model_path=None):
    model_path = model_path or DEFAULT_MODEL_PATH
    model_path = os.path.abspath(model_path)
    if os.path.exists(model_path):
        return model_path

    directory = os.path.dirname(model_path)
    os.makedirs(directory, exist_ok=True)

    print(f"Downloading OpenCV LBF landmark model to {model_path} ...")
    try:
        urllib.request.urlretrieve(LBF_MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the LBF model from {LBF_MODEL_URL}: {exc}\n"
            "Please download it manually and place it at the path above."
        )
    return model_path


def create_landmark_detector(model_path=None):
    model_path = ensure_lbf_model(model_path)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(model_path)
    return facemark


HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
HAAR_CASCADE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "haarcascade_frontalface_default.xml")


def ensure_haar_cascade(cascade_path=None):
    cascade_path = cascade_path or HAAR_CASCADE_PATH
    cascade_path = os.path.abspath(cascade_path)
    if os.path.exists(cascade_path):
        return cascade_path

    directory = os.path.dirname(cascade_path)
    os.makedirs(directory, exist_ok=True)

    print(f"Downloading Haar cascade to {cascade_path} ...")
    try:
        urllib.request.urlretrieve(HAAR_CASCADE_URL, cascade_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download Haar cascade from {HAAR_CASCADE_URL}: {exc}\n"
            "Please download it manually and place it at the path above."
        )
    return cascade_path


def _face_cascade():
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    except AttributeError:
        cascade_path = ensure_haar_cascade()
    if not os.path.exists(cascade_path):
        cascade_path = ensure_haar_cascade()
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
    return cascade


def detect_landmarks(image, facemark, face_cascade=None, rgb=True, min_size=(50, 50)):
    """find 2D face landmarks in an image

    first runs the haar cascade to find faces, then the facemark model on each
    returns one (N, 2) array per face, or [] if no face was found
    """
    if face_cascade is None:
        face_cascade = _face_cascade()

    img = image.copy()
    if rgb and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=min_size)
    if len(faces) == 0:
        return []

    faces = sorted(faces,key=lambda box: box[2] * box[3],reverse=True,)

    faces = np.asarray(faces, dtype=np.int32)
    ok, landmarks = facemark.fit(gray, faces)
    if not ok:
        return []

    out = []
    for lm in landmarks:
        out.append(lm.reshape(-1, 2))
    return out


def draw_landmarks(image, landmarks, rgb=True, color=(0, 0, 255), radius=4):
    img = image.copy()
    if rgb and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for point in landmarks:
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(img, (x, y), radius + 1, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def draw_landmark_labels(image, landmarks, correspondences, rgb=True,
                         text_color=(255, 255, 255), bg_color=(0, 0, 0),
                         font_scale=0.35, thickness=1):
    img = image.copy()
    if rgb and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, info in correspondences.items():
        if idx < len(landmarks):
            x, y = int(round(landmarks[idx][0])), int(round(landmarks[idx][1]))
            label = info["bfm_name"].replace(".", " ")
            text = f"{idx}:{label}"
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            tx = x + 5
            ty = y - 5
            if ty - th - baseline < 0:
                ty = y + th + baseline + 5
            if tx + tw > img.shape[1]:
                tx = max(0, x - tw - 5)
            cv2.rectangle(img, (tx, ty - th - baseline), (tx + tw, ty + baseline), bg_color, cv2.FILLED)
            cv2.putText(img, text, (tx, ty), font, font_scale, text_color,
                        thickness, lineType=cv2.LINE_AA)
            cv2.circle(img, (x, y), 4, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# turns the index->name mapping into index->(name, bfm vertex) using the loaded model
def bfm_correspondences(bfm, mapping=LBF68_TO_BFM):
    out = {}

    for idx, name in mapping.items():
        vertex_idx = bfm.landmark_index(name)

        if vertex_idx < 0:
            print(f"[landmarks] WARNING: BFM landmark missing: {name}")
            continue

        out[idx] = {
            "lbf68_index": idx,
            "bfm_name": name,
            "bfm_vertex_idx": vertex_idx,
        }

    print(f"[landmarks] resolved {len(out)}/{len(mapping)} correspondences")

    for idx, info in sorted(out.items()):
        print(
            f"  LBF {idx:2d} -> "
            f"{info['bfm_name']} -> "
            f"BFM vertex {info['bfm_vertex_idx']}"
        )

    return out


def save_correspondence_table(correspondences, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in correspondences.items()}, f, indent=2)


def save_correspondence_text(correspondences, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx, info in sorted(correspondences.items()):
            f.write(f"{idx}\t{info['bfm_name']}\t{info['bfm_vertex_idx']}\n")


def landmark_correspondence_pairs(detected_landmarks, correspondence):
    """pair up detected 2D points with their BFM vertex indices

    detected_landmarks is an (N, 2) array of OpenCV 68-point detections
    correspondence is the dict from bfm_correspondences(bfm)
    """
    detected_landmarks = np.asarray(detected_landmarks, dtype=np.float64).reshape(-1, 2)
    pts2d = []
    vertex_indices = []
    for idx, info in sorted(correspondence.items()):
        if idx < len(detected_landmarks):
            pts2d.append(detected_landmarks[idx])
            vertex_indices.append(info["bfm_vertex_idx"])

    if len(pts2d) < 6 and len(pts2d) > 0:
        raise RuntimeError(
            f"Only {len(pts2d)} valid landmark correspondences were found. "
            "At least 6 are recommended for a stable pose fit."
        )

    if not pts2d:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.int32)
    return np.vstack(pts2d), np.asarray(vertex_indices, dtype=np.int32)
