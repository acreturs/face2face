# loads the BFM-2017 face model
# the model is a generator, it turns shape coeffs into a 3D mesh, it does not look at images
from unicodedata import name

import h5py
import numpy as np
import json

class BFM:
    def __init__(self, path):
        with h5py.File(path, "r") as f:
            # identity shape model
            self.shape_mean = f["shape/model/mean"][:].astype(np.float64)        # (3N,)
            self.shape_basis = f["shape/model/pcaBasis"][:].astype(np.float64)   # (3N, 199)
            self.shape_std = np.sqrt(f["shape/model/pcaVariance"][:].astype(np.float64))
            # per-vertex rgb colour
            self.color_mean = f["color/model/mean"][:].astype(np.float64)        # (3N,)
            # triangle faces
            self.triangles = f["shape/representer/cells"][:].astype(np.int64).T  # (M, 3)
            # named landmarks stored in the model metadata
            self.bfm_landmarks = self._load_landmarks(f)
            if not self.bfm_landmarks:
                print("[BFM] WARNING: no BFM landmarks were loaded")
            else:
                for name, vertex_idx in list(self.bfm_landmarks.items())[:10]:
                    print(f"[BFM] landmark {name} -> vertex {vertex_idx}")

        self.n_vertices = self.shape_mean.size // 3
        print(f"[BFM] {self.n_vertices} vertices, "
              f"{self.shape_basis.shape[1]} shape components, "
              f"{self.triangles.shape[0]} triangles")
        print(f"[BFM] {len(self.bfm_landmarks)} named landmark mappings")

    def shape(self, alpha):
        """build a face from identity coeffs alpha, returns (N, 3) in mm

        vertices = mean + basis * (alpha * sigma), where sigma = sqrt(variance)
        alpha is scaled like a standard normal so we multiply it by sigma
        """
        k = len(alpha)
        v = self.shape_mean + self.shape_basis[:, :k] @ (alpha * self.shape_std[:k])
        return v.reshape(-1, 3)

    def mean_shape(self):
        """the average face when alpha = 0, (N, 3) in mm"""
        return self.shape_mean.reshape(-1, 3)

    def landmark_index(self, name):
        """vertex index for a named landmark, or -1 if we don't have it"""
        return self.bfm_landmarks.get(name, -1)

    def landmarks(self):
        return dict(self.bfm_landmarks)

    def albedo(self):
        """per-vertex rgb colour in [0, 1], (N, 3)"""
        c = self.color_mean.reshape(-1, 3)
        if c.max() > 1.5:        # some models store it as 0..255
            c = c / 255.0
        return np.clip(c, 0.0, 1.0)

    # reads the landmark metadata and maps each named point to its nearest mean-shape vertex
    def _load_landmarks(self, f):
        mean_shape = self.mean_shape()

        coordinates = {}

        if "metadata/landmarks/json" in f:
            raw = f["metadata/landmarks/json"][()]

            if isinstance(raw, bytes):
                text = raw.decode("utf-8", errors="ignore")
            else:
                text = raw.tobytes().decode("utf-8", errors="ignore")

            parsed = json.loads(text)

            for name, xyz in parsed.items():
                coordinates[name] = np.asarray(xyz, dtype=np.float64)

        elif "metadata/landmarks/text" in f:
            raw = f["metadata/landmarks/text"][()]

            if isinstance(raw, bytes):
                text = raw.decode("utf-8", errors="ignore")
            else:
                text = raw.tobytes().decode("utf-8", errors="ignore")

            for line in text.splitlines():
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) < 4:
                    continue

                name = parts[0]

                try:
                    coordinates[name] = np.array(
                        [
                            float(parts[-3]),
                            float(parts[-2]),
                            float(parts[-1]),
                        ],
                        dtype=np.float64,
                    )
                except ValueError:
                    continue

        else:
            print("[BFM] WARNING: no landmark metadata found")
            return {}

        names = {}

        for name, xyz in coordinates.items():
            dists = np.linalg.norm(mean_shape - xyz.reshape(1, 3), axis=1)
            names[name] = int(np.argmin(dists))

        return names