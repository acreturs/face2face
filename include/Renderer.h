#pragma once
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "ProjectionUtils.h"

// ── Per-frame inputs that change every optimiser iteration
// pose/intrinsics are small fixed-size Eigen types. Triangles are NOT here — they are constant
// across all frames and live in the Renderer (set once at construction).
struct RenderInput {
    Eigen::MatrixX3f shape;          // (N,3)
    Eigen::MatrixX3f albedo;         // (N,3)
    Eigen::Matrix3f  R;              // (3,3)
    Eigen::Vector3f  t;              // (3,)
    Eigen::Matrix3f  K;              // (3,3)
    Eigen::Matrix<float, 9, 3> sh;   // order-2 spherical-harmonics coeffs (9×RGB)
};

// ── Render result: image-shaped buffers live in cv::Mat
// `image`/`depth`/`mask` are the visible result; `triIdx`/`bary` are the
// G-buffer that makes the photometric term differentiable
struct RenderOutput {
    cv::Mat image;   // CV_32FC3  RGB in [0,1]
    cv::Mat depth;   // CV_32F    camera-frame Z, mm  (+inf where empty)
    cv::Mat mask;    // CV_8U     255 where a triangle was rasterised
    cv::Mat triIdx;  // CV_32S    covering triangle index, -1 if none
    cv::Mat bary;    // CV_32FC3  perspective-correct barycentric weights
};

class Renderer {
public:
    // Triangles + image size are constant for the whole run → set once.
    Renderer(int height, int width, Eigen::MatrixX3i triangles);

    RenderOutput render(const RenderInput& in) const;

    // exposed so the Ceres photometric cost functor can map triIdx → vertices
    const Eigen::MatrixX3i& triangles() const { return triangles_; }

    // Area-weighted per-vertex normals. (N,3) unit vectors.
    // Exposed publicly so main / debug helpers can serialise normals alongside
    // a mesh (e.g. saveCurrentModel writes them as `vn` lines in the .obj).
    static Eigen::MatrixX3f computeNormals(const Eigen::MatrixX3f& shape,
                                           const Eigen::MatrixX3i& triangles);

private:
    int H_;
    int W_;
    Eigen::MatrixX3i triangles_;     // (M,3)

    // Step 3 — back-face culling. Given camera-frame vertices, return a
    // per-triangle flag: true = front-facing (keep), false = back-facing (skip).
    // Size == triangles_.rows().
    static std::vector<bool> backfaceMask(const Eigen::MatrixX3f& V_cam,
                                          const Eigen::MatrixX3i& triangles);

    // Step 4 — z-buffer rasterizer. Fills the G-buffer (triIdx, bary, depth,
    // mask) in `out` for every front-facing triangle. `uv` are the projected
    // pixel coords, `V_cam` provides per-vertex camera-frame z for perspective
    // correction + depth.
    void rasterize(const proj::Pixels&       uv,
                   const Eigen::MatrixX3f&   V_cam,
                   const std::vector<bool>&  frontFacing,
                   RenderOutput&             out) const;

    // Step 6 — blend per-vertex shaded colours through the G-buffer into
    // out.image. `shadedColors` is (N,3) RGB from light::shadeVertices.
    void interpolateShading(const Eigen::MatrixX3f& shadedColors,
                            RenderOutput&           out) const;
};
