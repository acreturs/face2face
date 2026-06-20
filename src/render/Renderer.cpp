#include "Renderer.h"
#include "ProjectionUtils.h"
#include "Lighting.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

// ── ctor: triangles + image size are fixed for the run ───────────────────────
Renderer::Renderer(int height, int width, Eigen::MatrixX3i triangles)
    : H_(height), W_(width), triangles_(std::move(triangles)) {}

// ── render orchestration (rasterizer + shading land here in M3.3 Steps 2-7) ──
RenderOutput Renderer::render(const RenderInput& in) const
{
    RenderOutput out;
    out.image  = cv::Mat::zeros(H_, W_, CV_32FC3);
    out.depth  = cv::Mat(H_, W_, CV_32F, std::numeric_limits<float>::infinity());
    out.mask   = cv::Mat::zeros(H_, W_, CV_8U);
    out.triIdx = cv::Mat(H_, W_, CV_32S, cv::Scalar(-1));
    out.bary   = cv::Mat::zeros(H_, W_, CV_32FC3);

    // ── Step 2: project to the camera ────────────────────────────────────────
    const Eigen::MatrixX3f V_cam = proj::toCameraFrame(in.shape, in.R, in.t);
    const proj::Pixels     uv    = proj::project(V_cam, in.K);
    // Normals into the camera frame too (needed by SH shading in Step 5).
    const Eigen::MatrixX3f N      = computeNormals(in.shape, triangles_);
    const Eigen::MatrixX3f N_cam  = proj::normalsToCameraFrame(N, in.R);

    // ── Step 3: back-face culling ────────────────────────────────────────────
    const std::vector<bool> frontFacing = backfaceMask(V_cam, triangles_);

    // ── Step 4: rasterize into the G-buffer ──────────────────────────────────
    rasterize(uv, V_cam, frontFacing, out);

    // ── Step 5: SH shading per vertex ────────────────────────────────────────
    const Eigen::MatrixX3f shaded = light::shadeVertices(in.albedo, N_cam, in.sh);

    // ── Step 6: interpolate shaded colours through the G-buffer → out.image ───
    interpolateShading(shaded, out);
    return out;
}

// ── Step 6: per-pixel colour from the G-buffer ───────────────────────────────
// For every covered pixel, blend its triangle's three shaded vertex colours by
// the perspective-correct barycentric weights stored in the G-buffer. This is
// the lecture-slide equation C_pixel = b0·C0 + b1·C1 + b2·C2.
void Renderer::interpolateShading(const Eigen::MatrixX3f& shadedColors,
                                  RenderOutput&           out) const
{
    for (int y = 0; y < H_; ++y) {
        for (int x = 0; x < W_; ++x) {
            const int f = out.triIdx.at<int>(y, x);
            if (f < 0) continue;                      // empty pixel — skip

            const cv::Vec3f bary = out.bary.at<cv::Vec3f>(y, x);
            const int index0 = triangles_(f, 0);
            const int index1 = triangles_(f, 1);
            const int index2 = triangles_(f, 2);

            // TODO Step 6: blend the three vertices' shaded colours.
            //   colour (1×3) = bary[0]·shadedColors.row(i0)
            //                + bary[1]·shadedColors.row(i1)
            //                + bary[2]·shadedColors.row(i2)
            // then write it (image is stored RGB float):
            //   out.image.at<cv::Vec3f>(y, x) =
            //       cv::Vec3f(colour(0), colour(1), colour(2));

            const Eigen::RowVector3f colour = bary[0] * shadedColors.row(index0) + bary[1] * shadedColors.row(index1) + bary[2] * shadedColors.row(index2);
            out.image.at<cv::Vec3f>(y, x) = cv::Vec3f(colour(0), colour(1), colour(2));
        }
    }
}


// ── Step 4: z-buffer rasterizer ──────────────────────────────────────────────
// For each front-facing triangle we walk the pixels in its 2D bounding box,
// compute barycentric weights via edge functions, and keep the nearest
// fragment per pixel (z-buffer). The G-buffer we write — triangle index +
// perspective-correct barycentric weights — is what makes the photometric
// term differentiable later (a pixel colour is a linear blend of 3 vertices).
void Renderer::rasterize(const proj::Pixels&      uv,
                         const Eigen::MatrixX3f&  V_cam,
                         const std::vector<bool>& frontFacing,
                         RenderOutput&            out) const
{
    for (int f = 0; f < triangles_.rows(); ++f) {
        if (!frontFacing[f]) continue;

        const int vertex0 = triangles_(f, 0);
        const int vertex1 = triangles_(f, 1);
        const int vertex2 = triangles_(f, 2);

        // Projected pixel positions
        const float x0 = uv(vertex0, 0), y0 = uv(vertex0, 1);
        const float x1 = uv(vertex1, 0), y1 = uv(vertex1, 1);
        const float x2 = uv(vertex2, 0), y2 = uv(vertex2, 1);

        // Camera-frame depth of each vertex (for perspective correction below).
        const float z0 = V_cam(vertex0, 2);
        const float z1 = V_cam(vertex1, 2);
        const float z2 = V_cam(vertex2, 2);

        // Skip if any vertex is behind / on the camera plane.
        if (z0 <= 1e-6f || z1 <= 1e-6f || z2 <= 1e-6f) continue;

        // 2D bounding box, clipped to the image.
        int xmin = static_cast<int>(std::floor(std::min({x0, x1, x2})));
        int xmax = static_cast<int>(std::ceil (std::max({x0, x1, x2})));
        int ymin = static_cast<int>(std::floor(std::min({y0, y1, y2})));
        int ymax = static_cast<int>(std::ceil (std::max({y0, y1, y2})));
        xmin = std::max(0, xmin);  
        xmax = std::min(W_ - 1, xmax);
        ymin = std::max(0, ymin);  
        ymax = std::min(H_ - 1, ymax);
        if (xmin > xmax || ymin > ymax) continue;     // fully off-screen

        // Edge-function denominator = 2× signed triangle area in screen space.
        // (Same quantity as the culling n_z, just reused here for barycentrics.)
        const float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
        if (std::abs(denom) < 1e-12f) continue;        // degenerate triangle
        const float invDenom = 1.0f / denom;

        for (int y = ymin; y <= ymax; ++y) {
            for (int x = xmin; x <= xmax; ++x) {
                // Sample at the pixel centre.
                const float px = x + 0.5f;
                const float py = y + 0.5f;

                // Screen-space barycentric weights (linear in pixel coords).
                const float w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) * invDenom;
                const float w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) * invDenom;
                const float w2 = 1.0f - w0 - w1;

                // Inside test: all weights non-negative.
                if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;

                // ── TODO (a): perspective-correct depth ──────────────────────
                // Screen-space barycentrics are NOT correct for interpolating
                // attributes on a perspective-projected triangle. But 1/z IS
                // linear in screen space, so:

                const float inv_z = w0 / z0 + w1 / z1 + w2 / z2;
                const float z_pixel = 1.0f / inv_z;

                // Perspective-correct barycentric weights: b_k = (w_k/z_k)/inv_z.
                // Since z_pixel = 1/inv_z, this is (w_k/z_k)*z_pixel.
                const float b0 = w0 / z0 * z_pixel;
                const float b1 = w1 / z1 * z_pixel;
                const float b2 = w2 / z2 * z_pixel;

                // Z-buffer test + write: keep the nearest fragment per pixel.
                if (z_pixel < out.depth.at<float>(y, x)) {
                    out.depth.at<float>(y, x) = z_pixel;
                    out.triIdx.at<int>(y, x) = f;
                    out.bary.at<cv::Vec3f>(y, x) = cv::Vec3f(b0, b1, b2);
                    out.mask.at<uchar>(y, x) = 255;
                }
            }
        }
    }
}

// ── Step 3: back-face culling (camera frame) ─────────────────────────────────
// A triangle is front-facing when its outward normal points back toward the
// camera. In the OpenCV camera frame the camera looks down +Z, so a triangle
// faces us when the z-component of its (camera-space) normal is negative.
//
// HINT, not the full solution:
//   • for each triangle f, grab its three camera-frame vertices via
//     V_cam.row(triangles(f, k))
//   • you only need the z-component of (v1 - v0) × (v2 - v0) — write out that
//     one cross-product term by hand instead of computing the whole vector
//   • push_back(n_z < 0) so the result lines up 1:1 with the triangle list
//
// (This mirrors backface_mask in the Python renderer; note it is a *camera-
//  frame* test, not a screen-space winding test — the latter breaks under the
//  BFM_TO_CAM Y-flip.)
std::vector<bool> Renderer::backfaceMask(const Eigen::MatrixX3f& V_cam,
                                         const Eigen::MatrixX3i& triangles)
{
    std::vector<bool> frontFacing;
    frontFacing.reserve(triangles.rows());

    for (int f = 0; f < triangles.rows(); ++f) {
        const Eigen::Vector3f v0 = V_cam.row(triangles(f, 0));
        const Eigen::Vector3f v1 = V_cam.row(triangles(f, 1));
        const Eigen::Vector3f v2 = V_cam.row(triangles(f, 2));

        // z-component of (v1 - v0) × (v2 - v0); front-facing when negative
        const float n_z = (v1.x() - v0.x()) * (v2.y() - v0.y()) -
                          (v1.y() - v0.y()) * (v2.x() - v0.x());
        frontFacing.push_back(n_z < 0);
    }

    return frontFacing;
}

// ── area-weighted per-vertex normals ─────────────────────────────────────────
// 1. For each triangle f: face_normal = (v1-v0) × (v2-v0)
//    The magnitude |cross| equals 2·triangle_area, so leaving it un-normalised
//    means bigger faces contribute more, providing a area based weighting.
// 2. Accumulate onto each of the triangle's 3 vertices.
// 3. Normalise each vertex's accumulated sum to unit length.
Eigen::MatrixX3f Renderer::computeNormals(const Eigen::MatrixX3f& shape,
                                          const Eigen::MatrixX3i& triangles)
{
    Eigen::MatrixX3f normals = Eigen::MatrixX3f::Zero(shape.rows(), 3);

    for (int f = 0; f < triangles.rows(); ++f) {
        const int i0 = triangles(f, 0);
        const int i1 = triangles(f, 1);
        const int i2 = triangles(f, 2);

        const Eigen::Vector3f v0 = shape.row(i0);
        const Eigen::Vector3f v1 = shape.row(i1);
        const Eigen::Vector3f v2 = shape.row(i2);

        const Eigen::Vector3f faceNormal = (v1 - v0).cross(v2 - v0);

        normals.row(i0) += faceNormal;
        normals.row(i1) += faceNormal;
        normals.row(i2) += faceNormal;
    }

    for (int i = 0; i < normals.rows(); ++i) {
        const float n = normals.row(i).norm();
        if (n > 1e-12f) normals.row(i) /= n; //Prevent degenerate vertices
    }
    return normals;
}
