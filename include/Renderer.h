#pragma once
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "ProjectionUtils.h"

// the CPU software renderer. give it a posed face plus lighting and it draws
// the face into an image. it also returns a little side buffer saying which
// triangle and blend weights landed on each pixel, which is what lets the
// photometric fit stay differentiable

// everything that changes per frame. the triangles are not here since they
// never change so they live in the Renderer itself
struct RenderInput {
    Eigen::MatrixX3f shape;          // (N,3) vertex positions
    Eigen::MatrixX3f albedo;         // (N,3) per-vertex colour
    Eigen::Matrix3f  R;              // head rotation
    Eigen::Vector3f  t;              // head translation
    Eigen::Matrix3f  K;              // camera intrinsics
    Eigen::Matrix<float, 9, 3> sh;   // lighting, 9 SH coeffs per channel
};

// what comes back. image/depth/mask are the picture, triIdx and bary are the
// g-buffer the photometric term reads
struct RenderOutput {
    cv::Mat image;   // CV_32FC3 RGB in [0,1]
    cv::Mat depth;   // CV_32F camera Z in mm, +inf where empty
    cv::Mat mask;    // CV_8U 255 where a triangle was drawn
    cv::Mat triIdx;  // CV_32S which triangle covers each pixel, -1 if none
    cv::Mat bary;    // CV_32FC3 that triangle's blend weights for the pixel
};

class Renderer {
public:
    // triangles and image size never change so we set them once here
    Renderer(int height, int width, Eigen::MatrixX3i triangles);

    RenderOutput render(const RenderInput& in) const;

    const Eigen::MatrixX3i& triangles() const { return triangles_; }

    // area-weighted unit normals per vertex. lives here but is also handy
    // outside when we want to save normals next to a mesh
    static Eigen::MatrixX3f computeNormals(const Eigen::MatrixX3f& shape,
                                           const Eigen::MatrixX3i& triangles);

private:
    int H_;
    int W_;
    Eigen::MatrixX3i triangles_;

    // per triangle flag, true means it faces the camera so we draw it
    static std::vector<bool> backfaceMask(const Eigen::MatrixX3f& V_cam,
                                          const Eigen::MatrixX3i& triangles);

    // walk every front-facing triangle and fill in the g-buffer
    void rasterize(const proj::Pixels&       uv,
                   const Eigen::MatrixX3f&   V_cam,
                   const std::vector<bool>&  frontFacing,
                   RenderOutput&             out) const;

    // blend the shaded vertex colours through the g-buffer into out.image
    void interpolateShading(const Eigen::MatrixX3f& shadedColors,
                            RenderOutput&           out) const;
};
