#pragma once
#include <Eigen/Dense>

// Camera / projection helpers shared by the renderer, the landmark term, and
// the debug overlays in main. Everything here is pure geometry — no OpenCV,
// no rasterisation.
namespace proj {

// (N×2) pixel buffer type used throughout.
using Pixels = Eigen::Matrix<float, Eigen::Dynamic, 2>;

// Static BFM→OpenCV-camera alignment: BFM is +Y-up / -Z-forward, OpenCV is
// +Y-down / +Z-forward, so we flip Y and Z. This composes with the head pose
// R everywhere, so callers always pass R in the OpenCV camera convention.
extern const Eigen::Matrix3f BFM_TO_CAM;

// Full camera rotation = head pose composed with the static alignment.
Eigen::Matrix3f cameraRotation(const Eigen::Matrix3f& R);

// Model-frame vertices → camera frame:  V_cam = (R · M_align) · Vᵀ + t
Eigen::MatrixX3f toCameraFrame(const Eigen::MatrixX3f& V,
                               const Eigen::Matrix3f&  R,
                               const Eigen::Vector3f&  t);

// BFModel-frame normals → camera frame (rotation only, no translation).
Eigen::MatrixX3f normalsToCameraFrame(const Eigen::MatrixX3f& N,
                                      const Eigen::Matrix3f&  R);

// Pinhole projection of camera-frame points. Returns (N×2) pixels;
// vertices with z ≤ eps are marked (-1, -1) (behind / on the camera plane).
Pixels project(const Eigen::MatrixX3f& V_cam, const Eigen::Matrix3f& K);

// Convenience: model frame → pixels in one call (transform + project).
Pixels projectMesh(const Eigen::MatrixX3f& V,
                   const Eigen::Matrix3f&  R,
                   const Eigen::Vector3f&  t,
                   const Eigen::Matrix3f&  K);

// Approximate pinhole intrinsics from a horizontal field of view — handy for
// synthetic render targets when you don't have a calibrated camera:
//   fx = fy = width / (2·tan(hfov/2)),  cx = width/2,  cy = height/2
Eigen::Matrix3f defaultIntrinsics(int width, int height, float hfov_deg = 60.0f);

}
