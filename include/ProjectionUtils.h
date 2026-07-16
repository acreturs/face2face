#pragma once
#include <Eigen/Dense>

// little math helpers for turning 3D face points into 2D image pixels
// just geometry in here so no image or rendering code
namespace proj {

// a list of 2D pixel positions where each row holds one point x and y
using Pixels = Eigen::Matrix<float, Eigen::Dynamic, 2>;

// the BFM model and the camera disagree on which way is up and forward
// so we flip Y and Z to line them up
extern const Eigen::Matrix3f BFM_TO_CAM;

// head rotation combined with the flip above
Eigen::Matrix3f cameraRotation(const Eigen::Matrix3f& R);

// move the face vertices from model space into the camera view
Eigen::MatrixX3f toCameraFrame(const Eigen::MatrixX3f& V,
                               const Eigen::Matrix3f&  R,
                               const Eigen::Vector3f&  t);

// same idea for normals but we only rotate them and never shift them
Eigen::MatrixX3f normalsToCameraFrame(const Eigen::MatrixX3f& N,
                                      const Eigen::Matrix3f&  R);

// turn 3D camera points into 2D pixels with a normal pinhole camera
// anything behind the camera comes back as (-1 -1)
Pixels project(const Eigen::MatrixX3f& V_cam, const Eigen::Matrix3f& K);

Pixels projectMesh(const Eigen::MatrixX3f& V,
                   const Eigen::Matrix3f&  R,
                   const Eigen::Vector3f&  t,
                   const Eigen::Matrix3f&  K);

// guess a camera matrix from a field of view angle
// handy when we have no real calibration like a random webcam
Eigen::Matrix3f defaultIntrinsics(int width, int height, float hfov_deg = 60.0f);

}
