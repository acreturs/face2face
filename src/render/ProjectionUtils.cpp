// helpers to move the face mesh into the camera frame and project it
// also holds the axis flip between the BFM model and the camera
#include "ProjectionUtils.h"

#define _USE_MATH_DEFINES
#include <cmath>

namespace proj {

// the BFM model and the camera disagree on which way is up and forward
// so we flip Y and Z to line them up
const Eigen::Matrix3f BFM_TO_CAM = (Eigen::Matrix3f() <<
    1,  0,  0,
    0, -1,  0,
    0,  0, -1).finished();

Eigen::Matrix3f cameraRotation(const Eigen::Matrix3f& R)
{
    return R * BFM_TO_CAM;
}

Eigen::MatrixX3f toCameraFrame(const Eigen::MatrixX3f& V,
                               const Eigen::Matrix3f&  R,
                               const Eigen::Vector3f&  t)
{
    const Eigen::Matrix3f R_total = cameraRotation(R);
    return (V * R_total.transpose()).rowwise() + t.transpose();
}

Eigen::MatrixX3f normalsToCameraFrame(const Eigen::MatrixX3f& N,
                                      const Eigen::Matrix3f&  R)
{
    // normals only rotate, no translation
    // for a pure rotation the inverse transpose is just the rotation so this works
    const Eigen::Matrix3f R_total = cameraRotation(R);
    return N * R_total.transpose();
}

// pinhole project the camera space vertices down to pixels
Pixels project(const Eigen::MatrixX3f& V_cam, const Eigen::Matrix3f& K)
{
    Pixels uv(V_cam.rows(), 2);
    for (int i = 0; i < V_cam.rows(); ++i) {
        const float z = V_cam(i, 2);
        if (z <= 1e-3f) { uv.row(i) << -1.f, -1.f; continue; }
        uv(i, 0) = K(0, 0) * V_cam(i, 0) / z + K(0, 2);
        uv(i, 1) = K(1, 1) * V_cam(i, 1) / z + K(1, 2);
    }
    return uv;
}

Pixels projectMesh(const Eigen::MatrixX3f& V,
                   const Eigen::Matrix3f&  R,
                   const Eigen::Vector3f&  t,
                   const Eigen::Matrix3f&  K)
{
    return project(toCameraFrame(V, R, t), K);
}

// build a camera matrix K from the horizontal field of view
Eigen::Matrix3f defaultIntrinsics(int width, int height, float hfov_deg)
{
    const float hfov = hfov_deg * static_cast<float>(M_PI) / 180.0f;
    const float f    = width / (2.0f * std::tan(hfov / 2.0f));
    Eigen::Matrix3f K;
    K << f, 0, width  / 2.0f,
         0, f, height / 2.0f,
         0, 0, 1;
    return K;
}

}  // namespace proj
