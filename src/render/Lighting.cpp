// spherical harmonics lighting
// builds the SH basis and shades each vertex with it
#include "Lighting.h"

#include <algorithm>

namespace light {

// order 2 real spherical harmonics basis
// the numbers are the SH basis functions up to l=2 baked in
// index 0 is the constant ambient band, 1..3 are the linear light direction
// and 4..8 are the quadratic softer variation
// n must be a unit vector
Vector9f shBasis(const Eigen::Vector3f& n)
{
    const float x = n.x(), y = n.y(), z = n.z();
    Vector9f b;
    b(0) = 0.282095f;                         // 1/(2√π)
    b(1) = 0.488603f * y;
    b(2) = 0.488603f * z;
    b(3) = 0.488603f * x;
    b(4) = 1.092548f * x * y;
    b(5) = 1.092548f * y * z;
    b(6) = 0.315392f * (3.0f * z * z - 1.0f);
    b(7) = 1.092548f * x * z;
    b(8) = 0.546274f * (x * x - y * y);
    return b;
}

SHCoeffs defaultWhite()
{
    SHCoeffs sh = SHCoeffs::Zero();
    // only the constant band so the light is flat ambient
    // the factor cancels the small DC basis value B0 so shading lands near 1.0
    sh.row(0).setConstant(1.0f / 0.282095f);
    return sh;
}

// shade each vertex using the SH light
Eigen::MatrixX3f shadeVertices(const Eigen::MatrixX3f& albedo,
                               const Eigen::MatrixX3f& normalsCam,
                               const SHCoeffs&         sh)
{
    const int N = static_cast<int>(albedo.rows());
    Eigen::MatrixX3f out(N, 3);

    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3f n = normalsCam.row(i);
        const Vector9f b = shBasis(n);

        // b^T * sh gives a 1x3 row of shading per colour channel
        const Eigen::RowVector3f shading = b.transpose() * sh;
        out.row(i) = albedo.row(i)
                         .cwiseProduct(shading)
                         .cwiseMax(0.0f)
                         .cwiseMin(1.0f);
    }
    return out;
}

}  // namespace light
