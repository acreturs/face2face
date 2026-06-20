#include "Lighting.h"

#include <algorithm>

namespace light {

// ── order-2 real spherical-harmonics basis ───────────────────────────────────
// These constants are the real SH basis functions Y_lm up to l=2, pre-baked.
// Index → band:
//   0          : l=0  (constant / ambient)
//   1,2,3      : l=1  (linear in x,y,z — overall light direction)
//   4,5,6,7,8  : l=2  (quadratic — softer directional variation)
// n MUST be a unit vector.
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
    // Constant band only → flat ambient light. The factor compensates for the
    // small DC basis value (B0 ≈ 0.2821) so shading lands near ~1.0.
    sh.row(0).setConstant(1.0f / 0.282095f);
    return sh;
}

// ── per-vertex shading ───────────────────────────────────────────────────────
Eigen::MatrixX3f shadeVertices(const Eigen::MatrixX3f& albedo,
                               const Eigen::MatrixX3f& normalsCam,
                               const SHCoeffs&         sh)
{
    const int N = static_cast<int>(albedo.rows());
    Eigen::MatrixX3f out(N, 3);

    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3f n = normalsCam.row(i);
        const Vector9f b = shBasis(n);

        // bᵀ·sh = (1×9)·(9×3) = 1×3 row of per-channel shadings.
        const Eigen::RowVector3f shading = b.transpose() * sh;
        out.row(i) = albedo.row(i)
                         .cwiseProduct(shading)
                         .cwiseMax(0.0f)
                         .cwiseMin(1.0f);
    }
    return out;
}

}  // namespace light
