#pragma once
#include <Eigen/Dense>

// Spherical-harmonics (order-2) lighting — Step 5 of the renderer.
//
// Under the Lambertian + distant-smooth-light assumptions, the shading at a
// surface point depends only on its normal n and can be written as a linear
// combination of 9 spherical-harmonics basis functions evaluated at n:
//
//     shading(n) = Σ_{i=0..8}  γ_i · B_i(n)             (per colour channel)
//     pixel_colour = albedo ⊙ shading                  (⊙ = element-wise)
//
// γ is a 9×3 matrix (9 coeffs per RGB channel). It is one of the parameters we
// optimise; because shading is *linear* in γ, fitting it is a linear problem.
namespace light {

using Vector9f = Eigen::Matrix<float, 9, 1>;
using SHCoeffs = Eigen::Matrix<float, 9, 3>;   // 9 basis × RGB

// Evaluate the 9 order-2 real SH basis functions at a unit normal n.
// Returns [B0(n) … B8(n)]ᵀ.
Vector9f shBasis(const Eigen::Vector3f& n);

// A neutral white light (ambient-dominant) — useful as a default so the first
// render isn't black. Sets the constant (DC) band on all three channels.
SHCoeffs defaultWhite();

// Per-vertex shaded colours = albedo ⊙ (B(nᵢ)·γ), clamped to [0,1].
//   albedo     : (N,3) per-vertex RGB in [0,1]
//   normalsCam : (N,3) unit normals in the CAMERA frame
//   sh         : (9,3) SH coefficients
// Returns (N,3) shaded colours.
Eigen::MatrixX3f shadeVertices(const Eigen::MatrixX3f& albedo,
                               const Eigen::MatrixX3f& normalsCam,
                               const SHCoeffs&         sh);

}  // namespace light
