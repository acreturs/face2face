#pragma once
#include <Eigen/Dense>

// simple lighting for the renderer. it assumes a smooth far-away light so the
// brightness at a point only depends on which way that point faces. we keep the
// light as 9 spherical-harmonics numbers per colour channel and the final pixel
// is just albedo times this shading. it is linear in those 9 numbers which is
// why we can recover the light with plain least squares
namespace light {

using Vector9f = Eigen::Matrix<float, 9, 1>;
using SHCoeffs = Eigen::Matrix<float, 9, 3>;   // 9 coeffs per RGB channel

// the 9 basis values for a given surface normal
Vector9f shBasis(const Eigen::Vector3f& n);

// a neutral white light so the very first render is not black
SHCoeffs defaultWhite();

// shade every vertex, gives back albedo times the light clamped to [0,1]
Eigen::MatrixX3f shadeVertices(const Eigen::MatrixX3f& albedo,
                               const Eigen::MatrixX3f& normalsCam,
                               const SHCoeffs&         sh);

}  // namespace light
