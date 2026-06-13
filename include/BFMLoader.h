#pragma once
#include <string>
#include <Eigen/Dense>

class BFMLoader
{
public:
    explicit BFMLoader(const std::string& path);

    Eigen::MatrixX3f mean_shape() const;                         // average face (alpha = 0)
    Eigen::MatrixX3f shape(const Eigen::VectorXf& alpha) const;  // mean + basis * (alpha .* sigma)
    Eigen::MatrixX3f albedo() const;                             // per-vertex RGB in [0,1]
    const Eigen::MatrixX3i& faces() const { return triangles; }  // triangle vertex indices (M,3)

private:
    Eigen::VectorXf  shape_mean;   // (3N,)
    Eigen::MatrixXf  shape_basis;  // (3N, 199)
    Eigen::VectorXf  shape_std;    // (199,) = sqrt(variance)
    Eigen::VectorXf  color_mean;   // (3N,)
    Eigen::MatrixX3i triangles;    // (M, 3)
};
