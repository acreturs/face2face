#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>

// loads the Basel Face Model from its h5 file and hands out the pieces the
// rest of the pipeline needs the mean face the PCA bases and the triangles

// one named landmark on the mean mesh and the vertex it sits on
struct BFMLandmark {
    std::string name;       // e.g. "right.eye.corner_outer"
    int         vertex_idx; // index into the (N,3) vertex array
};

class BFMLoader
{
public:
    explicit BFMLoader(const std::string& path);

    // geometry (identity + expression)
    Eigen::MatrixX3f mean_shape() const;                          // alpha = 0
    Eigen::MatrixX3f shape(const Eigen::VectorXf& alpha) const;   // identity only
    // full face once you also feed in expression coeffs
    Eigen::MatrixX3f shape(const Eigen::VectorXf& alpha,
                           const Eigen::VectorXf& delta) const;

    // albedo (skin colour)
    Eigen::MatrixX3f albedo() const;                              // mean albedo in [0,1]
    Eigen::MatrixX3f albedo(const Eigen::VectorXf& beta) const;   // mean + basis · (beta .* sigma)

    const Eigen::MatrixX3i& faces() const { return triangles; }   // (M,3) triangles

    // raw bases and sigmas the optimiser needs for the priors and Jacobians
    const Eigen::VectorXf& shape_sigma() const { return shape_std; }         // σ_id (K_id)
    const Eigen::VectorXf& expr_sigma()  const { return expr_std;  }         // σ_exp (K_exp)
    const Eigen::VectorXf& color_sigma() const { return color_std; }         // σ_alb (K_alb)
    const Eigen::MatrixXf& shape_basis_raw() const { return shape_basis; }   // (3N, K_id)
    const Eigen::MatrixXf& expr_basis_raw()  const { return expr_basis;  }   // (3N, K_exp)
    const Eigen::MatrixXf& color_basis_raw() const { return color_basis; }   // (3N, K_alb)

    const std::vector<BFMLandmark>& landmarks() const { return landmarks_; }
    int landmark_index(const std::string& name) const;   // -1 if not found

    // prints what got loaded so you can eyeball the shapes
    void summariseBFM(const std::string& path) const;

private:
    // identity (shape) PCA
    Eigen::VectorXf  shape_mean;    // (3N,)
    Eigen::MatrixXf  shape_basis;   // (3N, K_id)   K_id = 199
    Eigen::VectorXf  shape_std;     // (K_id,)      σ_id = sqrt(variance)

    // expression PCA, added on top of identity
    Eigen::VectorXf  expr_mean;     // (3N,)
    Eigen::MatrixXf  expr_basis;    // (3N, K_exp)  K_exp = 100
    Eigen::VectorXf  expr_std;      // (K_exp,)

    // albedo (colour) PCA
    Eigen::VectorXf  color_mean;    // (3N,)
    Eigen::MatrixXf  color_basis;   // (3N, K_alb)  K_alb = 199
    Eigen::VectorXf  color_std;     // (K_alb,)

    Eigen::MatrixX3i triangles;     // (M, 3)
    std::vector<BFMLandmark> landmarks_;
};
