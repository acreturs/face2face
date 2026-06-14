#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>

// One named BFM landmark with its associated vertex index on the mean mesh.
struct BFMLandmark {
    std::string name;       // e.g. "right.eye.corner_outer"
    int         vertex_idx; // index into the (N,3) vertex array
};

class BFMLoader
{
public:
    explicit BFMLoader(const std::string& path);

    // ── Geometry (identity + expression) ────────────────────────────────────
    Eigen::MatrixX3f mean_shape() const;                                                 // alpha = 0
    Eigen::MatrixX3f shape(const Eigen::VectorXf& alpha) const;                          // identity only
    Eigen::MatrixX3f shape(const Eigen::VectorXf& alpha,
                           const Eigen::VectorXf& delta) const;                          // identity + expression

    // ── Albedo ──────────────────────────────────────────────────────────────
    Eigen::MatrixX3f albedo() const;                                                     // mean albedo, [0,1]
    Eigen::MatrixX3f albedo(const Eigen::VectorXf& beta) const;                          // mean + basis · (beta .* sigma)

    // ── Topology ────────────────────────────────────────────────────────────
    const Eigen::MatrixX3i& faces() const { return triangles; }                          // (M,3) triangles

    // ── Accessors for the optimiser (Mahalanobis priors & PCA Jacobians) ────
    const Eigen::VectorXf& shape_sigma() const { return shape_std; }                     // σ_id (K_id)
    const Eigen::VectorXf& expr_sigma()  const { return expr_std;  }                     // σ_exp (K_exp)
    const Eigen::VectorXf& color_sigma() const { return color_std; }                     // σ_alb (K_alb)
    const Eigen::MatrixXf& shape_basis_raw() const { return shape_basis; }               // (3N, K_id)
    const Eigen::MatrixXf& expr_basis_raw()  const { return expr_basis;  }               // (3N, K_exp)
    const Eigen::MatrixXf& color_basis_raw() const { return color_basis; }               // (3N, K_alb)

    // ── Named landmarks (for the BFM↔dlib mapping used by E_sparse) ─────────
    const std::vector<BFMLandmark>& landmarks() const { return landmarks_; }
    int landmark_index(const std::string& name) const;                                   // -1 if not found

    // ── Debug ───────────────────────────────────────────────────────────────
    void summariseBFM(const std::string& path) const;

private:
    // Identity (shape) PCA
    Eigen::VectorXf  shape_mean;    // (3N,)
    Eigen::MatrixXf  shape_basis;   // (3N, K_id)   K_id = 199
    Eigen::VectorXf  shape_std;     // (K_id,)      σ_id = sqrt(variance)

    // Expression PCA (additive, identity-independent)
    Eigen::VectorXf  expr_mean;     // (3N,)
    Eigen::MatrixXf  expr_basis;    // (3N, K_exp)  K_exp = 100
    Eigen::VectorXf  expr_std;      // (K_exp,)

    // Albedo (color) PCA
    Eigen::VectorXf  color_mean;    // (3N,)
    Eigen::MatrixXf  color_basis;   // (3N, K_alb)  K_alb = 199
    Eigen::VectorXf  color_std;     // (K_alb,)

    // Topology + landmarks
    Eigen::MatrixX3i triangles;     // (M, 3)
    std::vector<BFMLandmark> landmarks_;
};
