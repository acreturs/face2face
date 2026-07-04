#pragma once

#include <Eigen/Dense>

#include <functional>
#include <string>
#include <vector>

struct LandmarkObservation {
    int vertexIndex;
    Eigen::Vector2d imagePoint;
};

struct PoseParameters {
    Eigen::Vector3d angleAxis = Eigen::Vector3d::Zero();
    Eigen::Vector3d translation = Eigen::Vector3d(0.0, 0.0, 350.0);

    Eigen::Matrix3f rotationMatrix() const;
};

constexpr int kShapeCoefficientCount = 30;

struct FitParameters {
    PoseParameters pose;

    Eigen::VectorXd shapeCoefficients =
        Eigen::VectorXd::Zero(kShapeCoefficientCount);
};

// Called after every outer ICP iteration of fitDense (e.g. to render progress).
using DenseIterationCallback =
    std::function<void(int iteration, const FitParameters& current, double rmseMM)>;

std::vector<LandmarkObservation> loadLandmarkObservations(
    const std::string& path
);

class CeresFitter {
public:
    // Stage 1: pose only. zMin/zMax bound the face depth (mm).
    static PoseParameters fitPose(
        const Eigen::MatrixX3f& shape,
        const std::vector<LandmarkObservation>& observations,
        const Eigen::Matrix3f& intrinsics,
        const PoseParameters& initialPose,
        double zMin = 100.0,
        double zMax = 10000.0
    );
    // Stage 2: pose + shape from 2D landmarks (+ L2 shape prior).
    static FitParameters fitPoseAndShape(
      const Eigen::MatrixX3f& meanShape,
      const Eigen::MatrixXf& shapeBasis,
      const Eigen::VectorXf& shapeSigma,
      const std::vector<LandmarkObservation>& observations,
      const Eigen::Matrix3f& intrinsics,
      const PoseParameters& initialPose,
      double regularizationWeight = 100.0,
      double zMin = 200.0,
      double zMax = 600.0
  );

    // Dense fit: outer ICP loop (re-find nearest-vertex correspondences →
    // Ceres-solve pose+shape → repeat) against a camera-frame depth cloud (mm).
    static FitParameters fitDense(
        const Eigen::MatrixX3f&              meanShape,
        const Eigen::MatrixXf&               shapeBasis,
        const Eigen::VectorXf&               shapeSigma,
        const std::vector<Eigen::Vector3d>&  targetCloud,
        const PoseParameters&                initialPose,
        double                               regularizationWeight = 50.0,
        int                                  numIcpIterations     = 15,
        double                               trimPercentile       = 80.0,
        int                                  vertexStride         = 8,
        double                               pointToPlaneWeight   = 1.0,
        const DenseIterationCallback&        onIteration          = nullptr
    );
};