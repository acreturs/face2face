#pragma once

#include <Eigen/Dense>

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

constexpr int kShapeCoefficientCount = 5;

struct FitParameters {
    PoseParameters pose;

    Eigen::VectorXd shapeCoefficients =
        Eigen::VectorXd::Zero(kShapeCoefficientCount);
};

std::vector<LandmarkObservation> loadLandmarkObservations(
    const std::string& path
);

class CeresFitter {
public:
    static PoseParameters fitPose(
        const Eigen::MatrixX3f& shape,
        const std::vector<LandmarkObservation>& observations,
        const Eigen::Matrix3f& intrinsics,
        const PoseParameters& initialPose
    );
    static FitParameters fitPoseAndShape(
      const Eigen::MatrixX3f& meanShape,
      const Eigen::MatrixXf& shapeBasis,
      const Eigen::VectorXf& shapeSigma,
      const std::vector<LandmarkObservation>& observations,
      const Eigen::Matrix3f& intrinsics,
      const PoseParameters& initialPose,
      double regularizationWeight = 100.0
  );
};