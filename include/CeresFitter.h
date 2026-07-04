#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "Lighting.h"   // light::SHCoeffs, light::defaultWhite()

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
constexpr int kAlbedoCoefficientCount = 30;   // BFM colour (albedo) PCA coeffs

struct FitParameters {
    PoseParameters pose;

    Eigen::VectorXd shapeCoefficients =
        Eigen::VectorXd::Zero(kShapeCoefficientCount);

    // Appearance, estimated by fitPhotometric (ignored by the geometry-only
    // fits). albedoCoefficients is empty ⇒ use the mean albedo; sh defaults to
    // flat white until lighting is estimated.
    Eigen::VectorXd albedoCoefficients;
    light::SHCoeffs sh = light::defaultWhite();
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

    // Photometric (appearance) fit against a single RGB image — the analysis-by-
    // synthesis term. Completely independent of fitDense's depth term: it needs
    // only an RGB image + intrinsics, so it also runs on RGB-only inputs (e.g. an
    // iPhone selfie, where there is no depth to feed the dense term).
    //
    // It alternates three sub-steps each outer iteration, analysis-by-synthesis:
    //   1. estimate SH LIGHTING (linear least-squares): with geometry + albedo
    //      fixed, the rendered colour is linear in the 9×3 SH coeffs, so a
    //      per-channel 9×9 solve recovers the lighting that best explains the
    //      photo.  (skipped if optimizeLighting == false)
    //   2. estimate ALBEDO (linear least-squares): with lighting fixed, the
    //      colour is linear in the BFM colour-basis coeffs β, recovered by a
    //      regularised Kβ×Kβ solve.  (skipped if optimizeAlbedo == false)
    //   3. refine GEOMETRY (Ceres, per-pixel): render() gives the predicted
    //      image + G-buffer (triIdx + perspective-correct bary); each covered
    //      pixel adds a residual — renderedColour(pixel) vs the input image
    //      sampled at the reprojection of that pixel's surface point (the
    //      barycentric blend of its triangle's 3 vertices). The image is sampled
    //      differentiably (Ceres bicubic interpolator), so the residual has a
    //      gradient w.r.t. pose and (if optimizeShape) shape.
    //
    // Steps 1–2 are what make the appearance actually fit; step 3 aligns the
    // geometry. The estimated lighting and albedo come back in the result's
    // `sh` and `albedoCoefficients`. Best run from a sparse (or dense) init.
    //
    // `imageBgr` is a normal OpenCV BGR image (8U or 32F); it is downscaled to a
    // working resolution and normalised to [0,1] internally. `pixelStride`
    // samples every Nth covered pixel in the geometry step.
    static FitParameters fitPhotometric(
        const Eigen::MatrixX3f&           meanShape,
        const Eigen::MatrixXf&            shapeBasis,
        const Eigen::VectorXf&            shapeSigma,
        const Eigen::MatrixX3i&           triangles,
        const Eigen::MatrixX3f&           meanAlbedo,
        const Eigen::MatrixXf&            colorBasis,
        const Eigen::VectorXf&            colorSigma,
        const cv::Mat&                    imageBgr,
        const Eigen::Matrix3f&            intrinsics,
        const FitParameters&              initialFit,
        double                            shapeRegWeight    = 100.0,
        double                            albedoRegWeight   = 30.0,
        int                               numIterations     = 10,
        int                               pixelStride       = 2,
        double                            photometricWeight = 1.0,
        bool                              optimizeShape     = true,
        bool                              optimizeLighting  = true,
        bool                              optimizeAlbedo    = true,
        const DenseIterationCallback&     onIteration       = nullptr
    );
};