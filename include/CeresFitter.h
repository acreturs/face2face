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

constexpr int kShapeCoefficientCount = 180;
constexpr int kAlbedoCoefficientCount = 180;       // BFM colour (albedo) PCA coeffs
constexpr int kExpressionCoefficientCount = 30;   // BFM expression PCA coeffs

struct FitParameters {
    PoseParameters pose;

    Eigen::VectorXd shapeCoefficients =
        Eigen::VectorXd::Zero(kShapeCoefficientCount);

    // Expression coeffs (delta), solved by the contour landmark fit. Empty ⇒
    // neutral face.
    Eigen::VectorXd exprCoefficients;

    // Appearance, estimated by fitPhotometric (ignored by the geometry-only
    // fits). albedoCoefficients is empty ⇒ use the mean albedo; sh defaults to
    // flat white until lighting is estimated.
    Eigen::VectorXd albedoCoefficients;
    light::SHCoeffs sh = light::defaultWhite();
};

// Called after every outer ICP iteration of fitDense (e.g. to render progress).
using DenseIterationCallback =
    std::function<void(int iteration, const FitParameters& current, double rmseMM)>;

// Non-rigid model-based bundling (Face2Face §6)
// One keyframe fed to fitIdentityBundle: its 2D observations, a per-frame pose
// init, and (optionally, RGBD) its depth cloud in THIS keyframe's camera frame.
struct BundleFrame {
    std::vector<LandmarkObservation>          observations;
    PoseParameters                            initialPose;
    const std::vector<Eigen::Vector3d>*       depthCloud = nullptr;  // nullable
};

// Result of the bundle: ONE shared identity, per-frame pose + expression.
struct BundleResult {
    Eigen::VectorXd              identity;   // shared α
    std::vector<PoseParameters>  poses;      // per keyframe
    std::vector<Eigen::VectorXd> exprs;      // per keyframe
};

std::vector<LandmarkObservation> loadLandmarkObservations(
    const std::string& path
);

class CeresFitter {
public:
    // When true (--photo-gpu), fitPhotometric's per-pixel geometry solve runs on
    // the GPU (CUDA finite-difference LM) instead of Ceres. Only in a USE_CUDA
    // build and only when geometry is solved; the Ceres path stays the reference.
    static bool usePhotometricGpu;

    // With usePhotometricGpu on, selects the GPU Jacobian: false → finite
    // differences (--photo-gpu); true → analytic image-gradient × projection ×
    // pose/shape chain (--photo-gpu-analytic).
    static bool photoGpuAnalytic;

    // 0 = use every covered pixel (old behaviour). >0 = per outer iteration,
    // randomly sample this many covered pixels instead. re-sampled each
    // iteration so over the whole fit all pixels get used (--photo-samples K).
    static int photoSamples;

    // when set (--photo-csv path), fitPhotometric appends one row per outer
    // iteration so the convergence can be plotted without scraping stdout.
    static std::string photoCsvPath;

    // Stage 1: pose only. zMin/zMax bound the face depth (mm).
    static PoseParameters fitPose(
        const Eigen::MatrixX3f& shape,
        const std::vector<LandmarkObservation>& observations,
        const Eigen::Matrix3f& intrinsics,
        const PoseParameters& initialPose,
        double zMin = 100.0,
        double zMax = 10000.0
    );
    // Stage 2a: pose + shape from 2D landmarks + shape prior.
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

    // Stage 2b: pose + shape with a contour term. Interior landmarks
    // (vertexIndex >= 0) are fixed; contour points (vertexIndex == -1, jawline)
    // have no fixed model vertex, so an ICP-style outer loop re-assigns each to
    // the nearest edge-on projected silhouette vertex before every Ceres solve.
    static FitParameters fitPoseAndShapeContour(
        const Eigen::MatrixX3f&                  meanShape,
        const Eigen::MatrixXf&                   shapeBasis,
        const Eigen::VectorXf&                   shapeSigma,
        const Eigen::MatrixXf&                   exprBasis,
        const Eigen::VectorXf&                   exprSigma,
        const Eigen::MatrixX3i&                  triangles,
        const std::vector<LandmarkObservation>&  observations,
        const Eigen::Matrix3f&                   intrinsics,
        const PoseParameters&                    initialPose,
        double                                   regularizationWeight = 30.0,
        double                                   exprRegWeight        = 30.0,
        double                                   zMin                 = 200.0,
        double                                   zMax                 = 600.0,
        int                                      numOuterIterations   = 6,
        const std::vector<Eigen::Vector3d>*      targetCloud          = nullptr,
        double                                   depthPointToPlaneWeight = 1.0,
        double                                   depthWeight             = 1.0,
        int                                      depthVertexStride       = 8,
        // video tracking    seed identity/expression (warm start) and freeze
        // identity so per-frame tracking only solves pose + expression. Empty
        // observations are allowed when a depth cloud drives the fit.
        const Eigen::VectorXd&                   initialIdentity      = Eigen::VectorXd(),
        const Eigen::VectorXd&                   initialExpr          = Eigen::VectorXd(),
        bool                                     optimizeIdentity     = true,
        // Camera focal estimation. When optimizeFocal, a single focal (fx=fy,
        // principal point fixed) joins the landmark/contour residuals; *focalInOut
        // seeds and receives it.
        bool                                     optimizeFocal        = false,
        double*                                  focalInOut           = nullptr,
        // Temporal expression prior (tracking): weight on ‖δ − initialExpr‖²,
        // damps frame-to-frame jitter without fighting a held articulation.
        // 0 = off (personalise / single-frame).
        double                                   exprTemporalWeight   = 0.0
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

    // Model-based bundling : jointly solve one shared identity α
    // with per-frame {pose, expression} over several keyframes at different
    // angles, in one Ceres problem. Each keyframe adds interior-landmark
    // reprojection + jaw contour (+ optional depth ICP) pointing at the shared α.
    static BundleResult fitIdentityBundle(
        const Eigen::MatrixX3f&          meanShape,
        const Eigen::MatrixXf&           shapeBasis,
        const Eigen::VectorXf&           shapeSigma,
        const Eigen::MatrixXf&           exprBasis,
        const Eigen::VectorXf&           exprSigma,
        const Eigen::MatrixX3i&          triangles,
        const std::vector<BundleFrame>&  frames,
        const Eigen::Matrix3f&           intrinsics,
        double                           regularizationWeight    = 3.0,
        double                           exprRegWeight           = 30.0,
        double                           zMin                    = 200.0,
        double                           zMax                    = 2000.0,
        int                              numOuterIterations      = 5,
        double                           depthPointToPlaneWeight = 1.0,
        double                           depthWeight             = 1.0,
        int                              depthVertexStride       = 8
    );

    // Photometric (appearance) fit against a single RGB image (analysis-by-
    // synthesis). Independent of the depth term, so it also runs on RGB-only
    // inputs. Each outer iteration alternates three sub-steps:
    //   1. SH lighting — linear least-squares (skipped if !optimizeLighting)
    //   2. albedo β    — regularised linear least-squares (skipped if !optimizeAlbedo)
    //   3. geometry    — per-pixel Ceres solve against render()'s G-buffer, with
    //                    the input sampled differentiably (bicubic) at each
    //                    covered pixel's reprojection.
    // Steps 1–2 fit the appearance, step 3 aligns the geometry. imageBgr is a
    // normal OpenCV BGR image, downscaled to a working resolution internally;
    // pixelStride samples every Nth covered pixel in the geometry step.
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
        // For video tracking: freeze pose too, so the call only estimates
        // lighting/albedo at a pose fixed by the (more reliable) depth fit.
        bool                              optimizePose      = true,
        const DenseIterationCallback&     onIteration       = nullptr,
        // Working-resolution cap (image + intrinsics are downscaled together).
        int                               maxImageWidth     = 400,
        // Joint E_col + E_lan: when landmarks is non-null and
        // landmarkWeight > 0, the interior landmark residuals join the per-pixel
        // photometric shape solve on the same blocks.
        const std::vector<LandmarkObservation>* landmarks   = nullptr,
        double                            landmarkWeight    = 0.0
    );

    // Dense-photometric identity bundle: refine the shared
    // α using the per-pixel E_col from all keyframes jointly
    static Eigen::VectorXd fitIdentityPhotometricBundle(
        const Eigen::MatrixX3f&          meanShape,
        const Eigen::MatrixXf&           shapeBasis,
        const Eigen::VectorXf&           shapeSigma,
        const Eigen::MatrixXf&           exprBasis,
        const Eigen::VectorXf&           exprSigma,
        const Eigen::MatrixX3i&          triangles,
        const Eigen::MatrixX3f&          meanAlbedo,
        const Eigen::MatrixXf&           colorBasis,
        const Eigen::VectorXf&           colorSigma,
        const std::vector<cv::Mat>&      bgrs,
        const std::vector<std::vector<LandmarkObservation>>& observations,
        const std::vector<PoseParameters>&  poses,
        const std::vector<Eigen::VectorXd>& exprs,
        const Eigen::Matrix3f&           intrinsics,
        const Eigen::VectorXd&           alphaInit,
        Eigen::VectorXd&                 betaOut,
        light::SHCoeffs&                 shOut,
        double                           shapeReg       = 5.0,
        double                           albedoReg      = 3.0,
        double                           landmarkWeight = 20.0,
        int                              numIterations  = 6,
        int                              pixelStride    = 2,
        int                              maxImageWidth  = 320
    );
};