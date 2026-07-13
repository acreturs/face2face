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
// 30, not 80: only ~10 mouth/brow landmarks constrain expression, and BFM
// expression modes 30–80 add just ~4% of variance (92%→99%) — 50 nearly-
// unobservable DOF the solver fills with a high-norm, jittery, asymmetric
// combination (‖δ‖≈5–7 for a near-closed mouth). Optimising the first 30 keeps
// 92% of the expression range while removing the instability at its source.
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

// ── Non-rigid model-based bundling (Face2Face §6) ────────────────────────────
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

    // Stage 2b: pose + shape with a CONTOUR (silhouette) term. Observations with
    // vertexIndex >= 0 are fixed named landmarks (interior); observations with
    // vertexIndex == -1 are contour points (jawline) with NO fixed model vertex.
    // Because the BFM has no jaw landmarks and the silhouette vertex slides with
    // pose, we run an ICP-style outer loop: each iteration re-assigns every
    // contour point to the nearest projected MODEL SILHOUETTE vertex (a vertex
    // seen near-edge-on, |n_cam.z| small, on the matching side), then Ceres-solves
    // pose+shape. This is what constrains face WIDTH/outline — the interior
    // landmarks cannot. Use a lower regularizationWeight than the interior-only
    // fit so the identity can actually widen.
    // Also solves EXPRESSION (delta): the expr basis is added to every landmark
    // residual as a 4th parameter block, so mouth/brow/lip landmarks move the
    // expression. Returned in result.exprCoefficients.
    //
    // FULL fit: if `targetCloud` is non-null (its points expressed in THIS fit's
    // camera frame), a depth ICP term (point-to-point + point-to-plane) is added
    // each outer iteration, so pose + identity + expression are solved jointly
    // from landmarks, jaw-contour AND depth. Pass nullptr for the RGB-only fit.
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
        // ── video tracking ──  seed identity/expression (warm start) and freeze
        // identity so per-frame tracking only solves pose + expression. Empty
        // observations are allowed when a depth cloud drives the fit.
        const Eigen::VectorXd&                   initialIdentity      = Eigen::VectorXd(),
        const Eigen::VectorXd&                   initialExpr          = Eigen::VectorXd(),
        bool                                     optimizeIdentity     = true,
        // ── camera focal estimation (realtime plan, Phase 3) ──
        // When optimizeFocal, a single focal parameter (fx = fy; principal
        // point fixed) joins the landmark/contour residuals; *focalInOut seeds
        // it (else intrinsics(0,0)) and receives the estimate. The identity
        // prior anchors the metric face size, which is what disambiguates
        // focal from distance. Only sensible during personalisation.
        bool                                     optimizeFocal        = false,
        double*                                  focalInOut           = nullptr,
        // ── temporal expression prior (tracking) ──  weight on
        // ‖δ − initialExpr‖²: damps frame-to-frame expression jitter IN the
        // solve without fighting a held articulation the way the zero-anchored
        // prior does. 0 = off (personalise / single-frame fits).
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

    // Non-rigid model-based bundling (Face2Face §6). Jointly solves ONE shared
    // identity α with per-frame {pose, expression} over several keyframes at
    // different viewing angles, in a single block-dense Ceres problem. Each
    // keyframe contributes interior-landmark reprojection + jaw-contour (sliding
    // silhouette, re-matched each outer iteration) + optional depth ICP, all
    // pointing at the shared α. Multi-view parallax + shared-identity
    // consistency is what resolves the depth ambiguity that a single view
    // cannot, so the identity reg can be near-zero without over-fitting.
    // Returns the shared α and each keyframe's pose/expr.
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
        // For video tracking: freeze pose too, so the call only estimates
        // lighting/albedo at a pose fixed by the (more reliable) depth fit.
        bool                              optimizePose      = true,
        const DenseIterationCallback&     onIteration       = nullptr,
        // Working-resolution cap (image + intrinsics are downscaled together).
        // The realtime path calls this per pyramid level (e.g. 100 then 200).
        int                               maxImageWidth     = 400,
        // ── JOINT E_col + E_lan (Face2Face Eq. 3) ──  When `landmarks` is
        // non-null and `landmarkWeight` > 0, the interior landmark reprojection
        // residuals are added to the per-pixel photometric SHAPE solve, on the
        // same pose+shape blocks. The dense photometric alone is
        // appearance-limited and shape-from-shading-ambiguous; the landmark
        // term anchors the shape inside the solve (paper w_lan ≫ w_col), which
        // is what makes a LOW shapeRegWeight safe — the coupling the paper
        // relies on. Default (nullptr / 0) reproduces the old behaviour exactly,
        // so existing callers (tracking lighting refresh) are unaffected.
        const std::vector<LandmarkObservation>* landmarks   = nullptr,
        double                            landmarkWeight    = 0.0
    );
};