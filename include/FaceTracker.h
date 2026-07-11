#pragma once
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <vector>

#include "BFMLoader.h"
#include "CeresFitter.h"
#include "Lighting.h"

// Personalise-then-track state machine, shared by the offline video modes and
// the realtime camera path (--mode live).
//
//   personalise(frame) — the expensive one-off: identity α + expression from
//     landmarks/contour (+ depth if given), then albedo β + lighting via the
//     photometric fit. Optionally also solves the camera FOCAL length (webcam
//     with unknown intrinsics); the identity prior anchors the metric face
//     size, which is what makes focal observable from a single view.
//   track(frame) — per frame: constant-velocity warm start → contour fit with
//     identity frozen (pose + expression) → optional pyramid photometric pose
//     refinement (accepted only if it does not worsen the landmark
//     reprojection) → lighting refresh → EMA smoothing. Detection gating with
//     automatic recovery after repeated rejects.
class FaceTracker {
public:
    struct Config {
        // priors / fit weights
        double sparseReg          = 30.0;
        double exprRegPersonalise = 500.0;
        double exprRegTrack       = 120.0;
        double albedoRegWeight    = 50.0;
        double smoothAlpha        = 0.6;     // EMA: new = a·fit + (1−a)·prev
        // personalise stage
        int  contourItersPersonalise = 40;
        int  photoIterations         = 20;
        int  photoPixelStride        = 1;
        bool personaliseOptimizeShape = true;   // matches the offline video path
        bool optimizeFocal            = false;  // solve fx=fy during personalise
        // track stage
        int  contourItersTrack     = 10;
        int  trackPhotoIterations  = 2;     // lighting-refresh photometric call
        bool trackPhotoOptimizePose = true; // offline parity (its pose is unused)
        int  lightingEvery         = 1;     // refresh lighting every k frames
        // depth term
        double depthPointToPlaneWeight = 1.0;
        double depthWeight             = 1.0;
        int    depthVertexStride       = 8;
        // detection gating
        double gateFrac    = 0.12;   // reject centroid jumps > frac·imageWidth
        int    maxGateFails = 10;    // then reset the gate (tracking recovery)
        // Phase 4: pyramid photometric pose refinement during track()
        bool             photoRefine   = false;
        std::vector<int> pyramidWidths = {100, 200};   // coarse → fine
    };

    FaceTracker(const BFMLoader& bfm, const Eigen::Matrix3f& K, const Config& cfg);

    bool personalised() const { return personalised_; }

    // Forget the person + tracking history (live mode 'p' key). K is restored
    // to `K0` so a fresh personalisation re-estimates the focal from the guess.
    void reset(const Eigen::Matrix3f& K0);

    // Returns false if the frame could not be personalised (no landmarks).
    // initZ = initial face depth in mm (Biwi: GT head centre; webcam: ~500).
    bool personalise(const cv::Mat& bgr,
                     const std::vector<LandmarkObservation>& observations,
                     double initZ,
                     const std::vector<Eigen::Vector3d>* depthCloud = nullptr);

    // Returns false if the frame was skipped (no landmarks / gated detection).
    // observations may be empty when a depth cloud drives the fit.
    bool track(const cv::Mat& bgr,
               const std::vector<LandmarkObservation>& observations,
               double initZ,
               const std::vector<Eigen::Vector3d>* depthCloud = nullptr);

    // ── current (smoothed) state, for rendering ──
    const PoseParameters&  pose()     const { return prevPose_; }
    const Eigen::VectorXd& expr()     const { return prevExpr_; }
    const Eigen::VectorXd& identity() const { return identity_; }
    const Eigen::VectorXd& beta()     const { return beta_; }
    const light::SHCoeffs& sh()       const { return prevSh_; }
    const Eigen::Matrix3f& K()        const { return K_; }
    Eigen::MatrixX3f currentShape()  const;   // identity + smoothed expression
    Eigen::MatrixX3f currentAlbedo() const;   // mean + colour basis · β

private:
    // Mean 2D position of the interior (fixed-vertex) observations.
    static Eigen::Vector2d centroid(const std::vector<LandmarkObservation>& obs);
    // Interior-landmark reprojection RMS of `shape` under `pose` — the
    // safeguard for the photometric pose refinement.
    double interiorRms(const std::vector<LandmarkObservation>& obs,
                       const Eigen::MatrixX3f& shape,
                       const PoseParameters& pose) const;

    const BFMLoader& bfm_;
    Eigen::Matrix3f  K_;
    Config           cfg_;

    bool            personalised_ = false;
    Eigen::VectorXd identity_, beta_;
    light::SHCoeffs prevSh_ = light::defaultWhite();
    PoseParameters  prevPose_, prev2Pose_;
    Eigen::VectorXd prevExpr_, prev2Expr_;
    bool            havePrev2_ = false;

    Eigen::Vector2d prevCentroid_{-1.0, -1.0};
    bool            haveCentroid_ = false;
    int             gateFails_    = 0;
    long            frameIdx_     = 0;
};
