#pragma once
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <vector>

#include "BFMLoader.h"
#include "CeresFitter.h"
#include "Lighting.h"

// personalise-then-track state machine shared by the offline video modes and
// the live camera path.
//
// personalise() is the expensive one-off, it works out who the person is
// (identity, expression, albedo, lighting) from the first frame. track() then
// runs every frame after that with identity frozen, so it only has to solve
// pose and expression plus a light refresh, which is what makes it fast
class FaceTracker {
public:
    struct Config {
        // priors and fit weights
        double sparseReg          = 30.0;
        double exprRegPersonalise = 200.0;
        double exprRegTrack       = 5.0;
        // temporal expression prior for track(), damps per-frame jitter while
        // still letting you hold an expression for free
        double exprTemporalReg    = 50.0;
        double albedoRegWeight    = 3.0;
        double smoothAlpha        = 0.1;     // EMA blend, lower is smoother but laggier

        // personalise stage
        int  contourItersPersonalise = 40;
        int  photoIterations         = 20;
        int  photoPixelStride        = 1;
        bool personaliseOptimizeShape = true;
        bool optimizeFocal            = false;  // solve fx=fy for an unknown webcam
        // coarse-to-fine widths for the photometric identity refinement, each
        // level warm-starts the next and the last is near full frame width
        std::vector<int> personalisePhotoPyramid = {256, 512};
        // the dense photometric shape solve keeps identity reg low, the joint
        // landmark anchor below is what keeps that safe
        double photoShapeReg       = 5.0;
        double photoLandmarkWeight = 20.0;

        // track stage
        int  contourItersTrack     = 10;
        int  trackPhotoIterations  = 2;
        // keep the lighting refresh a pure linear estimate at the tracked pose,
        // solving pose here again just burned time and moved the light off pose
        bool trackPhotoOptimizePose = false;
        int  lightingEvery         = 1;      // refresh lighting every k frames

        // depth term
        double depthPointToPlaneWeight = 1.0;
        double depthWeight             = 1.0;
        int    depthVertexStride       = 8;

        // detection gating, reject big centroid jumps then recover after a while
        double gateFrac    = 0.12;
        int    maxGateFails = 10;

        // optional pyramid photometric pose refinement during track()
        bool             photoRefine   = false;
        std::vector<int> pyramidWidths = {100, 200};
    };

    FaceTracker(const BFMLoader& bfm, const Eigen::Matrix3f& K, const Config& cfg);

    bool personalised() const { return personalised_; }

    // forget the person and history (the live 'p' key), K goes back to K0 so a
    // fresh personalise re-guesses the focal
    void reset(const Eigen::Matrix3f& K0);

    // one-off fit on the first good frame, false if there were no landmarks.
    // initZ is the starting face depth in mm
    bool personalise(const cv::Mat& bgr,
                     const std::vector<LandmarkObservation>& observations,
                     double initZ,
                     const std::vector<Eigen::Vector3d>* depthCloud = nullptr);

    // multi-keyframe version, solves one shared identity across several
    // yaw-varied frames which pins down way more of the face than one view can
    bool personaliseBundle(const std::vector<cv::Mat>& bgrs,
                           const std::vector<std::vector<LandmarkObservation>>& obs,
                           const std::vector<double>& initZs,
                           const std::vector<const std::vector<Eigen::Vector3d>*>& depthClouds,
                           int anchor);

    // landmark reprojection error of the current fit, the identity-quality
    // number for the bundle vs single-frame comparison
    double currentInteriorRms(const std::vector<LandmarkObservation>& obs) const {
        return interiorRms(obs, currentShape(), prevPose_);
    }

    // rough yaw estimate from nose vs eye midpoint, 0 is frontal, NaN if the
    // anchor points are missing
    static double yawProxy(const std::vector<LandmarkObservation>& obs);

    // per-frame update, false if the frame was skipped
    bool track(const cv::Mat& bgr,
               const std::vector<LandmarkObservation>& observations,
               double initZ,
               const std::vector<Eigen::Vector3d>* depthCloud = nullptr);

    // current smoothed state for rendering
    const PoseParameters&  pose()     const { return prevPose_; }
    const Eigen::VectorXd& expr()     const { return prevExpr_; }
    const Eigen::VectorXd& identity() const { return identity_; }
    const Eigen::VectorXd& beta()     const { return beta_; }
    const light::SHCoeffs& sh()       const { return prevSh_; }
    const Eigen::Matrix3f& K()        const { return K_; }
    Eigen::MatrixX3f currentShape()  const;
    Eigen::MatrixX3f currentAlbedo() const;

private:
    static Eigen::Vector2d centroid(const std::vector<LandmarkObservation>& obs);
    // fit albedo + lighting + a photometric identity refinement then commit the
    // tracking state, shared by personalise() and personaliseBundle()
    void finalizeAppearance(const cv::Mat& bgr,
                            const std::vector<LandmarkObservation>& obs,
                            const PoseParameters& pose,
                            const Eigen::VectorXd& expr);
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
