#include "FaceTracker.h"

#include "ProjectionUtils.h"
#include "ScopedTimer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <utility>

FaceTracker::FaceTracker(const BFMLoader& bfm, const Eigen::Matrix3f& K,
                         const Config& cfg)
    : bfm_(bfm), K_(K), cfg_(cfg) {}

void FaceTracker::reset(const Eigen::Matrix3f& K0)
{
    K_            = K0;
    personalised_ = false;
    haveCentroid_ = false;
    havePrev2_    = false;
    gateFails_    = 0;
    frameIdx_     = 0;
}

Eigen::Vector2d FaceTracker::centroid(const std::vector<LandmarkObservation>& obs)
{
    Eigen::Vector2d c(0, 0);
    int n = 0;
    for (const LandmarkObservation& o : obs)
        if (o.vertexIndex >= 0) { c += o.imagePoint; ++n; }
    return n ? Eigen::Vector2d(c / n) : Eigen::Vector2d(-1, -1);
}

Eigen::MatrixX3f FaceTracker::currentShape() const
{
    if (!personalised_) return bfm_.mean_shape();
    return bfm_.shape(identity_.cast<float>(), prevExpr_.cast<float>());
}

Eigen::MatrixX3f FaceTracker::currentAlbedo() const
{
    if (beta_.size() == 0) return bfm_.albedo();
    Eigen::VectorXf bf = Eigen::VectorXf::Zero(bfm_.color_sigma().size());
    const int n = std::min<int>(static_cast<int>(beta_.size()),
                                static_cast<int>(bf.size()));
    bf.head(n) = beta_.head(n).cast<float>();
    return bfm_.albedo(bf);
}

double FaceTracker::interiorRms(const std::vector<LandmarkObservation>& obs,
                                const Eigen::MatrixX3f& shape,
                                const PoseParameters& pose) const
{
    const proj::Pixels uv = proj::projectMesh(
        shape, pose.rotationMatrix(), pose.translation.cast<float>(), K_);
    double s = 0.0;
    int n = 0;
    for (const LandmarkObservation& o : obs) {
        if (o.vertexIndex < 0 || o.vertexIndex >= uv.rows()) continue;
        if (uv(o.vertexIndex, 0) < 0) continue;            // behind the camera
        const double du = uv(o.vertexIndex, 0) - o.imagePoint.x();
        const double dv = uv(o.vertexIndex, 1) - o.imagePoint.y();
        s += du * du + dv * dv;
        ++n;
    }
    return n ? std::sqrt(s / n) : 1e9;
}

bool FaceTracker::personalise(const cv::Mat& bgr,
                              const std::vector<LandmarkObservation>& observations,
                              double initZ,
                              const std::vector<Eigen::Vector3d>* depthCloud)
{
    if (observations.empty()) return false;
    const double zMin = 0.4 * initZ, zMax = 2.5 * initZ;

    prevCentroid_ = centroid(observations);
    haveCentroid_ = true;
    gateFails_    = 0;

    const Eigen::MatrixX3f meanShape = bfm_.mean_shape();
    std::vector<LandmarkObservation> interior;
    std::copy_if(observations.begin(), observations.end(),
                 std::back_inserter(interior),
                 [](const LandmarkObservation& o) { return o.vertexIndex >= 0; });

    PoseParameters init;
    init.translation = Eigen::Vector3d(0.0, 0.0, initZ);
    const PoseParameters poseOnly = CeresFitter::fitPose(
        meanShape, interior, K_, init, zMin, zMax);

    double focal = K_(0, 0);
    const FitParameters geo = CeresFitter::fitPoseAndShapeContour(
        meanShape, bfm_.shape_basis_raw(), bfm_.shape_sigma(),
        bfm_.expr_basis_raw(), bfm_.expr_sigma(), bfm_.faces(),
        observations, K_, poseOnly, cfg_.sparseReg, cfg_.exprRegPersonalise,
        zMin, zMax, cfg_.contourItersPersonalise, depthCloud,
        cfg_.depthPointToPlaneWeight, cfg_.depthWeight, cfg_.depthVertexStride,
        Eigen::VectorXd(), Eigen::VectorXd(), /*optimizeIdentity=*/true,
        cfg_.optimizeFocal, cfg_.optimizeFocal ? &focal : nullptr);
    if (cfg_.optimizeFocal) {
        K_(0, 0) = K_(1, 1) = static_cast<float>(focal);
        std::cout << "[tracker] personalised focal: " << focal << " px\n";
    }
    identity_ = geo.shapeCoefficients;

    // ── Appearance + PHOTOMETRIC IDENTITY refinement (coarse-to-fine) ─────────
    // Depth is too coarse (~3 mm Kinect noise) for the fine surface detail that
    // carries identity — the soft brow/cheek/jaw that read as feminine sit
    // below that floor and default to the androgynous mean. The RGB SHADING
    // carries that detail, so we drive it with an analysis-by-synthesis
    // photometric fit: render the current model, compare per-pixel to the
    // photo, and move SHAPE (+ albedo + lighting) to match. Run coarse-to-fine
    // (each level warm-starts the next) so it doesn't stick in a local minimum,
    // with POSE FIXED at the reliable geometry pose so the shape gradient is not
    // confounded by pose drift. Each level's shape delta is accepted only if it
    // does not blow up the landmark reprojection (guard against a photometric
    // step corrupting the pose-critical fit under bad lighting/albedo).
    const Eigen::VectorXf exprF = geo.exprCoefficients.cast<float>();
    prevPose_ = geo.pose;                     // geometry pose is the reference
    beta_     = Eigen::VectorXd();            // accumulated across levels
    prevSh_   = light::defaultWhite();
    const double rmsBefore = interiorRms(
        observations, bfm_.shape(identity_.cast<float>(), exprF), geo.pose);

    for (const int width : cfg_.personalisePhotoPyramid) {
        const Eigen::MatrixX3f fitted =
            bfm_.shape(identity_.cast<float>(), exprF);
        FitParameters pin;
        pin.pose = geo.pose;                  // fixed (optimizePose=false below)
        pin.shapeCoefficients  = Eigen::VectorXd::Zero(kShapeCoefficientCount);
        pin.albedoCoefficients = beta_;
        pin.sh = prevSh_;
        const FitParameters p = CeresFitter::fitPhotometric(
            fitted, bfm_.shape_basis_raw(), bfm_.shape_sigma(), bfm_.faces(),
            bfm_.albedo(), bfm_.color_basis_raw(), bfm_.color_sigma(),
            bgr, K_, pin, cfg_.photoShapeReg, cfg_.albedoRegWeight,
            cfg_.photoIterations, cfg_.photoPixelStride, 1.0,
            cfg_.personaliseOptimizeShape, /*optimizeLighting=*/true,
            /*optimizeAlbedo=*/true, /*optimizePose=*/false, nullptr, width,
            /*landmarks=*/&observations, cfg_.photoLandmarkWeight);
        beta_   = p.albedoCoefficients;
        prevSh_ = p.sh;
        if (cfg_.personaliseOptimizeShape &&
            p.shapeCoefficients.size() == identity_.size()) {
            const Eigen::VectorXd refined = identity_ + p.shapeCoefficients;
            const double after = interiorRms(
                observations, bfm_.shape(refined.cast<float>(), exprF), geo.pose);
            const double deltaNorm = p.shapeCoefficients.norm();
            if (after <= rmsBefore * 1.15) {  // looser than 1.05: coarse-to-fine
                identity_ = refined;
                std::cout << "[tracker] photometric id refine @" << width
                          << "px kept (RMS " << rmsBefore << "->" << after
                          << ", |Δα|=" << deltaNorm << ")\n";
            } else {
                std::cout << "[tracker] photometric id refine @" << width
                          << "px rejected (RMS " << rmsBefore << "->" << after
                          << ")\n";
            }
        }
    }

    prevExpr_ = geo.exprCoefficients;
    havePrev2_ = false;
    frameIdx_  = 0;
    personalised_ = true;
    return true;
}

bool FaceTracker::track(const cv::Mat& bgr,
                        const std::vector<LandmarkObservation>& observations,
                        double initZ,
                        const std::vector<Eigen::Vector3d>* depthCloud)
{
    if (!personalised_) return false;
    const bool haveObs = !observations.empty();
    if (!haveObs && !depthCloud) return false;   // nothing to fit against
    const double zMin = 0.4 * initZ, zMax = 2.5 * initZ;

    // Detection gating (only meaningful when landmarks drive the frame): reject
    // a detection whose centroid teleports; after maxGateFails consecutive
    // rejects assume the head really moved (or tracking is lost) and reset the
    // gate + motion history so the next detection re-anchors from scratch.
    if (haveObs) {
        const Eigen::Vector2d c = centroid(observations);
        const double gate = cfg_.gateFrac * bgr.cols;
        if (haveCentroid_ && (c - prevCentroid_).norm() > gate) {
            if (++gateFails_ >= cfg_.maxGateFails) {
                std::cout << "[tracker] gate reset after " << gateFails_
                          << " rejects — re-anchoring\n";
                haveCentroid_ = false;
                havePrev2_    = false;
                gateFails_    = 0;
            } else {
                std::cout << "[tracker] reject frame (detection jumped "
                          << (c - prevCentroid_).norm() << " px)\n";
                return false;
            }
        } else {
            gateFails_ = 0;
        }
        prevCentroid_ = c;
        haveCentroid_ = true;
    }
    ++frameIdx_;

    // Constant-velocity prediction → warm-start where the head is heading.
    PoseParameters  initPose = prevPose_;
    Eigen::VectorXd initExpr = prevExpr_;
    if (havePrev2_) {
        initPose.angleAxis   = 2.0 * prevPose_.angleAxis   - prev2Pose_.angleAxis;
        initPose.translation = 2.0 * prevPose_.translation - prev2Pose_.translation;
        initExpr = (2.0 * prevExpr_ - prev2Expr_).cwiseMax(-3.5).cwiseMin(3.5);
    }

    const Eigen::MatrixX3f meanShape = bfm_.mean_shape();
    FitParameters geo;
    {
        ScopedTimer t("track/contour");
        geo = CeresFitter::fitPoseAndShapeContour(
            meanShape, bfm_.shape_basis_raw(), bfm_.shape_sigma(),
            bfm_.expr_basis_raw(), bfm_.expr_sigma(), bfm_.faces(),
            observations, K_, initPose, cfg_.sparseReg, cfg_.exprRegTrack,
            zMin, zMax, cfg_.contourItersTrack, depthCloud,
            cfg_.depthPointToPlaneWeight, cfg_.depthWeight, cfg_.depthVertexStride,
            identity_, initExpr, /*optimizeIdentity=*/false,
            /*optimizeFocal=*/false, /*focalInOut=*/nullptr,
            cfg_.exprTemporalReg);
    }

    const Eigen::MatrixX3f fitted = bfm_.shape(
        identity_.cast<float>(), geo.exprCoefficients.cast<float>());

    // Phase 4: pyramid photometric pose refinement (coarse → fine), accepted
    // only if it does not worsen the landmark reprojection — the safeguard that
    // keeps the (weaker) dense term from dragging the pose off the landmarks.
    if (cfg_.photoRefine && haveObs) {
        ScopedTimer t("track/photoRefine");
        FitParameters ph;
        ph.pose = geo.pose;
        ph.shapeCoefficients  = Eigen::VectorXd::Zero(kShapeCoefficientCount);
        ph.albedoCoefficients = beta_;
        ph.sh = prevSh_;
        for (const int wpx : cfg_.pyramidWidths) {
            const FitParameters r = CeresFitter::fitPhotometric(
                fitted, bfm_.shape_basis_raw(), bfm_.shape_sigma(), bfm_.faces(),
                bfm_.albedo(), bfm_.color_basis_raw(), bfm_.color_sigma(),
                bgr, K_, ph, cfg_.sparseReg, cfg_.albedoRegWeight,
                /*numIterations=*/1, /*pixelStride=*/2, 1.0,
                /*optimizeShape=*/false, /*optimizeLighting=*/false,
                /*optimizeAlbedo=*/false, /*optimizePose=*/true,
                nullptr, wpx);
            ph.pose = r.pose;
        }
        const double before = interiorRms(observations, fitted, geo.pose);
        const double after  = interiorRms(observations, fitted, ph.pose);
        if (after <= before * 1.05)
            geo.pose = ph.pose;
        else
            std::cout << "[tracker] photo refine rejected (landmark RMS "
                      << before << " → " << after << " px)\n";
    }

    // Lighting refresh (linear estimate; every k-th frame).
    light::SHCoeffs newSh = prevSh_;
    if (cfg_.lightingEvery > 0 && frameIdx_ % cfg_.lightingEvery == 0) {
        ScopedTimer t("track/lighting");
        FitParameters photoInit;
        photoInit.pose = geo.pose;
        photoInit.shapeCoefficients  = Eigen::VectorXd::Zero(kShapeCoefficientCount);
        photoInit.albedoCoefficients = beta_;
        photoInit.sh = prevSh_;
        const FitParameters photo = CeresFitter::fitPhotometric(
            fitted, bfm_.shape_basis_raw(), bfm_.shape_sigma(), bfm_.faces(),
            bfm_.albedo(), bfm_.color_basis_raw(), bfm_.color_sigma(),
            bgr, K_, photoInit, cfg_.sparseReg, cfg_.albedoRegWeight,
            cfg_.trackPhotoIterations, cfg_.photoPixelStride, 1.0,
            /*optimizeShape=*/false, /*optimizeLighting=*/true,
            /*optimizeAlbedo=*/false, cfg_.trackPhotoOptimizePose);
        newSh = photo.sh;
    }

    // Temporal smoothing (EMA) on pose + expression to damp jitter.
    // ADAPTIVE: heavy smoothing at rest (damps solver noise), light smoothing
    // under real motion — a fixed α visibly lagged the overlay behind fast
    // head turns. Motion is measured raw-fit vs previous smoothed state;
    // ~6°/frame of rotation, ~40 mm/frame of translation, or a fast
    // expression change (a mouth opening is ‖Δδ‖ ≈ 1+/frame — without this
    // term the EMA damped articulation onset just like pose lag) pushes α
    // toward pass-through.
    const double exprDelta = (prevExpr_.size() == geo.exprCoefficients.size())
        ? (geo.exprCoefficients - prevExpr_).norm() : 0.0;
    const double motion =
        (geo.pose.translation - prevPose_.translation).norm() / 40.0 +
        (geo.pose.angleAxis   - prevPose_.angleAxis).norm()   / 0.10 +
        exprDelta / 0.8;
    const double a = std::min(0.95, cfg_.smoothAlpha * (1.0 + 0.5 * motion));
    PoseParameters smoothPose;
    smoothPose.angleAxis   = a * geo.pose.angleAxis   + (1 - a) * prevPose_.angleAxis;
    smoothPose.translation = a * geo.pose.translation + (1 - a) * prevPose_.translation;
    const Eigen::VectorXd smoothExpr =
        a * geo.exprCoefficients + (1 - a) * prevExpr_;

    prev2Pose_ = prevPose_;
    prev2Expr_ = prevExpr_;
    prevPose_  = smoothPose;
    prevExpr_  = smoothExpr;
    prevSh_    = newSh;
    havePrev2_ = true;
    return true;
}
