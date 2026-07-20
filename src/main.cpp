#include "BFMLoader.h"
#include "BiwiLoader.h"
#include "Renderer.h"
#include "ProjectionUtils.h"
#include "Lighting.h"
#include "CeresFitter.h"
#include "LandmarkDetector.h"
#include "FaceTracker.h"
#include "ScopedTimer.h"
#ifdef USE_CUDA
#include "CudaRenderer.h"   // GPU rasteriser (built with `make USE_CUDA=1`)
#endif

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>   // imshow/waitKey for --mode live

#include <algorithm>
#include <numeric>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>   // MediaPipe landmark coprocess (live mode)
#include <unistd.h>

// Config — every path and tunable lives here, grouped by concern.
namespace cfg {

// input paths    the only supported dataset is Biwi (Kinect RGB-D).
const std::string kBfmPath = "data/bfm/model2017-1_bfm_nomouth.h5";
std::string       kBiwiDir = "data/biwi/01";   // set via --biwi-seq / --biwi-dir
// Pretrained landmark models: LBF for 68-point landmarks, YuNet ONNX for
// pose-robust face detection. Resolved from ./models (dev) or the baked
// /opt/models (devcontainer image).
std::string modelPath(const std::string& name) {
    for (const std::string& dir : {std::string("models"), std::string("/opt/models")})
        if (std::filesystem::exists(dir + "/" + name)) return dir + "/" + name;
    return "models/" + name;   // reported if genuinely missing
}
const std::string kLbfModelPath = modelPath("lbfmodel.yaml");
const std::string kYuNetPath    = modelPath("face_detection_yunet.onnx");

// landmark detector backend (--detector):
//   yunet     : YuNet face box + 5 points + LBF jaw contour (default)
//   lbf       : Haar box + LBF 68-point landmarks
//   mediapipe : per-frame landmarks_mp_XXXXX.txt from the offline pre-pass
std::string kDetector = "yunet";
// Whether --detector was passed on the CLI. Live mode upgrades the DEFAULT to
// the MediaPipe coprocess (best landmarks) but never overrides an explicit
// choice.
bool kDetectorExplicit = false;

// Multi-keyframe identity bundling (Face2Face §6), offline video only. Opt-in
// via --bundle; default is the single-frame personalise (for A/B comparison).
bool kBundlePersonalise = false;
int  kBundleKeyframes   = 7;   // target keyframe count (--bundle-keyframes)

// output layout    each run writes into data/out/<tag>/ (biwi_video_rgb,
// biwi_dense, debug, …) so modes never clobber each other.
const std::string kOutDir = "data/out";
std::string outDir(const std::string& tag) {
    const std::string d = kOutDir + "/" + tag;
    std::filesystem::create_directories(d);
    return d;
}

// Depth (mm) of the synthetic frontal camera used for the "reconstruction
// frontal" panel and the mean-face debug render — a comfortable framing, not a
// fit parameter.
constexpr float kFrontalRenderDepthMM = 350.0f;

// sparse / contour landmark fit
// Identity reg is deliberately low: the contour fit normalises residuals by face
// size, so a low reg lets the silhouette widen the face instead of pinning it to
// the mean. Override with --sparse-reg.
constexpr double kDefaultSparseReg  = 30.0;   // identity reg (--sparse-reg)
constexpr int    kContourOuterIters = 40;
// Expression prior for personalise: stiffer than the identity reg so identity,
// not expression, explains a non-neutral first frame.
constexpr double kExprRegWeight = 200.0;
// Expression prior for tracking: weak, so a held articulation isn't pulled shut
// every frame (jitter is damped by the temporal prior, not this one).
constexpr double kTrackExprRegWeight = 5.0;

// photometric (appearance) fit
// Albedo prior: low, so the person's actual colouring comes through instead of
// collapsing to the mean skin/lips/brows; still resists baking in lighting.
constexpr double kAlbedoRegWeight  = 3.0;
constexpr int    kPhotoIterations  = 20;
constexpr int    kPhotoPixelStride = 1;

// video temporal smoothing    EMA on the per-frame pose + expression to
// damp jitter: new = α·fit + (1−α)·previous. 1 = no smoothing, lower = smoother
// (but laggier). Paired with velocity prediction so it stays responsive.
constexpr double kSmoothAlpha = 0.95;   // NOTE: LOW = heavy smoothing (laggy); raise toward 0.9 for responsive

// depth term (Biwi "full" fit)
constexpr int    kDepthBackprojStride     = 2;    // subsample the depth map
constexpr double kDepthCropRadiusMM       = 90.0;   // tight → excludes hair/neck
constexpr double kDepthCropFrontSlabMM    = 90.0;
constexpr double kDepthPointToPlaneWeight = 1.0;
constexpr double kDepthWeight             = 1.0;  // depth term scale vs landmarks
constexpr int    kDepthVertexStride       = 8;    // model subsample for ICP

}  // namespace cfg

// This is a single translation unit; pull the config names into scope so call
// sites read cleanly (kOutDir, outDir(), tunables …).
using namespace cfg;

// Raw uint16 depth (mm) → 3-channel 8-bit BGR visualisation (near = bright,
// empty = black). Used as the "background photo" for the dense depth overlay.
static cv::Mat depthToBgr(const cv::Mat& depthRaw)
{
    cv::Mat mask = depthRaw > 0;
    cv::Mat vis;
    cv::normalize(depthRaw, vis, 0, 255, cv::NORM_MINMAX, CV_8U, mask);
    cv::bitwise_not(vis, vis, mask);     // near = bright
    vis.setTo(0, ~mask);                 // no measurement = black
    cv::Mat bgr;
    cv::cvtColor(vis, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

// Debug overlays

// Draw the BFM mesh as a wireframe over an input photo. Each edge is coloured
// by the average albedo of its endpoints; triangles with a vertex behind the
// camera (uv == -1) or fully off-image are skipped.
static cv::Mat overlayWireframe(const cv::Mat&            bgr,
                                const proj::Pixels&       uv,
                                const Eigen::MatrixX3i&   triangles,
                                const Eigen::MatrixX3f&   albedo,
                                int                       stride = 1)
{
    cv::Mat out = bgr.clone();
    const int H = out.rows, W = out.cols;

    auto inImage = [&](const cv::Point& p) {
        return p.x >= 0 && p.y >= 0 && p.x < W && p.y < H;
    };
    auto vertColour = [&](int i) {
        return cv::Vec3b(static_cast<uchar>(albedo(i, 2) * 255.0f),
                         static_cast<uchar>(albedo(i, 1) * 255.0f),
                         static_cast<uchar>(albedo(i, 0) * 255.0f));
    };
    auto pixel = [&](int i) {
        return cv::Point(static_cast<int>(uv(i, 0) + 0.5f),
                         static_cast<int>(uv(i, 1) + 0.5f));
    };
    auto mix = [](cv::Vec3b a, cv::Vec3b b) {
        return cv::Vec3b((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2);
    };

    for (int t = 0; t < triangles.rows(); t += stride) {
        const int i0 = triangles(t, 0), i1 = triangles(t, 1), i2 = triangles(t, 2);
        if (uv(i0, 0) < 0 || uv(i1, 0) < 0 || uv(i2, 0) < 0) continue;

        const cv::Point p0 = pixel(i0), p1 = pixel(i1), p2 = pixel(i2);
        if (!inImage(p0) && !inImage(p1) && !inImage(p2)) continue;

        const cv::Vec3b c0 = vertColour(i0), c1 = vertColour(i1), c2 = vertColour(i2);
        cv::line(out, p0, p1, mix(c0, c1), 1, cv::LINE_AA);
        cv::line(out, p1, p2, mix(c1, c2), 1, cv::LINE_AA);
        cv::line(out, p2, p0, mix(c2, c0), 1, cv::LINE_AA);
    }
    return out;
}

// Normalised, masked depth → PNG. Nearer surfaces render brighter; empty
// pixels are black.
static void writeDepthVis(const RenderOutput& r, const std::string& path)
{
    cv::Mat vis;
    cv::normalize(r.depth, vis, 0, 255, cv::NORM_MINMAX, CV_8U, r.mask);
    cv::bitwise_not(vis, vis, r.mask);   // near = bright
    vis.setTo(0, ~r.mask);               // empty = black
    cv::imwrite(path, vis);
    std::cout << "wrote depth debug → " << path << "\n";
}

// Rendered RGB image (float [0,1], RGB) → 8-bit BGR PNG.
static void writeColourImage(const RenderOutput& r, const std::string& path)
{
    cv::Mat bgr;
    cv::cvtColor(r.image, bgr, cv::COLOR_RGB2BGR);
    bgr.convertTo(bgr, CV_8UC3, 255.0f);
    cv::imwrite(path, bgr);
    std::cout << "wrote rendered image → " << path << "\n";
}

// Shared render / IO helpers

// Draw a small green status label in the top-left corner (panel captions, HUD).
static void drawLabel(cv::Mat& img, const std::string& text, double scale = 0.6)
{
    cv::putText(img, text, {8, 22}, cv::FONT_HERSHEY_SIMPLEX, scale,
                {0, 220, 0}, 2, cv::LINE_AA);
}

// Render a face into an 8-bit BGR image (the reconstruction alone on black) —
// the panel format shared by every mask/reconstruction view. Renderer float RGB
// [0,1] → 8-bit BGR.
static cv::Mat renderFaceBgr(const Renderer& renderer, const RenderInput& in)
{
    cv::Mat bgr;
    cv::cvtColor(renderer.render(in).image, bgr, cv::COLOR_RGB2BGR);
    bgr.convertTo(bgr, CV_8UC3, 255.0);
    return bgr;
}

// A path is an image SEQUENCE (needs cv::CAP_IMAGES, else FFmpeg opens it as a
// one-frame video) if it names a still image or contains a printf pattern.
static bool isImageSequence(const std::string& path)
{
    return path.find(".png") != std::string::npos ||
           path.find(".jpg") != std::string::npos ||
           path.find('%')    != std::string::npos;
}

// Downscale to a target processing width (never upscales); keeps aspect ratio.
static cv::Mat resizeToWidth(const cv::Mat& in, int width)
{
    const double scale = std::min(1.0, static_cast<double>(width) / in.cols);
    if (scale >= 1.0) return in;
    cv::Mat out;
    cv::resize(in, out, {}, scale, scale, cv::INTER_AREA);
    return out;
}

// FaceTracker config for the offline video paths (rgb/rgbd) and the transfer
// target personalise — full-quality personalise (per-pixel shape refine on) with
// the metric depth term wired up. Tunables live in namespace cfg.
static FaceTracker::Config offlineConfig(double sparseReg)
{
    FaceTracker::Config tc;
    tc.sparseReg               = sparseReg;
    tc.exprRegPersonalise      = kExprRegWeight;
    tc.exprRegTrack            = kTrackExprRegWeight;
    tc.albedoRegWeight         = kAlbedoRegWeight;
    tc.smoothAlpha             = kSmoothAlpha;
    tc.contourItersPersonalise = kContourOuterIters;
    tc.photoIterations         = kPhotoIterations;
    tc.photoPixelStride        = kPhotoPixelStride;
    tc.depthPointToPlaneWeight = kDepthPointToPlaneWeight;
    tc.depthWeight             = kDepthWeight;
    tc.depthVertexStride       = kDepthVertexStride;
    return tc;
}

// Pipeline stages

// Alpha-blend a rendered face over the photo. The render MUST have been
// produced at the photo's resolution (otherwise the two images don't line up
// and cv::addWeighted would throw). Rendered pixels outside the face mask are
// left as the original photo so we only blend where the model actually is.
static cv::Mat blendRenderOnPhoto(const RenderOutput& r, const cv::Mat& photo)
{
    // Rendered image: float RGB [0,1] → 8-bit BGR to match the photo.
    cv::Mat bgr;
    cv::cvtColor(r.image, bgr, cv::COLOR_RGB2BGR);
    bgr.convertTo(bgr, CV_8UC3, 255.0f);

    // Render-dominant blend (70/30): a skin-coloured render at 50/50 is nearly
    // invisible on skin / on the bright depth background; 30 % photo is enough
    // to judge alignment. Soft alpha edge (blurred mask as per-pixel weight)
    // instead of a hard mask border (staircase artefacts).
    const int kernel = std::max(3, (photo.cols / 400) | 1);   // odd, ~image size
    cv::Mat alpha;
    cv::GaussianBlur(r.mask, alpha, {kernel, kernel}, 0.0);
    alpha.convertTo(alpha, CV_32F, 0.7 / 255.0);              // 0 … 0.7 render weight
    cv::Mat alpha3;
    cv::merge(std::vector<cv::Mat>{alpha, alpha, alpha}, alpha3);

    cv::Mat photoF, renderF, invAlpha3;
    photo.convertTo(photoF, CV_32FC3);
    bgr.convertTo(renderF, CV_32FC3);
    cv::subtract(cv::Scalar::all(1.0), alpha3, invAlpha3);
    cv::Mat overlay;
    cv::Mat blended = photoF.mul(invAlpha3) + renderF.mul(alpha3);
    blended.convertTo(overlay, CV_8UC3);

    // Green contour around the rendered region: makes the fit boundary clear on
    // any background; approxPolyDP smooths the mask's pixel jaggies.
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(r.mask.clone(), contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    const double epsilon = std::max(2.0, photo.cols / 600.0);
    for (std::vector<cv::Point>& contour : contours) {
        std::vector<cv::Point> smoothed;
        cv::approxPolyDP(contour, smoothed, epsilon, /*closed=*/true);
        contour = std::move(smoothed);
    }
    cv::polylines(overlay, contours, /*closed=*/true, {0, 255, 0},
                  std::max(1, photo.cols / 1000), cv::LINE_AA);
    return overlay;
}

static void overlayRenderOnPhoto(const RenderOutput& r,
                                 const cv::Mat&      photo,
                                 const std::string&  path)
{
    const cv::Mat overlay = blendRenderOnPhoto(r, photo);
    cv::imwrite(path, overlay);
    std::cout << "render overlay → " << path
              << "  (" << overlay.cols << "x" << overlay.rows << ")\n";
}

// 3-panel strip: overlay | reconstruction at the fitted pose (on black) |
// reconstruction frontal. Mirrors the per-frame video panels so the single-frame
// geometry modes (dense/full) get the same views.
static cv::Mat renderMaskPanels(
    const cv::Mat&           photo,
    const Eigen::MatrixX3f&  shape,
    const Eigen::MatrixX3f&  albedo,
    const Eigen::Matrix3f&   intrinsics,
    const PoseParameters&    pose,
    const Eigen::MatrixX3i&  faces,
    const light::SHCoeffs&   sh = light::defaultWhite())
{
    const Renderer renderer(photo.rows, photo.cols, faces);
    const auto reconBgr = [&](const PoseParameters& p, const Eigen::Matrix3f& K) {
        return renderFaceBgr(renderer, { .shape = shape, .albedo = albedo,
            .R = p.rotationMatrix(), .t = p.translation.cast<float>(),
            .K = K, .sh = sh });
    };

    const RenderInput overlayIn{ .shape = shape, .albedo = albedo,
        .R = pose.rotationMatrix(), .t = pose.translation.cast<float>(),
        .K = intrinsics, .sh = sh };
    cv::Mat overlay = blendRenderOnPhoto(renderer.render(overlayIn), photo);
    drawLabel(overlay, "overlay");

    cv::Mat reconPose = reconBgr(pose, intrinsics);
    drawLabel(reconPose, "reconstruction @ pose");

    PoseParameters frontal;                                // identity rotation, centred
    frontal.translation = Eigen::Vector3d(0, 0, kFrontalRenderDepthMM);
    cv::Mat reconFront =
        reconBgr(frontal, proj::defaultIntrinsics(photo.cols, photo.rows));
    drawLabel(reconFront, "reconstruction frontal");

    cv::Mat composite;
    cv::hconcat(std::vector<cv::Mat>{overlay, reconPose, reconFront}, composite);
    return composite;
}

static void writeFitOutputs(
    const std::string& suffix,
    const cv::Mat& photo,
    const BFMLoader& bfm,
    const Eigen::MatrixX3f& shape,
    const Eigen::MatrixX3f& albedo,
    const Eigen::Matrix3f& intrinsics,
    const PoseParameters& pose,
    const std::string& tag,             // output subfolder, e.g. "biwi_dense"
    // Optional: the fit's 2D landmark observations. If set, they (green) and the
    // projected model vertices (red) are drawn into the wireframe and the
    // reprojection RMS is logged — the visual proof the sparse term worked.
    const std::vector<LandmarkObservation>* observations = nullptr,
    // Estimated lighting for this stage (photometric passes it; earlier stages
    // default to flat white, matching the plain overlay below).
    const light::SHCoeffs& sh = light::defaultWhite()
)
{
    const Eigen::Matrix3f rotation =
        pose.rotationMatrix();

    const Eigen::Vector3f translation =
        pose.translation.cast<float>();

    const proj::Pixels projectedVertices =
        proj::projectMesh(
            shape,
            rotation,
            translation,
            intrinsics
        );

    cv::Mat wireframe =
        overlayWireframe(
            photo,
            projectedVertices,
            bfm.faces(),
            albedo,
            2
        );

    if (observations) {
        const int radius = std::max(4, photo.cols / 400);
        double sumSquaredError = 0.0;
        int fixedCount = 0;
        for (const LandmarkObservation& obs : *observations) {
            const cv::Point detected(
                static_cast<int>(obs.imagePoint.x() + 0.5),
                static_cast<int>(obs.imagePoint.y() + 0.5));
            if (obs.vertexIndex < 0) {
                // contour point (dynamic correspondence) — cyan, no fixed model vertex
                cv::circle(wireframe, detected, radius, {255, 255, 0}, -1, cv::LINE_AA);
                continue;
            }
            const cv::Point model(
                static_cast<int>(projectedVertices(obs.vertexIndex, 0) + 0.5f),
                static_cast<int>(projectedVertices(obs.vertexIndex, 1) + 0.5f));
            cv::circle(wireframe, detected, radius, {0, 255, 0}, -1, cv::LINE_AA);
            cv::drawMarker(wireframe, model, {0, 0, 255}, cv::MARKER_CROSS,
                           2 * radius, std::max(2, radius / 2), cv::LINE_AA);
            const double du = projectedVertices(obs.vertexIndex, 0) - obs.imagePoint.x();
            const double dv = projectedVertices(obs.vertexIndex, 1) - obs.imagePoint.y();
            sumSquaredError += du * du + dv * dv;
            ++fixedCount;
        }
        const double rms = fixedCount
            ? std::sqrt(sumSquaredError / static_cast<double>(fixedCount)) : 0.0;
        std::cout << tag << ' ' << suffix << ": interior reprojection RMS "
                  << rms << " px (green = detected, red = model, cyan = contour)\n";
    }

    const std::string wireframePath =
        outDir(tag) + "/overlay_" + suffix + ".png";

    cv::imwrite(wireframePath, wireframe);

    std::cout << "Wrote wireframe overlay: "
              << wireframePath << '\n';

    RenderInput renderInput{
        .shape = shape,
        .albedo = albedo,
        .R = rotation,
        .t = translation,
        .K = intrinsics,
        .sh = sh,
    };

    const RenderOutput renderOutput =
        Renderer(
            photo.rows,
            photo.cols,
            bfm.faces()
        ).render(renderInput);

    const std::string renderPath =
        outDir(tag) + "/render_overlay_" + suffix + ".png";

    overlayRenderOnPhoto(
        renderOutput,
        photo,
        renderPath
    );

    // 3-panel mask visualisation: overlay | reconstruction @ pose | frontal.
    const cv::Mat maskPanels =
        renderMaskPanels(photo, shape, albedo, intrinsics, pose, bfm.faces(), sh);
    const std::string maskPath = outDir(tag) + "/mask_panels_" + suffix + ".png";
    cv::imwrite(maskPath, maskPanels);
    std::cout << "Wrote mask panels: " << maskPath << '\n';
}

// Keyframe selection for the identity bundle (Face2Face §6): farthest-point
// sampling in yaw so the keyframes span the widest range of head rotation,
// seeded with the most frontal well-detected frame as the anchor (index 0).
static std::vector<int> selectBundleKeyframes(
    const std::vector<double>& yaw,   // NaN where undetected
    int k)
{
    std::vector<int> cand;
    for (int i = 0; i < static_cast<int>(yaw.size()); ++i)
        if (!std::isnan(yaw[i])) cand.push_back(i);
    if (cand.empty()) return {};

    std::vector<int> sel;
    int frontal = cand[0];
    for (int i : cand) if (std::abs(yaw[i]) < std::abs(yaw[frontal])) frontal = i;
    sel.push_back(frontal);                                   // anchor first
    while (static_cast<int>(sel.size()) < k && sel.size() < cand.size()) {
        int best = -1; double bestSep = -1.0;
        for (int i : cand) {
            if (std::find(sel.begin(), sel.end(), i) != sel.end()) continue;
            double minD = 1e30;
            for (int s : sel) minD = std::min(minD, std::abs(yaw[i] - yaw[s]));
            if (minD > bestSep) { bestSep = minD; best = i; }
        }
        if (best < 0) break;
        sel.push_back(best);
    }
    return sel;
}

// Personalise identity + albedo on frame 0, then track pose + expression (+
// lighting) on the rest, warm-started with identity/albedo frozen (see
// FaceTracker). --mode rgb = no depth; --mode rgbd adds the Kinect depth ICP term.
static void runVideoReconstruction(
    const BFMLoader& bfm,
    const Eigen::MatrixX3f& albedo,
    int numFrames,
    double sparseReg,
    bool useDepth)
{
    try {
        BiwiLoader biwi(kBiwiDir, numFrames);
        const std::vector<BiwiFrame> frames = biwi.getFrames();
        if (frames.empty()) { std::cout << "Biwi: no frames in " << kBiwiDir << '\n'; return; }
        const BiwiCalibration cal = biwi.getCalibration();

        const std::string tag      = useDepth ? "biwi_video_full" : "biwi_video_rgb";
        const std::string dir      = outDir(tag);
        const std::string frameDir = outDir(tag + "/frames");
        const cv::Size sz(frames[0].rgb.cols, frames[0].rgb.rows);
        const Renderer renderer(sz.height, sz.width, bfm.faces());
        // Output is a 3-panel strip: overlay | reconstruction @ tracked pose |
        // reconstruction frontal (both recon views on a black background).
        const cv::Size outSz(sz.width * 3, sz.height);
        cv::VideoWriter writer(dir + "/tracking.mp4",
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 15.0, outSz);
        const Eigen::Matrix3f frontalK = proj::defaultIntrinsics(sz.width, sz.height);

        const auto albedoOf = [&](const Eigen::VectorXd& beta) -> Eigen::MatrixX3f {
            if (beta.size() == 0) return albedo;
            Eigen::VectorXf bf = Eigen::VectorXf::Zero(bfm.color_sigma().size());
            bf.head(std::min<int>(beta.size(), bf.size())) =
                beta.head(std::min<int>(beta.size(), bf.size())).cast<float>();
            return bfm.albedo(bf);
        };
        // Render the reconstruction alone (face on black) → 8-bit BGR panel.
        const auto reconBgr = [&](const Eigen::MatrixX3f& shape,
                                  const Eigen::VectorXd& beta,
                                  const PoseParameters& pose,
                                  const light::SHCoeffs& sh,
                                  const Eigen::Matrix3f& K) {
            return renderFaceBgr(renderer, { .shape = shape, .albedo = albedoOf(beta),
                .R = pose.rotationMatrix(), .t = pose.translation.cast<float>(),
                .K = K, .sh = sh });
        };
        const auto headCloudRgb = [&](const BiwiFrame& f) {
            std::vector<Eigen::Vector3d> head = cropHead(
                backprojectDepth(f.depth, cal.K_depth, kDepthBackprojStride),
                f.headCenter, kDepthCropRadiusMM, kDepthCropFrontSlabMM);
            for (Eigen::Vector3d& p : head) p = cal.R_rgb * p + cal.t_rgb;  // → rgb frame
            return head;
        };
        // Landmark source per --detector: in-process (YuNet or Haar+LBF), or the
        // MediaPipe pre-pass files (landmarks_mp_XXXXX.txt) written offline by
        // python/gen_landmarks_mediapipe.py.
        LandmarkDetector detector(kLbfModelPath,
                                  kDetector == "yunet" ? kYuNetPath : "");
        if (kDetector != "mediapipe" && !detector.ok()) {
            std::cerr << "video: landmark detector unavailable "
                         "(is models/lbfmodel.yaml present?)\n";
            return;
        }
        const auto frameLandmarks =
            [&](const BiwiFrame& f) -> std::vector<LandmarkObservation> {
            if (kDetector == "mediapipe") {
                std::ostringstream mp;
                mp << kBiwiDir << "/landmarks_mp_" << std::setw(5)
                   << std::setfill('0') << f.frameNumber << ".txt";
                try { return loadLandmarkObservations(mp.str()); }
                catch (const std::exception&) { return {}; }
            }
            return detector.detect(f.rgb);
        };

        // The personalise-then-track state machine lives in FaceTracker (shared
        // with --mode live); this loop only feeds frames and renders panels.
        FaceTracker tracker(bfm, cal.K_rgb, offlineConfig(sparseReg));

        // Identity-quality metric: interior-landmark reprojection RMS of the
        // frozen identity at each TRACKED frame. A better identity generalises
        // across poses → lower RMS, especially on frames far from the
        // personalise view. Reported as mean/median at the end.
        std::vector<double> trackRms;

        for (size_t i = 0; i < frames.size(); ++i) {
            const BiwiFrame& f = frames[i];
            const Eigen::Vector3d headRgb = cal.R_rgb * f.headCenter + cal.t_rgb;
            std::vector<Eigen::Vector3d> cloud;
            if (useDepth) cloud = headCloudRgb(f);
            const std::vector<Eigen::Vector3d>* cloudPtr = useDepth ? &cloud : nullptr;

            if (!tracker.personalised()) {
                ScopedTimer t("personalise");
                bool ok = false;
                if (kBundlePersonalise) {
                    // Multi-keyframe bundle: select yaw-diverse keyframes from
                    //    the sequence, gather their landmarks + depth, solve one
                    // shared identity jointly, then track the rest.
                    std::vector<double> yaw(frames.size(),
                                            std::numeric_limits<double>::quiet_NaN());
                    std::vector<std::vector<LandmarkObservation>> allObs(frames.size());
                    for (size_t j = 0; j < frames.size(); ++j) {
                        allObs[j] = frameLandmarks(frames[j]);
                        yaw[j] = FaceTracker::yawProxy(allObs[j]);
                    }
                    const std::vector<int> kf =
                        selectBundleKeyframes(yaw, kBundleKeyframes);
                    if (kf.empty()) { std::cerr << "video: no detectable keyframe\n"; return; }
                    std::vector<cv::Mat> bgrs;
                    std::vector<std::vector<LandmarkObservation>> obs;
                    std::vector<double> initZs;
                    std::vector<std::vector<Eigen::Vector3d>> clouds(kf.size());
                    std::vector<const std::vector<Eigen::Vector3d>*> cloudPtrs;
                    std::cout << "[bundle] keyframes (frame:yaw):";
                    for (size_t m = 0; m < kf.size(); ++m) {
                        const BiwiFrame& kff = frames[kf[m]];
                        bgrs.push_back(kff.rgb);
                        obs.push_back(allObs[kf[m]]);
                        const Eigen::Vector3d hr = cal.R_rgb * kff.headCenter + cal.t_rgb;
                        initZs.push_back(hr.z());
                        if (useDepth) { clouds[m] = headCloudRgb(kff);
                                        cloudPtrs.push_back(&clouds[m]); }
                        else            cloudPtrs.push_back(nullptr);
                        std::cout << ' ' << kff.frameNumber << ':'
                                  << std::fixed << std::setprecision(2) << yaw[kf[m]];
                    }
                    std::cout << '\n';
                    ok = tracker.personaliseBundle(bgrs, obs, initZs, cloudPtrs, 0);
                } else {
                    const std::vector<LandmarkObservation> obs = frameLandmarks(f);
                    ok = tracker.personalise(f.rgb, obs, headRgb.z(), cloudPtr);
                }
                if (!ok) { std::cerr << "video: could not personalise\n"; return; }
                std::cout << "[personalise] "
                          << (kBundlePersonalise ? "BUNDLE" : "single-frame")
                          << " — identity + albedo fixed for the rest\n";
            } else {
                // both modes track from per-frame landmarks: they are the only
                // data term that drives EXPRESSION (the depth cloud drives pose
                // only — see fitPoseAndShapeContour). In rgbd mode a frame
                // without landmarks still tracks pose from the cloud alone.
                std::vector<LandmarkObservation> obs = frameLandmarks(f);
                if (obs.empty() && !useDepth) {
                    std::cout << "  skip frame " << f.frameNumber
                              << " (no landmarks)\n";
                    continue;
                }
                ScopedTimer t("track");
                if (!tracker.track(f.rgb, obs, headRgb.z(), cloudPtr))
                    continue;   // gated / unusable frame (tracker logged why)
                if (!obs.empty()) trackRms.push_back(tracker.currentInteriorRms(obs));
            }

            // 3-panel composite: overlay | reconstruction @ tracked pose |
            // reconstruction frontal (both recon panels on a black background).
            const Eigen::MatrixX3f fitted = tracker.currentShape();
            const Eigen::VectorXd& betaVec = tracker.beta();
            const PoseParameters&  curPose = tracker.pose();
            const light::SHCoeffs& curSh   = tracker.sh();
            const RenderInput in{
                .shape = fitted, .albedo = albedoOf(betaVec),
                .R = curPose.rotationMatrix(), .t = curPose.translation.cast<float>(),
                .K = cal.K_rgb, .sh = curSh };
            cv::Mat overlay = blendRenderOnPhoto(renderer.render(in), f.rgb);
            drawLabel(overlay, i == 0 ? "personalise" : "track");

            cv::Mat reconPose = reconBgr(fitted, betaVec, curPose, curSh, cal.K_rgb);
            drawLabel(reconPose, "reconstruction @ pose");

            PoseParameters frontal;                     // identity rotation, centred
            frontal.translation = Eigen::Vector3d(0, 0, kFrontalRenderDepthMM);
            cv::Mat reconFront = reconBgr(fitted, betaVec, frontal, curSh, frontalK);
            drawLabel(reconFront, "reconstruction frontal");

            cv::Mat composite;
            cv::hconcat(std::vector<cv::Mat>{overlay, reconPose, reconFront}, composite);

            std::ostringstream name;
            name << frameDir << "/frame_" << std::setw(5) << std::setfill('0')
                 << f.frameNumber << ".png";
            cv::imwrite(name.str(), composite);
            if (writer.isOpened()) writer.write(composite);
            std::cout << "video " << (i + 1) << "/" << frames.size()
                      << " (biwi frame " << f.frameNumber << ")\n";
        }
        if (writer.isOpened()) writer.release();
        std::cout << "Wrote tracking video → " << dir << "/tracking.mp4  (+ frames/)\n";

        // Identity-quality summary (the bundle-vs-single metric).
        if (!trackRms.empty()) {
            std::vector<double> s = trackRms;
            std::sort(s.begin(), s.end());
            const double mean = std::accumulate(s.begin(), s.end(), 0.0) / s.size();
            const double median = s[s.size() / 2];
            std::cout << "[METRIC] " << (useDepth ? "rgbd" : "rgb") << " "
                      << (kBundlePersonalise ? "BUNDLE " : "single ")
                      << "interior-reproj RMS over " << s.size() << " tracked frames:"
                      << "  mean " << mean << " px  median " << median
                      << " px  p90 " << s[static_cast<size_t>(s.size() * 0.9)] << " px\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Biwi video skipped: " << e.what() << '\n';
    }
}

// Live MediaPipe landmarks via a child process (mp_landmark_server.py): stream
// frames on stdin (int32 w,h header + BGR bytes), read back "vertexIndex u v"
// lines (-1 = jaw contour). Falls back to the in-process YuNet detector if
// start() fails (e.g. no python with mediapipe installed).
class MpLandmarkStream {
public:
    ~MpLandmarkStream() { stop(); }

    bool start()
    {
        // A dead child must surface as a failed read, not a fatal SIGPIPE.
        std::signal(SIGPIPE, SIG_IGN);
        // The devcontainer bakes mediapipe into python3; on a Mac host it is
        // usually a pyenv/homebrew versioned binary. First one that starts
        // and prints READY wins.
        for (const char* py : {"python3", "python3.12", "python3.11", "python3.10"})
            if (startWith(py)) {
                std::cout << "live: MediaPipe landmark server up (" << py << ")\n";
                return true;
            }
        return false;
    }

    bool running() const { return pid_ > 0; }

    // Returns the frame's observations (empty = no face). On pipe failure the
    // stream shuts down and running() turns false — caller falls back.
    std::vector<LandmarkObservation> detect(const cv::Mat& bgr)
    {
        std::vector<LandmarkObservation> obs;
        if (pid_ <= 0) return obs;
        cv::Mat frame = bgr.isContinuous() ? bgr : bgr.clone();
        const int32_t wh[2] = {frame.cols, frame.rows};
        if (!writeAll(wh, sizeof wh) ||
            !writeAll(frame.data, static_cast<size_t>(frame.cols) * frame.rows * 3)) {
            fail("write");
            return obs;
        }
        char line[128];
        if (!std::fgets(line, sizeof line, rx_)) { fail("read"); return obs; }
        const int n = std::atoi(line);
        for (int i = 0; i < n; ++i) {
            if (!std::fgets(line, sizeof line, rx_)) { fail("read"); return {}; }
            int vertexIndex; double u, v;
            if (std::sscanf(line, "%d %lf %lf", &vertexIndex, &u, &v) == 3)
                obs.push_back({vertexIndex, Eigen::Vector2d(u, v)});
        }
        return obs;
    }

private:
    bool startWith(const char* python)
    {
        int toChild[2], fromChild[2];
        if (pipe(toChild) != 0) return false;
        if (pipe(fromChild) != 0) { close(toChild[0]); close(toChild[1]); return false; }

        const pid_t pid = fork();
        if (pid < 0) {
            for (int fd : {toChild[0], toChild[1], fromChild[0], fromChild[1]}) close(fd);
            return false;
        }
        if (pid == 0) {                                    // child
            dup2(toChild[0], STDIN_FILENO);
            dup2(fromChild[1], STDOUT_FILENO);
            for (int fd : {toChild[0], toChild[1], fromChild[0], fromChild[1]}) close(fd);
            execlp(python, python, "python/mp_landmark_server.py", nullptr);
            _exit(127);                                    // exec failed
        }
        close(toChild[0]);
        close(fromChild[1]);
        pid_ = pid;
        tx_  = toChild[1];
        rx_  = fdopen(fromChild[0], "r");

        // Handshake: the server prints READY after the (slow) mediapipe import.
        // A python without mediapipe exits immediately → fgets returns NULL.
        char line[64];
        if (rx_ && std::fgets(line, sizeof line, rx_) &&
            std::string(line).rfind("READY", 0) == 0)
            return true;
        stop();
        return false;
    }

    bool writeAll(const void* data, size_t n)
    {
        const char* p = static_cast<const char*>(data);
        while (n > 0) {
            const ssize_t w = write(tx_, p, n);
            if (w <= 0) return false;
            p += w;
            n -= static_cast<size_t>(w);
        }
        return true;
    }

    void fail(const char* what)
    {
        std::cerr << "live: MediaPipe landmark server " << what
                  << " failed — falling back to YuNet\n";
        stop();
    }

    void stop()
    {
        if (pid_ <= 0) return;
        if (tx_ >= 0) close(tx_);
        if (rx_) std::fclose(rx_);
        kill(pid_, SIGTERM);
        int status;
        waitpid(pid_, &status, 0);
        pid_ = -1; tx_ = -1; rx_ = nullptr;
    }

    pid_t pid_ = -1;
    int   tx_  = -1;
    FILE* rx_  = nullptr;
};

// Realtime camera mode (--mode live) — HOST ONLY (Docker has no camera access)
struct LiveOptions {
    int         camera        = 0;      // --camera
    std::string source;                 // --live-source: video file / image seq
    int         width         = 640;    // --live-width: processing width
    int         maxFrames     = 0;      // --live-frames: stop + dump PNGs (test)
    bool        display       = true;   // --live-nodisplay disables imshow
    bool        photoRefine   = false;  // --photo-refine: pyramid pose refine
    // OFF by default: single-frame focal estimation is under-determined (the
    // focal↔distance ambiguity tilts toward long focal; verified on the iPhone
    // known-focal test) — the 60°-HFOV guess is more reliable. Opt in with
    // --optimize-focal; robust recovery needs multi-keyframe bundling (plan §3).
    bool        optimizeFocal = false;
    double      initZ         = 500.0;  // webcam ≈ arm's length (mm)
    bool        gpuRender     = false;  // --gpu-render: CUDA display renderer (transfer)
    // --transfer-target <biwi-dir>: expression-transfer mode. The named subject
    // is personalised once as the TARGET avatar; the live camera drives it with
    // YOUR expressions (neutral-relative δ). Empty ⇒ normal live tracking.
    std::string transferTarget;
};

// Open the driving input (--live-source image/video, or a camera) and return the
// first usable frame. Warms the camera up and falls back over device indices to
// dodge macOS/AVFoundation quirks (black frames before the permission dialog,
// Continuity Camera on device 0, native size only). Returns false on failure.
static bool openLiveCapture(cv::VideoCapture& cap, const LiveOptions& lo,
                            cv::Mat& firstFrame, const char* who)
{
    if (!lo.source.empty()) {
        cap.open(lo.source,
                 isImageSequence(lo.source) ? cv::CAP_IMAGES : cv::CAP_ANY);
        if (!cap.isOpened() || !cap.read(firstFrame) || firstFrame.empty()) {
            std::cerr << who << ": could not read " << lo.source << '\n';
            return false;
        }
        return true;
    }
    const auto openWorkingCamera = [&](int idx) -> bool {
        cap.release();
        if (!cap.open(idx)) return false;
        cv::Mat probe;
        for (int attempt = 0; attempt < 50; ++attempt) {   // ~2.5 s warm-up
            if (cap.read(probe) && !probe.empty()) {
                const cv::Scalar m = cv::mean(probe);
                if (m[0] + m[1] + m[2] > 6.0) { firstFrame = probe; return true; }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        std::cout << who << ": camera " << idx
                  << " opened but only delivers black frames — skipping\n";
        return false;
    };
    bool opened = openWorkingCamera(lo.camera);
    // Continuity Camera is often index 0 while the built-in FaceTime camera is 1.
    for (int idx = 0; !opened && idx <= 2; ++idx)
        if (idx != lo.camera) opened = openWorkingCamera(idx);
    // Final fresh retry: a stream opened before the permission grant stays black.
    if (!opened) opened = openWorkingCamera(lo.camera);
    if (!opened) {
        std::cerr << who << ": no camera delivered usable frames.\n"
                     "  - run on the HOST (Docker has no camera)\n"
                     "  - System Settings → Privacy & Security → Camera → "
                     "enable your terminal, then RE-RUN\n"
                     "  - or pick a device explicitly: --camera 1\n";
        return false;
    }
    std::cout << who << ": camera delivering " << firstFrame.cols << "x"
              << firstFrame.rows << " (" << cap.getBackendName() << ")\n";
    return true;
}

// FaceTracker config for the realtime paths (speed over polish), shared by
// live-cpu and the transfer driver. Lighting/albedo and the per-pixel shape
// refinement run only at personalise; tracking is warm-started landmarks.
static FaceTracker::Config liveConfig(double sparseReg, const LiveOptions& lo)
{
    FaceTracker::Config tc;
    tc.sparseReg               = sparseReg;
    tc.exprRegPersonalise      = kExprRegWeight;
    tc.exprRegTrack            = kTrackExprRegWeight;
    tc.albedoRegWeight         = kAlbedoRegWeight;
    tc.smoothAlpha             = kSmoothAlpha;
    tc.contourItersPersonalise = kContourOuterIters;
    tc.photoIterations         = 20;      // one-off → keep startup snappy
    tc.photoPixelStride        = 2;
    tc.personaliseOptimizeShape = false;  // per-pixel shape refine too slow live
    tc.contourItersTrack       = 3;       // warm-started → converges fast
    tc.trackPhotoIterations    = 1;
    tc.trackPhotoOptimizePose  = false;   // lighting = pure linear estimate
    tc.lightingEvery           = 10;
    tc.photoRefine             = lo.photoRefine;
    tc.optimizeFocal           = lo.optimizeFocal;
    return tc;
}

// Personalise-quality gate: identity (and optional focal) is fit once and kept
// for the whole session, so refusing a turned/tiny first face is cheap insurance.
// Yaw proxy: the nose tip sits near the pupil midpoint when frontal and shifts
// toward one eye under yaw (pupils 4540/11681, nose 8156 in both landmark sets).
static bool frontalEnough(const std::vector<LandmarkObservation>& obs)
{
    Eigen::Vector2d nose(-1, -1), eyeR(-1, -1), eyeL(-1, -1);
    for (const LandmarkObservation& o : obs) {
        if      (o.vertexIndex ==  8156) nose = o.imagePoint;
        else if (o.vertexIndex ==  4540) eyeR = o.imagePoint;
        else if (o.vertexIndex == 11681) eyeL = o.imagePoint;
    }
    if (nose.x() < 0 || eyeR.x() < 0 || eyeL.x() < 0) return false;
    const double eyeDist = (eyeL - eyeR).norm();
    if (eyeDist < 25.0) return false;                        // face too small/far
    const Eigen::Vector2d mid = 0.5 * (eyeL + eyeR);
    return std::abs(nose.x() - mid.x()) < 0.35 * eyeDist;    // |yaw| ≲ 25–30°
}

// Keypoint debug view: the camera frame with the detected observations
// (green = interior, cyan = jaw contour) and — once personalised — the
// corresponding PROJECTED model vertices (red crosses + yellow error vector),
// so a bad pose/landmark is visible as a long yellow line.
static cv::Mat drawKeypointsDebug(const cv::Mat&                          frame,
                                  const std::vector<LandmarkObservation>& obs,
                                  const proj::Pixels*                     projected)
{
    cv::Mat out = frame.clone();
    for (const LandmarkObservation& o : obs) {
        const cv::Point det(static_cast<int>(o.imagePoint.x() + 0.5),
                            static_cast<int>(o.imagePoint.y() + 0.5));
        if (o.vertexIndex < 0) {                       // contour (matched later)
            cv::circle(out, det, 3, {255, 255, 0}, -1, cv::LINE_AA);
            continue;
        }
        cv::circle(out, det, 3, {0, 220, 0}, -1, cv::LINE_AA);
        if (projected && o.vertexIndex < projected->rows() &&
            (*projected)(o.vertexIndex, 0) >= 0) {
            const cv::Point mdl(
                static_cast<int>((*projected)(o.vertexIndex, 0) + 0.5f),
                static_cast<int>((*projected)(o.vertexIndex, 1) + 0.5f));
            cv::line(out, det, mdl, {0, 255, 255}, 1, cv::LINE_AA);
            cv::drawMarker(out, mdl, {0, 0, 255}, cv::MARKER_CROSS, 7, 1,
                           cv::LINE_AA);
        }
    }
    cv::putText(out, "green detected | red model | cyan contour", {8, 24},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 220, 0}, 1, cv::LINE_AA);
    return out;
}

// Realtime reconstruction from the camera. Templated on the display renderer
// (CPU Renderer or CudaRenderer); tracking + photometric stay on the CPU inside
// FaceTracker, only the overlay render is swapped. --photo-gpu moves the
// photometric solve to the GPU as well.
template <class RendererT>
static void runLiveImpl(const BFMLoader& bfm, double sparseReg, const LiveOptions& lo)
{
    cv::VideoCapture cap;
    cv::Mat raw;
    if (!openLiveCapture(cap, lo, raw, "live")) return;

    cv::Mat frame = resizeToWidth(raw, lo.width);
    const int W = frame.cols, H = frame.rows;

    // Unknown webcam intrinsics: start from a 60°-HFOV guess; personalisation
    // refines the focal (unless --no-optimize-focal).
    const Eigen::Matrix3f K0 = proj::defaultIntrinsics(W, H);
    std::cout << "live: " << W << "x" << H << " @ focal guess " << K0(0, 0)
              << " px" << (lo.optimizeFocal ? " (optimised during personalise)" : "")
              << "\n      keys: q quit | p re-personalise | s snapshot\n";

    // Landmark source. Default: the MediaPipe coprocess (same dense 21+14 set as
    // the offline modes — chin/brows/eye corners make pitch and identity
    // observable). An explicit --detector yunet|lbf skips it; YuNet is the fallback.
    MpLandmarkStream mpStream;
    if (kDetector == "mediapipe" || !kDetectorExplicit) {
        if (!mpStream.start())
            std::cout << "live: MediaPipe landmark server unavailable (needs a "
                         "python3 with mediapipe installed — pip install "
                         "mediapipe==0.10.18) — using YuNet\n";
    }
    LandmarkDetector detector(kLbfModelPath,
                              kDetector == "lbf" ? "" : kYuNetPath);
    if (!detector.ok() && !mpStream.running()) {
        std::cerr << "live: landmark detector unavailable (models/ missing?)\n";
        return;
    }

    FaceTracker tracker(bfm, K0, liveConfig(sparseReg, lo));

    const RendererT renderer(H, W, bfm.faces());
    const std::string liveDir = outDir("live");
    double fpsEma = 0.0;
    long processed = 0, written = 0;

    for (;;) {
        if (!cap.read(raw) || raw.empty()) break;
        const auto t0 = std::chrono::steady_clock::now();
        frame = resizeToWidth(raw, lo.width);

        std::vector<LandmarkObservation> obs;
        {
            ScopedTimer t("live/detect");
            obs = mpStream.running() ? mpStream.detect(frame)
                                     : detector.detect(frame);
        }

        bool tracked = false;
        if (!tracker.personalised()) {
            // Identity is locked in for the whole session by this one frame —
            // only personalise on a frontal-enough, large-enough face.
            if (!obs.empty() && frontalEnough(obs)) {
                ScopedTimer t("live/personalise");
                tracked = tracker.personalise(frame, obs, lo.initZ);
            }
        } else {
            ScopedTimer t("live/track");
            tracked = tracker.track(frame, obs, lo.initZ);
        }

        // Rendered-model views (shared by the overlay and the mask window).
        cv::Mat vis, maskVis, kpVis;
        if (tracker.personalised()) {
            ScopedTimer t("live/render");
            const Eigen::MatrixX3f shape = tracker.currentShape();
            const RenderInput in{
                .shape  = shape,
                .albedo = tracker.currentAlbedo(),
                .R      = tracker.pose().rotationMatrix(),
                .t      = tracker.pose().translation.cast<float>(),
                .K      = tracker.K(),
                .sh     = tracker.sh() };
            const RenderOutput r = renderer.render(in);
            vis = blendRenderOnPhoto(r, frame);

            // Mask debug window: the reconstruction alone on black, annotated
            // with the current fit state.
            cv::cvtColor(r.image, maskVis, cv::COLOR_RGB2BGR);
            maskVis.convertTo(maskVis, CV_8UC3, 255.0);
            {
                const PoseParameters&  p  = tracker.pose();
                const Eigen::VectorXd& ex = tracker.expr();
                std::ostringstream l1, l2, l3;
                l1 << "t [mm]  " << std::fixed << std::setprecision(1)
                   << p.translation.x() << "  " << p.translation.y() << "  "
                   << p.translation.z();
                l2 << "rot [deg]  " << std::fixed << std::setprecision(1)
                   << p.angleAxis.x() * 180.0 / M_PI << "  "
                   << p.angleAxis.y() * 180.0 / M_PI << "  "
                   << p.angleAxis.z() * 180.0 / M_PI;
                l3 << "|id| " << std::setprecision(2) << tracker.identity().norm()
                   << "  |expr| " << (ex.size() ? ex.norm() : 0.0)
                   << "  f " << static_cast<int>(tracker.K()(0, 0)) << "px";
                int y = 22;
                for (const std::ostringstream* s : {&l1, &l2, &l3}) {
                    cv::putText(maskVis, s->str(), {8, y},
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 220, 0}, 1,
                                cv::LINE_AA);
                    y += 22;
                }
            }

            // Keypoint debug window: detections + projected model landmarks.
            const proj::Pixels projected = proj::projectMesh(
                shape, in.R, in.t, tracker.K());
            kpVis = drawKeypointsDebug(frame, obs, &projected);
        } else {
            vis     = frame.clone();
            maskVis = cv::Mat::zeros(frame.size(), CV_8UC3);
            cv::putText(maskVis, "not personalised yet", {8, 22},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 220, 0}, 1,
                        cv::LINE_AA);
            kpVis = drawKeypointsDebug(frame, obs, nullptr);
        }

        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        fpsEma = fpsEma <= 0.0 ? 1000.0 / ms : 0.9 * fpsEma + 0.1 * (1000.0 / ms);
        std::ostringstream hud;
        hud << (tracker.personalised() ? (tracked ? "track" : "hold")
                                       : "looking for a frontal face...")
            << "  " << std::fixed << std::setprecision(1) << fpsEma << " fps"
            << "  f=" << static_cast<int>(tracker.K()(0, 0)) << "px";
        cv::putText(vis, hud.str(), {8, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    {0, 220, 0}, 2, cv::LINE_AA);

        ++processed;
        if (lo.maxFrames > 0) {                 // headless test: dump frames
            std::ostringstream n;
            n << liveDir << "/live_" << std::setw(4) << std::setfill('0')
              << written++;
            cv::imwrite(n.str() + ".png", vis);
            cv::imwrite(n.str() + "_mask.png", maskVis);
            cv::imwrite(n.str() + "_kp.png", kpVis);
        }
        if (lo.display) {
            cv::imshow("face2face live", vis);
            cv::imshow("face2face mask", maskVis);
            cv::imshow("face2face keypoints", kpVis);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) break;
            if (key == 'p') {                   // re-personalise from scratch
                tracker.reset(K0);
                std::cout << "live: re-personalising…\n";
            }
            if (key == 's') {
                cv::imwrite(liveDir + "/snapshot.png", vis);
                cv::imwrite(liveDir + "/snapshot_mask.png", maskVis);
                cv::imwrite(liveDir + "/snapshot_kp.png", kpVis);
            }
        }
        if (lo.maxFrames > 0 && processed >= lo.maxFrames) break;
    }
    std::cout << "live: processed " << processed << " frames @ "
              << std::fixed << std::setprecision(1) << fpsEma << " fps";
    if (written > 0) std::cout << ", wrote " << written << " → " << liveDir;
    std::cout << '\n';
}

// Live expression transfer: personalise the target avatar once, then drive its
// expression from the webcam via a neutral-relative delta copy
//   δ_target = δ_target_neutral + (δ_driver − δ_driver_neutral).
// Templated on the display renderer so --gpu-render can draw the avatar on CUDA.
template <class RendererT>
static void runTransferLive(const BFMLoader& bfm, double sparseReg,
                            const LiveOptions& lo)
{
    // 1. Personalise the TARGET avatar (Biwi frame 0, + depth)
    Eigen::VectorXd  tgtAlpha, tgtNeutral;
    Eigen::MatrixX3f tgtAlbedo;
    light::SHCoeffs  tgtSh;
    cv::Mat          tgtImg;           // the target's own photo (frame 0)
    Eigen::Matrix3f  tgtImgK;          // its intrinsics
    PoseParameters   tgtPose;          // its personalised head pose
    try {
        BiwiLoader biwi(lo.transferTarget, 1);
        const std::vector<BiwiFrame> tf = biwi.getFrames();
        if (tf.empty()) { std::cerr << "transfer: no frames in " << lo.transferTarget << '\n'; return; }
        const BiwiCalibration cal = biwi.getCalibration();
        const BiwiFrame& f = tf[0];
        std::vector<LandmarkObservation> obs;
        std::ostringstream mp;
        mp << lo.transferTarget << "/landmarks_mp_" << std::setw(5)
           << std::setfill('0') << f.frameNumber << ".txt";
        try { obs = loadLandmarkObservations(mp.str()); } catch (const std::exception&) {}
        if (obs.empty()) {
            LandmarkDetector d(kLbfModelPath, kYuNetPath); obs = d.detect(f.rgb);
        }
        if (obs.empty()) { std::cerr << "transfer: no landmarks for target\n"; return; }
        std::vector<Eigen::Vector3d> head = cropHead(
            backprojectDepth(f.depth, cal.K_depth, kDepthBackprojStride),
            f.headCenter, kDepthCropRadiusMM, kDepthCropFrontSlabMM);
        for (Eigen::Vector3d& p : head) p = cal.R_rgb * p + cal.t_rgb;
        const Eigen::Vector3d headRgb = cal.R_rgb * f.headCenter + cal.t_rgb;
        FaceTracker tt(bfm, cal.K_rgb, offlineConfig(sparseReg));
        if (!tt.personalise(f.rgb, obs, headRgb.z(), &head)) {
            std::cerr << "transfer: target personalise failed\n"; return; }
        tgtAlpha = tt.identity(); tgtAlbedo = tt.currentAlbedo();
        tgtSh = tt.sh();          tgtNeutral = tt.expr();
        tgtImg = f.rgb.clone();   tgtImgK = cal.K_rgb;   tgtPose = tt.pose();
        std::cout << "transfer: target '" << lo.transferTarget
                  << "' personalised (|id|=" << tgtAlpha.norm() << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "transfer: target load failed: " << e.what() << '\n'; return;
    }

    // 2. Camera / source for the DRIVING actor (you)
    cv::VideoCapture cap; cv::Mat raw;
    if (!openLiveCapture(cap, lo, raw, "transfer")) return;
    cv::Mat frame = resizeToWidth(raw, lo.width);
    const int W = frame.cols, H = frame.rows;
    const Eigen::Matrix3f K0 = proj::defaultIntrinsics(W, H);

    MpLandmarkStream mpStream;
    if (kDetector == "mediapipe" || !kDetectorExplicit) mpStream.start();
    LandmarkDetector detector(kLbfModelPath, kDetector == "lbf" ? "" : kYuNetPath);
    if (!detector.ok() && !mpStream.running()) {
        std::cerr << "transfer: no landmark detector\n"; return; }

    FaceTracker driver(bfm, K0, liveConfig(sparseReg, lo));

    const int TH = H, TW = H;                       // square target panel
    const Eigen::Matrix3f tgtK = proj::defaultIntrinsics(TW, TH);
    const RendererT tgtRenderer(TH, TW, bfm.faces());
    // Renderer for the driven avatar over the target's OWN photo (its pose).
    const RendererT tgtOverlayRenderer(tgtImg.rows, tgtImg.cols, bfm.faces());
    PoseParameters frontal; frontal.translation = Eigen::Vector3d(0, 0, kFrontalRenderDepthMM);

    Eigen::VectorXd userNeutral; bool haveUserNeutral = false;
    const std::string outDirT = outDir("transfer");
    std::cout << "transfer: driving with your expressions. keys: q quit | "
                 "p re-personalise | n set neutral | s snapshot\n";
    long processed = 0, written = 0; double fpsEma = 0.0;

    for (;;) {
        if (!cap.read(raw) || raw.empty()) break;
        const auto t0 = std::chrono::steady_clock::now();
        frame = resizeToWidth(raw, lo.width);
        std::vector<LandmarkObservation> obs =
            mpStream.running() ? mpStream.detect(frame) : detector.detect(frame);

        if (!driver.personalised()) {
            if (!obs.empty() && frontalEnough(obs) &&
                driver.personalise(frame, obs, lo.initZ)) {
                userNeutral = driver.expr(); haveUserNeutral = true;   // rest pose
                std::cout << "transfer: driver personalised — hold neutral, or press 'n'\n";
            }
        } else {
            driver.track(frame, obs, lo.initZ);
        }

        // Render the target avatar with the NEUTRAL-RELATIVE transferred δ, in
        // two views: (a) frontal on black, (b) over the target's OWN photo at
        // its personalised pose (the target person making your expression).
        cv::Mat tgtVis = cv::Mat::zeros(TH, TW, CV_8UC3);
        cv::Mat tgtOverlay = tgtImg.clone();
        if (driver.personalised()) {
            Eigen::VectorXd d = tgtNeutral;
            const Eigen::VectorXd ue = driver.expr();
            if (haveUserNeutral && ue.size() == userNeutral.size() &&
                tgtNeutral.size() == ue.size())
                d = tgtNeutral + (ue - userNeutral);
            d = d.cwiseMax(-3.5).cwiseMin(3.5);
            const Eigen::MatrixX3f ts =
                bfm.shape(tgtAlpha.cast<float>(), d.cast<float>());
            const RenderInput frontIn{ .shape = ts, .albedo = tgtAlbedo,
                .R = frontal.rotationMatrix(), .t = frontal.translation.cast<float>(),
                .K = tgtK, .sh = tgtSh };
            cv::cvtColor(tgtRenderer.render(frontIn).image, tgtVis, cv::COLOR_RGB2BGR);
            tgtVis.convertTo(tgtVis, CV_8UC3, 255.0);

            const RenderInput overIn{ .shape = ts, .albedo = tgtAlbedo,
                .R = tgtPose.rotationMatrix(), .t = tgtPose.translation.cast<float>(),
                .K = tgtImgK, .sh = tgtSh };
            tgtOverlay = blendRenderOnPhoto(tgtOverlayRenderer.render(overIn), tgtImg);
        }

        // Composite: your camera (left) | target avatar (right).
        cv::Mat driverVis = frame.clone();
        for (const LandmarkObservation& o : obs)
            cv::circle(driverVis, {int(o.imagePoint.x()), int(o.imagePoint.y())}, 2,
                       o.vertexIndex < 0 ? cv::Scalar(255,255,0) : cv::Scalar(0,220,0),
                       -1, cv::LINE_AA);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        fpsEma = fpsEma <= 0.0 ? 1000.0/ms : 0.9*fpsEma + 0.1*(1000.0/ms);
        std::ostringstream hud;
        hud << (driver.personalised() ? "driving" : "hold a frontal face...")
            << "  " << std::fixed << std::setprecision(1) << fpsEma << " fps";
        cv::putText(driverVis, hud.str(), {8, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    {0, 220, 0}, 2, cv::LINE_AA);
        cv::putText(tgtVis, "target avatar", {8, 24},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 220, 0}, 1, cv::LINE_AA);
        // Match the overlay panel to the row height for hconcat.
        if (tgtOverlay.rows != H) {
            const double r = static_cast<double>(H) / tgtOverlay.rows;
            cv::resize(tgtOverlay, tgtOverlay, {}, r, r, cv::INTER_AREA);
        }
        cv::putText(tgtOverlay, "target over photo", {8, 24},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 220, 0}, 1, cv::LINE_AA);
        cv::Mat composite;
        cv::hconcat(std::vector<cv::Mat>{driverVis, tgtVis, tgtOverlay}, composite);

        ++processed;
        if (lo.maxFrames > 0) {
            std::ostringstream n; n << outDirT << "/transfer_" << std::setw(4)
                << std::setfill('0') << written++ << ".png";
            cv::imwrite(n.str(), composite);
        }
        if (lo.display) {
            cv::imshow("face2face transfer", composite);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) break;
            if (key == 'p') { driver.reset(K0); haveUserNeutral = false; }
            if (key == 'n' && driver.personalised()) {   // recapture neutral
                userNeutral = driver.expr(); haveUserNeutral = true;
                std::cout << "transfer: neutral recaptured\n";
            }
            if (key == 's') cv::imwrite(outDirT + "/snapshot.png", composite);
        }
        if (lo.maxFrames > 0 && processed >= lo.maxFrames) break;
    }
    std::cout << "transfer: processed " << processed << " frames";
    if (written) std::cout << ", wrote " << written << " → " << outDirT;
    std::cout << '\n';
}

// ENTRY POINTS: live-cpu / live-gpu — thin wrappers over runLiveImpl<>.
static void runLiveCpu(const BFMLoader& bfm, double sparseReg, const LiveOptions& lo)
{
    runLiveImpl<Renderer>(bfm, sparseReg, lo);
}

#ifdef USE_CUDA
// Same realtime loop as live-cpu, but the overlay is rendered by the CUDA
// rasteriser. Tracking stays on the CPU; add --photo-gpu to also move the
// photometric solve to the GPU.
static void runLiveGpu(const BFMLoader& bfm, double sparseReg, const LiveOptions& lo)
{
    std::cout << "live-gpu: CUDA display renderer active "
                 "(tracking on CPU; add --photo-gpu for the GPU photometric solve)\n";
    runLiveImpl<CudaRenderer>(bfm, sparseReg, lo);
}
#else
static void runLiveGpu(const BFMLoader& bfm, double sparseReg, const LiveOptions& lo)
{
    (void)bfm; (void)sparseReg; (void)lo;
    std::cerr << "live-gpu: this binary was built without CUDA support.\n"
                 "  Rebuild on an NVIDIA machine with:  make USE_CUDA=1\n"
                 "  (optionally CUDA_ARCH=sm_XX). Use --mode live-cpu meanwhile.\n";
}
#endif

#ifdef USE_CUDA
// ENTRY POINT: verify-gpu — CPU vs CUDA renderer parity + speed on the mean face
// (deterministic; needs only the BFM). The correctness gate for the GPU
// rasteriser: it must reproduce the CPU G-buffer, not just a similar image.
static void runGpuCheck(const BFMLoader&        bfm,
                        const Eigen::MatrixX3f& shape,
                        const Eigen::MatrixX3f& albedo)
{
    constexpr int RH = 480, RW = 640;
    const RenderInput in{
        .shape  = shape,
        .albedo = albedo,
        .R      = Eigen::Matrix3f::Identity(),
        .t      = Eigen::Vector3f(0.0f, 0.0f, kFrontalRenderDepthMM),
        .K      = proj::defaultIntrinsics(RW, RH),
        .sh     = light::defaultWhite(),
    };
    const Renderer     cpu(RH, RW, bfm.faces());
    const CudaRenderer gpu(RH, RW, bfm.faces());
    const RenderOutput rc = cpu.render(in);
    const RenderOutput rg = gpu.render(in);

    double imgMax = 0.0, imgSum = 0.0, baryMax = 0.0, depthMax = 0.0;
    long   maskMismatch = 0, triMismatch = 0, covered = 0, both = 0;
    for (int y = 0; y < RH; ++y)
        for (int x = 0; x < RW; ++x) {
            const cv::Vec3f a = rc.image.at<cv::Vec3f>(y, x);
            const cv::Vec3f b = rg.image.at<cv::Vec3f>(y, x);
            for (int c = 0; c < 3; ++c) {
                const double d = std::abs(a[c] - b[c]);
                imgMax = std::max(imgMax, d); imgSum += d;
            }
            const bool mc = rc.mask.at<uchar>(y, x) != 0;
            const bool mg = rg.mask.at<uchar>(y, x) != 0;
            if (mc) ++covered;
            if (mc != mg) ++maskMismatch;
            const int tc = rc.triIdx.at<int>(y, x), tg = rg.triIdx.at<int>(y, x);
            if (mc && mg) {
                ++both;
                if (tc != tg) ++triMismatch;
                else {
                    const cv::Vec3f ba = rc.bary.at<cv::Vec3f>(y, x);
                    const cv::Vec3f bb = rg.bary.at<cv::Vec3f>(y, x);
                    for (int c = 0; c < 3; ++c)
                        baryMax = std::max(baryMax, (double)std::abs(ba[c] - bb[c]));
                    depthMax = std::max(depthMax,
                        (double)std::abs(rc.depth.at<float>(y, x) - rg.depth.at<float>(y, x)));
                }
            }
        }

    std::cout << "\n== verify-gpu: CPU vs CUDA on the mean face (" << RW << "x" << RH << ") ==\n"
              << "  CPU covered pixels      : " << covered << "\n"
              << "  mask mismatches         : " << maskMismatch << "\n"
              << "  triIdx mismatches       : " << triMismatch
              << " (of " << both << " shared)\n"
              << "  image  max|Δ| / mean|Δ| : " << imgMax << " / " << (imgSum / (RW * RH * 3)) << "\n"
              << "  bary   max|Δ| (same tri): " << baryMax << "\n"
              << "  depth  max|Δ| mm        : " << depthMax << "\n";

    constexpr int WARMUP = 3, N = 50;
    for (int i = 0; i < WARMUP; ++i) { cpu.render(in); gpu.render(in); }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) cpu.render(in);
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) gpu.render(in);
    const auto t2 = std::chrono::steady_clock::now();
    const double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
    const double gpuMs = std::chrono::duration<double, std::milli>(t2 - t1).count() / N;
    std::cout << "  render time  CPU " << std::fixed << std::setprecision(2) << cpuMs
              << " ms   GPU " << gpuMs << " ms   speedup "
              << std::setprecision(1) << (gpuMs > 0 ? cpuMs / gpuMs : 0.0) << "x\n\n";

    const std::string dbg = outDir("debug");
    writeColourImage(rc, dbg + "/gpu_check_cpu.png");
    writeColourImage(rg, dbg + "/gpu_check_gpu.png");
}
#endif

// Render the mean face into a synthetic camera and dump depth + colour PNGs.
// Verifies the full forward pipeline (project → cull → rasterize → shade).
static void renderMeanFace(const BFMLoader&         bfm,
                           const Eigen::MatrixX3f&  shape,
                           const Eigen::MatrixX3f&  albedo)
{
    constexpr int RH = 480, RW = 640;

    RenderInput input{
        .shape  = shape,
        .albedo = albedo,
        .R      = Eigen::Matrix3f::Identity(),
        .t      = Eigen::Vector3f(0.0f, 0.0f, kFrontalRenderDepthMM),
        .K      = proj::defaultIntrinsics(RW, RH),
        .sh     = light::defaultWhite(),
    };

    const RenderOutput r = Renderer(RH, RW, bfm.faces()).render(input);

    const int covered = cv::countNonZero(r.mask);
    std::cout << "render: " << covered << " / " << (RW * RH) << " pixels covered\n";

    const std::string dbg = outDir("debug");
    writeDepthVis  (r, dbg + "/render_depth.png");
    writeColourImage(r, dbg + "/render_image.png");
}

// Biwi: sparse + dense in the same camera system

// Sparse landmark fit on the Biwi RGB frame using the real rgb.cal intrinsics.
// The pose is metric and transfers into the depth camera via the extrinsics
// (see fitDenseOnBiwi). Landmarks come from the per-frame pre-pass files.
static std::optional<FitParameters> fitSparseOnBiwi(
    const BFMLoader&        bfm,
    const Eigen::MatrixX3f& meanShape,
    const Eigen::MatrixX3f& albedo)
{
    try {
        BiwiLoader biwi(kBiwiDir, 1);
        const std::vector<BiwiFrame> frames = biwi.getFrames();
        if (frames.empty() || frames[0].rgb.empty()) {
            std::cout << "Biwi sparse: no RGB frame in " << kBiwiDir << '\n';
            return std::nullopt;
        }
        const BiwiFrame& frame = frames[0];
        const BiwiCalibration cal = biwi.getCalibration();

        std::ostringstream landmarkPath;
        landmarkPath << kBiwiDir << "/landmarks_" << std::setw(5)
                     << std::setfill('0') << frame.frameNumber << ".txt";
        const std::vector<LandmarkObservation> observations =
            loadLandmarkObservations(landmarkPath.str());

        // Subject sits ~1 m from the Kinect → init + z-bounds accordingly.
        PoseParameters init;
        init.translation = Eigen::Vector3d(0.0, 0.0, 900.0);

        const PoseParameters poseOnly =
            CeresFitter::fitPose(meanShape, observations, cal.K_rgb, init,
                                 /*zMin=*/400.0, /*zMax=*/2000.0);

        const FitParameters poseAndShape =
            CeresFitter::fitPoseAndShape(
                meanShape, bfm.shape_basis_raw(), bfm.shape_sigma(),
                observations, cal.K_rgb, poseOnly,
                /*regularizationWeight=*/100.0, /*zMin=*/400.0, /*zMax=*/2000.0);

        const Eigen::MatrixX3f fittedShape =
            bfm.shape(poseAndShape.shapeCoefficients.cast<float>());

        writeFitOutputs("sparse", frame.rgb, bfm, fittedShape, albedo,
                        cal.K_rgb, poseAndShape.pose, /*tag=*/"biwi",
                        &observations);

        // Frontal portrait of the Stage-1 shape (same camera as the dense
        // portraits) — for the stages figure. Expected to look almost like the
        // mean face (reg=100, 9 landmarks barely constrain the shape).
        {
            constexpr int RH = 480, RW = 640;
            const RenderInput portrait{
                .shape  = fittedShape,
                .albedo = albedo,
                .R      = Eigen::Matrix3f::Identity(),
                .t      = Eigen::Vector3f(0.0f, 0.0f, 350.0f),
                .K      = proj::defaultIntrinsics(RW, RH),
                .sh     = light::defaultWhite(),
            };
            writeColourImage(Renderer(RH, RW, bfm.faces()).render(portrait),
                             outDir("biwi_sparse") + "/face_render.png");
        }
        return poseAndShape;
    }
    catch (const std::exception& exception) {
        std::cerr << "Biwi sparse fit skipped: " << exception.what() << '\n';
        return std::nullopt;
    }
}

// Dense ICP fit on the Biwi depth frame: head centre from the GT pose, sparseInit
// transformed into the depth camera via the extrinsics, and the result rendered
// back into the RGB camera and blended over the photo.
static void fitDenseOnBiwi(const BFMLoader&        bfm,
                           const Eigen::MatrixX3f& meanShape,
                           const Eigen::MatrixX3f& albedo,
                           const FitParameters*    sparseInit = nullptr,
                           int                     icpIterations = 30)
{
    try {
        BiwiLoader biwi(kBiwiDir, 1);
        const std::vector<BiwiFrame> frames = biwi.getFrames();
        if (frames.empty() || frames[0].depth.empty()) {
            std::cout << "Biwi: no depth frame in " << kBiwiDir << '\n';
            return;
        }
        const BiwiFrame& frame = frames[0];
        const BiwiCalibration cal = biwi.getCalibration();
        const Eigen::Matrix3f K = cal.K_depth;

        std::vector<Eigen::Vector3d> cloud =
            backprojectDepth(frame.depth, K, /*stride=*/1);

        // Tight crop (90/80): keep only the face (~150 mm), else the
        // coefficients rail. Centre = the dataset's GT head position.
        std::vector<Eigen::Vector3d> head =
            cropHead(cloud, frame.headCenter, 90.0, 80.0);
        std::cout << "Biwi dense: " << cloud.size() << " cloud pts → "
                  << head.size() << " head pts (GT centre "
                  << frame.headCenter.transpose() << " mm)\n";

        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (const Eigen::Vector3d& p : head) centroid += p;
        centroid /= static_cast<double>(head.size());

        PoseParameters init;
        init.angleAxis.setZero();
        if (sparseInit) {
            // Map the full Stage-1 pose from the RGB into the depth camera:
            //   p_rgb = R_ext·p_depth + t_ext
            //   ⇒ R_depth = R_extᵀ·R_sparse,  t_depth = R_extᵀ·(t_sparse − t_ext)
            const Eigen::Matrix3d Rs =
                sparseInit->pose.rotationMatrix().cast<double>();
            const Eigen::Matrix3d Rd = cal.R_rgb.transpose() * Rs;
            const Eigen::AngleAxisd aa(Rd);
            init.angleAxis = aa.angle() * aa.axis();

            const Eigen::Vector3d td =
                cal.R_rgb.transpose() * (sparseInit->pose.translation - cal.t_rgb);
            std::cout << "Dense init: Stage-1 pose transferred via extrinsics "
                      << "(angle-axis " << init.angleAxis.transpose() << ")\n"
                      << "  Stage-1 translation (depth camera): "
                      << td.transpose() << " mm\n";
        }
        // Translation init still via the cloud centroid (with the −M·c0
        // compensation, because the BFM mean face is not origin-centred,
        // c0 ≈ (0,−5,56) mm): landmarks constrain Z only weakly while the ICP
        // basin is only ~56 mm. The rotation (in full mode) comes from Stage 1.
        const Eigen::Vector3d c0 =
            meanShape.colwise().mean().transpose().cast<double>();
        init.translation =
            centroid - init.rotationMatrix().cast<double>() *
                           (proj::BFM_TO_CAM.cast<double>() * c0);

        const cv::Mat depthViz = depthToBgr(frame.depth);
        const std::string progressDir = outDir("biwi_dense") + "/icp_progress";
        std::filesystem::create_directories(progressDir);
        const Renderer depthRenderer(depthViz.rows, depthViz.cols, bfm.faces());

        const DenseIterationCallback writeProgressImage =
            [&](int iteration, const FitParameters& current, double rmseMM) {
                const RenderInput in{
                    .shape  = bfm.shape(current.shapeCoefficients.cast<float>()),
                    .albedo = albedo,
                    .R      = current.pose.rotationMatrix(),
                    .t      = current.pose.translation.cast<float>(),
                    .K      = K,
                    .sh     = light::defaultWhite(),
                };
                cv::Mat progress =
                    blendRenderOnPhoto(depthRenderer.render(in), depthViz);

                std::ostringstream label;
                label << "icp " << std::setw(2) << std::setfill('0') << iteration
                      << " | RMSE " << std::fixed << std::setprecision(1)
                      << rmseMM << " mm";
                cv::putText(progress, label.str(), {8, 20},
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 200, 0}, 1,
                            cv::LINE_AA);

                std::ostringstream name;
                name << progressDir << "/icp_" << std::setw(2)
                     << std::setfill('0') << iteration << ".png";
                cv::imwrite(name.str(), progress);
            };

        // Scale the regulariser with the point count: E_data grows with the number
        // of target points but E_reg does not, so a constant reg would rail the
        // coefficients on Biwi's ~8700 head points. Base 150 personalises without
        // flattening the shape back toward the mean.
        const double regWeight = 150.0 * (static_cast<double>(head.size()) / 600.0);

        const FitParameters dense = CeresFitter::fitDense(
            meanShape, bfm.shape_basis_raw(), bfm.shape_sigma(), head, init,
            regWeight,
            icpIterations,
            /*trimPercentile=*/80.0,
            /*vertexStride=*/8,
            /*pointToPlaneWeight=*/1.0,
            writeProgressImage);

        const Eigen::MatrixX3f fitted =
            bfm.shape(dense.shapeCoefficients.cast<float>());

        const Eigen::Matrix3f R = dense.pose.rotationMatrix();
        const Eigen::Vector3f t = dense.pose.translation.cast<float>();

        // Overlay 1: depth camera.
        RenderInput depthInput{
            .shape  = fitted,
            .albedo = albedo,
            .R      = R,
            .t      = t,
            .K      = K,
            .sh     = light::defaultWhite(),
        };
        overlayRenderOnPhoto(depthRenderer.render(depthInput), depthViz,
                             outDir("biwi_dense") + "/depth_overlay.png");

        // Overlay 2 — the Biwi payoff: the same fitted pose transformed into
        // the RGB camera via the extrinsics and blended over the real photo.
        // Model → depth camera is R·(M·v)+t, so:
        //   p_rgb = R_ext·p_depth + t_ext = (R_ext·R)·(M·v) + (R_ext·t + t_ext)
        const Eigen::Matrix3f Rrgb = cal.R_rgb.cast<float>() * R;
        const Eigen::Vector3f trgb =
            cal.R_rgb.cast<float>() * t + cal.t_rgb.cast<float>();
        const Renderer rgbRenderer(frame.rgb.rows, frame.rgb.cols, bfm.faces());
        RenderInput rgbInput{
            .shape  = fitted,
            .albedo = albedo,
            .R      = Rrgb,
            .t      = trgb,
            .K      = cal.K_rgb,
            .sh     = light::defaultWhite(),
        };
        overlayRenderOnPhoto(rgbRenderer.render(rgbInput), frame.rgb,
                             outDir("biwi_dense") + "/rgb_overlay.png");

        // 3-panel mask visualisation (overlay | recon @ pose | recon frontal),
        // in the RGB camera.
        {
            const Eigen::AngleAxisd rgbAA(Rrgb.cast<double>());
            PoseParameters rgbPose;
            rgbPose.angleAxis   = rgbAA.angle() * rgbAA.axis();
            rgbPose.translation = trgb.cast<double>();
            const cv::Mat maskPanels = renderMaskPanels(
                frame.rgb, fitted, albedo, cal.K_rgb, rgbPose, bfm.faces());
            const std::string maskPath = outDir("biwi_dense") + "/mask_panels.png";
            cv::imwrite(maskPath, maskPanels);
            std::cout << "Wrote mask panels: " << maskPath << '\n';
        }

        // Frontal portrait of the fitted identity with the mean albedo (pure
        // GEOMETRY — the individual face shape straight-on).
        {
            constexpr int RH = 480, RW = 640;
            const RenderInput portrait{
                .shape  = fitted,
                .albedo = albedo,
                .R      = Eigen::Matrix3f::Identity(),
                .t      = Eigen::Vector3f(0.0f, 0.0f, 350.0f),
                .K      = proj::defaultIntrinsics(RW, RH),
                .sh     = light::defaultWhite(),
            };
            writeColourImage(Renderer(RH, RW, bfm.faces()).render(portrait),
                             outDir("biwi_dense") + "/face_render.png");
        }

        const Eigen::VectorXf disp = (fitted - meanShape).rowwise().norm();
        std::cout << "Biwi dense mean displacement: " << disp.mean()
                  << " mm, max " << disp.maxCoeff() << " mm\n";
    }
    catch (const std::exception& exception) {
        std::cerr << "Biwi dense fit skipped: " << exception.what() << '\n';
    }
}

// CLI usage (Biwi is the only dataset).
static void printUsage()
{
    std::cerr <<
      "Usage: face_recon --mode <MODE> [options]   (dataset: Biwi RGB-D only)\n\n"
      "Video reconstruction (personalise frame 0, then track the sequence):\n"
      "  rgb        landmarks + jaw contour + photometric      (no depth term)\n"
      "  rgbd       + the metric Kinect depth ICP term\n"
      "Realtime (Mac camera, HOST only — Docker has no camera):\n"
      "  live-cpu   the CPU tracker (this build)\n"
      "  transfer   live expression transfer onto a target avatar\n"
      "  live-gpu   CPU tracker + CUDA display renderer (needs make USE_CUDA=1)\n"
      "GPU renderer checks (needs make USE_CUDA=1):\n"
      "  verify-gpu CPU vs CUDA renderer parity + speed on the mean face\n"
      "Single-frame geometry (kept for reports):\n"
      "  dense      depth-only ICP\n"
      "  full       sparse landmark fit → dense ICP\n\n"
      "Options:\n"
      "  --biwi-dir <path> | --biwi-seq <NN>   Biwi subject folder\n"
      "  --frames <n>       frame count for rgb/rgbd (default 30)\n"
      "  --sparse-reg <λ>   identity/expression reg (default 30)\n"
      "  --icp-iters <n>    dense/full ICP rounds (default 30)\n"
      "  --detector <yunet|lbf|mediapipe>   landmark backend (default: yunet;\n"
      "                     live-cpu defaults to the MediaPipe coprocess and\n"
      "                     falls back to yunet if no python3 has mediapipe)\n"
      "  --camera <i> | --live-source <path> | --live-width <px>\n"
      "  --live-frames <n> --live-nodisplay   headless live test\n"
      "  --photo-refine     pyramid photometric pose refinement (live)\n"
      "  --optimize-focal   solve fx=fy during personalise (experimental)\n"
      "  --transfer-target <biwi-dir>   avatar to drive in --mode transfer\n"
      "  --bundle           multi-keyframe identity bundle (rgb/rgbd, Face2Face \u00a76)\n"
      "  --bundle-keyframes <k>   keyframes for --bundle (default 7)\n"
      "  --photo-gpu        photometric geometry solve on the GPU "
      "(finite-diff; needs make USE_CUDA=1)\n"
      "  --photo-gpu-analytic  GPU photometric solve with analytic Jacobian\n"
      "  --gpu-render       render on the GPU in --mode transfer (needs make USE_CUDA=1)\n"
      "  --photo-samples <K>  randomly use K covered pixels per photometric "
      "iteration (0 = all)\n"
      "  --photo-csv <path>   append per-iteration photometric convergence rows\n"
      "  --timers           print per-stage timings\n";
}

int main(int argc, char** argv)
{
    // CLI
    std::string mode      = "rgb";                 // see printUsage()
    double      sparseReg = kDefaultSparseReg;     // --sparse-reg
    int         icpIters  = 30;                     // --icp-iters (dense/full)
    int         numFrames = 30;                     // --frames (rgb/rgbd)
    LiveOptions live;                              // live-cpu options
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--sparse-reg" && i + 1 < argc) sparseReg = std::stod(argv[++i]);
        else if (arg == "--biwi-seq" && i + 1 < argc) {
            std::string seq = argv[++i];
            if (seq.size() == 1) seq = "0" + seq;
            kBiwiDir = "data/biwi/" + seq;
        }
        else if (arg == "--biwi-dir" && i + 1 < argc) kBiwiDir = argv[++i];   // e.g. data/BK-1/01
        else if (arg == "--icp-iters" && i + 1 < argc) icpIters = std::stoi(argv[++i]);
        else if (arg == "--frames" && i + 1 < argc) numFrames = std::stoi(argv[++i]);
        else if (arg == "--detector" && i + 1 < argc) {                      // yunet|lbf|mediapipe
            kDetector = argv[++i];
            kDetectorExplicit = true;
        }
        // live / realtime flags
        else if (arg == "--camera" && i + 1 < argc) live.camera = std::stoi(argv[++i]);
        else if (arg == "--live-source" && i + 1 < argc) live.source = argv[++i];
        else if (arg == "--live-width" && i + 1 < argc) live.width = std::stoi(argv[++i]);
        else if (arg == "--live-frames" && i + 1 < argc) live.maxFrames = std::stoi(argv[++i]);
        else if (arg == "--live-nodisplay") live.display = false;
        else if (arg == "--transfer-target" && i + 1 < argc) live.transferTarget = argv[++i];
        else if (arg == "--photo-refine") live.photoRefine = true;
        else if (arg == "--optimize-focal") live.optimizeFocal = true;
        else if (arg == "--bundle") kBundlePersonalise = true;
        else if (arg == "--bundle-keyframes" && i + 1 < argc) kBundleKeyframes = std::stoi(argv[++i]);
        else if (arg == "--photo-gpu") {
#ifdef USE_CUDA
            CeresFitter::usePhotometricGpu = true;
            std::cout << "photometric geometry solve: GPU (CUDA finite-diff LM)\n";
#else
            std::cerr << "--photo-gpu ignored: built without CUDA (rebuild: make USE_CUDA=1)\n";
#endif
        }
        else if (arg == "--photo-gpu-analytic") {
#ifdef USE_CUDA
            CeresFitter::usePhotometricGpu = true;
            CeresFitter::photoGpuAnalytic  = true;
            std::cout << "photometric geometry solve: GPU (CUDA analytic-Jacobian LM)\n";
#else
            std::cerr << "--photo-gpu-analytic ignored: built without CUDA (rebuild: make USE_CUDA=1)\n";
#endif
        }
        else if (arg == "--gpu-render") {
#ifdef USE_CUDA
            live.gpuRender = true;
#else
            std::cerr << "--gpu-render ignored: built without CUDA (rebuild: make USE_CUDA=1)\n";
#endif
        }
        else if (arg == "--photo-samples" && i + 1 < argc)
            CeresFitter::photoSamples = std::stoi(argv[++i]);
        else if (arg == "--photo-csv" && i + 1 < argc)
            CeresFitter::photoCsvPath = argv[++i];
        else if (arg == "--timers") ScopedTimer::enabled = true;
        else { std::cerr << "unknown argument: " << arg << "\n\n"; printUsage(); return 1; }
    }
    if (kDetector != "yunet" && kDetector != "lbf" && kDetector != "mediapipe") {
        std::cerr << "unknown --detector '" << kDetector
                  << "' (expected yunet | lbf | mediapipe)\n";
        return 1;
    }

    std::filesystem::create_directories(kOutDir);

    BFMLoader bfm(kBfmPath);
    bfm.summariseBFM(kBfmPath);
    const Eigen::MatrixX3f meanShape = bfm.mean_shape();
    const Eigen::MatrixX3f albedo    = bfm.albedo();   // mean albedo (β = 0)
    std::cout << "BFM: " << meanShape.rows() << " vertices, "
              << bfm.faces().rows() << " triangles\nMode: " << mode << '\n';

    // mode dispatch: one clear entry point per mode
    if (mode == "rgb" || mode == "rgbd") {
        runVideoReconstruction(bfm, albedo, numFrames, sparseReg,
                               /*useDepth=*/mode == "rgbd");
    } else if (mode == "live-cpu") {
        runLiveCpu(bfm, sparseReg, live);
    } else if (mode == "transfer") {
        if (live.transferTarget.empty()) {
            std::cerr << "transfer: need --transfer-target <biwi-dir> (the avatar "
                         "to drive), e.g. --transfer-target data/BK-1/05\n";
            return 1;
        }
#ifdef USE_CUDA
        if (live.gpuRender) runTransferLive<CudaRenderer>(bfm, sparseReg, live);
        else                runTransferLive<Renderer>(bfm, sparseReg, live);
#else
        runTransferLive<Renderer>(bfm, sparseReg, live);
#endif
    } else if (mode == "live-gpu") {
        runLiveGpu(bfm, sparseReg, live);
    } else if (mode == "verify-gpu") {
#ifdef USE_CUDA
        runGpuCheck(bfm, meanShape, albedo);
        return 0;
#else
        std::cerr << "verify-gpu: built without CUDA. Rebuild with: make USE_CUDA=1\n";
        return 1;
#endif
    } else if (mode == "dense") {
        fitDenseOnBiwi(bfm, meanShape, albedo, /*sparseInit=*/nullptr, icpIters);
    } else if (mode == "full") {
        // Cheap robust landmark fit pins the pose, then dense ICP recovers
        // identity from depth — linked through the RGB↔depth extrinsics.
        std::cout << "\n== Stage 1/2: sparse (landmarks, Biwi RGB) ==\n";
        const std::optional<FitParameters> sparse =
            fitSparseOnBiwi(bfm, meanShape, albedo);
        std::cout << "\n== Stage 2/2: dense (ICP, Biwi depth) ==\n";
        fitDenseOnBiwi(bfm, meanShape, albedo, sparse ? &*sparse : nullptr, icpIters);
    } else {
        std::cerr << "unknown --mode '" << mode << "'\n\n";
        printUsage();
        return 1;
    }

    renderMeanFace(bfm, meanShape, albedo);
    return 0;
}
