#include "BFMLoader.h"
#include "BiwiLoader.h"
#include "Renderer.h"
#include "ProjectionUtils.h"
#include "Lighting.h"
#include "CeresFitter.h"
#include "LandmarkDetector.h"
#include "FaceTracker.h"
#include "ScopedTimer.h"

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>   // imshow/waitKey for --mode live

#include <algorithm>
#include <cmath>
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

// ─────────────────────────────────────────────────────────────────────────────
// Config — every path and tunable lives here, grouped by concern.
// ─────────────────────────────────────────────────────────────────────────────
namespace cfg {

// ── input paths ──  the only supported dataset is Biwi (Kinect RGB-D).
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

// ── landmark detector backend (--detector) ──
//   yunet     : YuNet CNN face box + 5 pose-robust points, LBF jaw contour (default)
//   lbf       : Haar face box + LBF 68-point landmarks only (the legacy detector)
//   mediapipe : read per-frame `landmarks_mp_XXXXX.txt` files written by the
//               Python pre-pass (python/gen_landmarks_mediapipe.py) — MediaPipe
//               is Bazel-built and cannot link into this binary, so it runs
//               offline; its 468-pt mesh adds vertical mouth points + a denser,
//               more pose-robust jaw than LBF.
std::string kDetector = "yunet";

// ── output layout ──  each run writes into data/out/<tag>/ (biwi_video_rgb,
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

// ── sparse / contour landmark fit ──
constexpr double kDefaultSparseReg  = 100.0;  // interior-only fit (--sparse-reg)
constexpr int    kContourOuterIters = 40;
// Expression prior for the SINGLE-FRAME fit. Kept much stiffer than the identity
// reg: only 5 interior landmarks (2 mouth corners) drive expression here, which
// cannot reliably determine 30 coeffs — so a neutral face must stay neutral
// instead of over-articulating (jaw-open) to absorb landmark noise. Video
// tracking uses a lower value (identity frozen → expression must move).
constexpr double kExprRegWeight = 500.0;
// Expression prior for video TRACKING (identity frozen, so expression must stay
// mobile to follow the mouth) — looser than the single-frame value above, but
// far stiffer than the identity reg so a neutral frame stays neutral.
constexpr double kTrackExprRegWeight = 120.0;

// ── photometric (appearance) fit ──
constexpr double kAlbedoRegWeight  = 50.0;
constexpr int    kPhotoIterations  = 20;
constexpr int    kPhotoPixelStride = 1;

// ── video temporal smoothing ──  EMA on the per-frame pose + expression to
// damp jitter: new = α·fit + (1−α)·previous. 1 = no smoothing, lower = smoother
// (but laggier). Paired with velocity prediction so it stays responsive.
constexpr double kSmoothAlpha = 0.6;

// ── depth term (Biwi "full" fit) ──
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

// ─────────────────────────────────────────────────────────────────────────────
// Mesh serialisation
// ─────────────────────────────────────────────────────────────────────────────

// Save the current BFM state as a .obj the viewer (and any mesh tool) can show:
//   v  x y z r g b        position + per-vertex albedo colour
//   vn nx ny nz           per-vertex unit normal (area-weighted)
//   f  v//vn ...          faces with matching normal indices (1-based)
static void saveCurrentModel(const std::string&      path,
                             const Eigen::MatrixX3f&  V,
                             const Eigen::MatrixX3i&  F,
                             const Eigen::MatrixX3f&  albedo)
{
    const Eigen::MatrixX3f N = Renderer::computeNormals(V, F);

    std::ofstream out(path);
    out << "# face2face current model — written by saveCurrentModel()\n";
    out << "# " << V.rows() << " vertices, " << F.rows()
        << " triangles, per-vertex normals + albedo colours\n";

    for (int i = 0; i < V.rows(); ++i)
        out << "v " << V(i, 0) << ' ' << V(i, 1) << ' ' << V(i, 2) << ' '
            << albedo(i, 0) << ' ' << albedo(i, 1) << ' ' << albedo(i, 2) << '\n';

    for (int i = 0; i < N.rows(); ++i)
        out << "vn " << N(i, 0) << ' ' << N(i, 1) << ' ' << N(i, 2) << '\n';

    for (int i = 0; i < F.rows(); ++i) {
        const int a = F(i, 0) + 1, b = F(i, 1) + 1, c = F(i, 2) + 1;
        out << "f " << a << "//" << a << ' '
                    << b << "//" << b << ' '
                    << c << "//" << c << '\n';
    }

    std::cout << "saved current model → " << path
              << "  (" << V.rows() << " v, " << F.rows() << " f, with normals)\n";
}

// Save a raw point cloud as a .obj (vertices only, no faces). Lets you open it
// next to a mesh in MeshLab to see the fit sitting on the measured data.
static void savePointCloud(const std::string&                  path,
                           const std::vector<Eigen::Vector3d>& points)
{
    std::ofstream out(path);
    out << "# point cloud (camera frame, mm) — " << points.size() << " points\n";
    for (const Eigen::Vector3d& p : points)
        out << "v " << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    std::cout << "saved point cloud → " << path
              << "  (" << points.size() << " points)\n";
}

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

// ─────────────────────────────────────────────────────────────────────────────
// Debug overlays
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline stages
// ─────────────────────────────────────────────────────────────────────────────

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

// 3-panel mask-visualisation strip: overlay | reconstruction @ the fitted pose
// (face alone on black) | reconstruction frontal (identity+expression,
// straight-on). Mirrors the per-frame panels the video path emits, so the
// single-frame geometry modes (dense/full/sparse) get the same "just the mask"
// views.
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
        const RenderInput in{ .shape = shape, .albedo = albedo,
            .R = p.rotationMatrix(), .t = p.translation.cast<float>(),
            .K = K, .sh = sh };
        cv::Mat bgr;
        cv::cvtColor(renderer.render(in).image, bgr, cv::COLOR_RGB2BGR);
        bgr.convertTo(bgr, CV_8UC3, 255.0);
        return bgr;
    };
    const auto label = [](cv::Mat& img, const std::string& s) {
        cv::putText(img, s, {8, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    {0, 220, 0}, 2, cv::LINE_AA);
    };

    const RenderInput overlayIn{ .shape = shape, .albedo = albedo,
        .R = pose.rotationMatrix(), .t = pose.translation.cast<float>(),
        .K = intrinsics, .sh = sh };
    cv::Mat overlay = blendRenderOnPhoto(renderer.render(overlayIn), photo);
    label(overlay, "overlay");

    cv::Mat reconPose = reconBgr(pose, intrinsics);
    label(reconPose, "reconstruction @ pose");

    PoseParameters frontal;                                // identity rotation, centred
    frontal.translation = Eigen::Vector3d(0, 0, kFrontalRenderDepthMM);
    cv::Mat reconFront =
        reconBgr(frontal, proj::defaultIntrinsics(photo.cols, photo.rows));
    label(reconFront, "reconstruction frontal");

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

// ─────────────────────────────────────────────────────────────────────────────
// ENTRY POINT: rgb / rgbd video reconstruction on a Biwi sequence
// ─────────────────────────────────────────────────────────────────────────────
// Personalise identity + albedo on frame 0 (the expensive full fit), then TRACK
// only pose + expression (+ lighting) on the rest, warm-started from the
// previous frame with identity/albedo frozen (see FaceTracker). Dispatched from:
//   --mode rgb   → useDepth = false (landmarks + jaw contour + photometric)
//   --mode rgbd  → useDepth = true  (+ the metric Kinect depth ICP term)
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
            const RenderInput in{ .shape = shape, .albedo = albedoOf(beta),
                .R = pose.rotationMatrix(), .t = pose.translation.cast<float>(),
                .K = K, .sh = sh };
            cv::Mat bgr;
            cv::cvtColor(renderer.render(in).image, bgr, cv::COLOR_RGB2BGR);
            bgr.convertTo(bgr, CV_8UC3, 255.0);
            return bgr;
        };
        const auto label = [](cv::Mat& img, const std::string& s) {
            cv::putText(img, s, {8, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        {0, 220, 0}, 2, cv::LINE_AA);
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
        FaceTracker::Config tc;
        tc.sparseReg              = sparseReg;
        tc.exprRegPersonalise     = kExprRegWeight;
        tc.exprRegTrack           = kTrackExprRegWeight;
        tc.albedoRegWeight        = kAlbedoRegWeight;
        tc.smoothAlpha            = kSmoothAlpha;
        tc.contourItersPersonalise = kContourOuterIters;
        tc.photoIterations        = kPhotoIterations;
        tc.photoPixelStride       = kPhotoPixelStride;
        tc.depthPointToPlaneWeight = kDepthPointToPlaneWeight;
        tc.depthWeight            = kDepthWeight;
        tc.depthVertexStride      = kDepthVertexStride;
        FaceTracker tracker(bfm, cal.K_rgb, tc);

        for (size_t i = 0; i < frames.size(); ++i) {
            const BiwiFrame& f = frames[i];
            const Eigen::Vector3d headRgb = cal.R_rgb * f.headCenter + cal.t_rgb;
            std::vector<Eigen::Vector3d> cloud;
            if (useDepth) cloud = headCloudRgb(f);
            const std::vector<Eigen::Vector3d>* cloudPtr = useDepth ? &cloud : nullptr;

            if (!tracker.personalised()) {
                const std::vector<LandmarkObservation> obs = frameLandmarks(f);
                ScopedTimer t("personalise");
                if (!tracker.personalise(f.rgb, obs, headRgb.z(), cloudPtr)) {
                    std::cerr << "video: no landmarks on the first frame — "
                                 "cannot personalise\n";
                    return;
                }
                std::cout << "[personalise] frame " << f.frameNumber
                          << " — identity + albedo fixed for the rest\n";
            } else {
                // RGB-only tracks from per-frame landmarks; depth tracks from
                // the cloud (no landmarks needed → empty observations).
                std::vector<LandmarkObservation> obs;
                if (!useDepth) {
                    obs = frameLandmarks(f);
                    if (obs.empty()) {
                        std::cout << "  skip frame " << f.frameNumber
                                  << " (no landmarks)\n";
                        continue;
                    }
                }
                ScopedTimer t("track");
                if (!tracker.track(f.rgb, obs, headRgb.z(), cloudPtr))
                    continue;   // gated / unusable frame (tracker logged why)
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
            label(overlay, i == 0 ? "personalise" : "track");

            cv::Mat reconPose = reconBgr(fitted, betaVec, curPose, curSh, cal.K_rgb);
            label(reconPose, "reconstruction @ pose");

            PoseParameters frontal;                     // identity rotation, centred
            frontal.translation = Eigen::Vector3d(0, 0, kFrontalRenderDepthMM);
            cv::Mat reconFront = reconBgr(fitted, betaVec, frontal, curSh, frontalK);
            label(reconFront, "reconstruction frontal");

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
    } catch (const std::exception& e) {
        std::cerr << "Biwi video skipped: " << e.what() << '\n';
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Realtime camera mode (--mode live) — HOST ONLY (Docker has no camera access)
// ─────────────────────────────────────────────────────────────────────────────
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
};

// ─────────────────────────────────────────────────────────────────────────────
// ENTRY POINT: live-cpu — realtime reconstruction from the Mac camera (CPU)
// ─────────────────────────────────────────────────────────────────────────────
static void runLiveCpu(const BFMLoader& bfm, double sparseReg, const LiveOptions& lo)
{
    cv::VideoCapture cap;
    cv::Mat raw;

    if (!lo.source.empty()) {
        // An image path/pattern (e.g. frame_00003_rgb.png) is an image SEQUENCE
        // — force CAP_IMAGES, else FFmpeg opens it as a one-frame video.
        const bool imageSeq = lo.source.find(".png") != std::string::npos ||
                              lo.source.find(".jpg") != std::string::npos ||
                              lo.source.find('%')   != std::string::npos;
        cap.open(lo.source, imageSeq ? cv::CAP_IMAGES : cv::CAP_ANY);
        if (!cap.isOpened() || !cap.read(raw) || raw.empty()) {
            std::cerr << "live: could not read " << lo.source << '\n';
            return;
        }
    } else {
        // macOS/AVFoundation gotchas this handles:
        //  - frames stay BLACK if the stream opened before the permission
        //    dialog was answered → re-opening after warm-up fixes it;
        //  - device 0 can be an inactive iPhone Continuity Camera (delivers
        //    black forever) → fall through to the next indices;
        //  - forcing CAP_PROP_FRAME_WIDTH/HEIGHT can also blank the stream, so
        //    we take the camera's native size (prep() downscales anyway).
        const auto openWorkingCamera = [&](int idx) -> bool {
            cap.release();
            if (!cap.open(idx)) return false;
            cv::Mat probe;
            for (int attempt = 0; attempt < 50; ++attempt) {   // ~2.5 s warm-up
                if (cap.read(probe) && !probe.empty()) {
                    const cv::Scalar m = cv::mean(probe);
                    if (m[0] + m[1] + m[2] > 6.0) { raw = probe; return true; }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            std::cout << "live: camera " << idx
                      << " opened but only delivers black frames — skipping\n";
            return false;
        };

        bool opened = openWorkingCamera(lo.camera);
        // Auto-fallback over the next device indices (Continuity Camera is
        // often index 0 while the built-in FaceTime camera is 1).
        for (int idx = 0; !opened && idx <= 2; ++idx)
            if (idx != lo.camera) opened = openWorkingCamera(idx);
        // Final retry of the requested device with a FRESH open — a stream that
        // was opened before the permission grant stays black until re-opened.
        if (!opened) opened = openWorkingCamera(lo.camera);
        if (!opened) {
            std::cerr << "live: no camera delivered usable frames.\n"
                         "  - run on the HOST (Docker has no camera)\n"
                         "  - System Settings → Privacy & Security → Camera → "
                         "enable your terminal, then RE-RUN\n"
                         "  - or pick a device explicitly: --camera 1\n";
            return;
        }
        std::cout << "live: camera delivering " << raw.cols << "x" << raw.rows
                  << " (" << cap.getBackendName() << ")\n";
    }
    const double scale = std::min(1.0, static_cast<double>(lo.width) / raw.cols);
    const auto prep = [&](const cv::Mat& in) {
        cv::Mat out;
        if (scale < 1.0) cv::resize(in, out, {}, scale, scale, cv::INTER_AREA);
        else             out = in;
        return out;
    };
    cv::Mat frame = prep(raw);
    const int W = frame.cols, H = frame.rows;

    // Unknown webcam intrinsics: start from a 60°-HFOV guess; personalisation
    // refines the focal (unless --no-optimize-focal).
    const Eigen::Matrix3f K0 = proj::defaultIntrinsics(W, H);
    std::cout << "live: " << W << "x" << H << " @ focal guess " << K0(0, 0)
              << " px" << (lo.optimizeFocal ? " (optimised during personalise)" : "")
              << "\n      keys: q quit | p re-personalise | s snapshot\n";

    if (kDetector == "mediapipe")
        std::cout << "live: --detector mediapipe is offline-only — using YuNet\n";
    LandmarkDetector detector(kLbfModelPath,
                              kDetector == "lbf" ? "" : kYuNetPath);
    if (!detector.ok()) {
        std::cerr << "live: landmark detector unavailable (models/ missing?)\n";
        return;
    }

    FaceTracker::Config tc;                    // live-tuned (speed over polish)
    tc.sparseReg              = sparseReg;
    tc.exprRegPersonalise     = kExprRegWeight;
    tc.exprRegTrack           = kTrackExprRegWeight;
    tc.albedoRegWeight        = kAlbedoRegWeight;
    tc.smoothAlpha            = kSmoothAlpha;
    tc.contourItersPersonalise = kContourOuterIters;
    tc.photoIterations        = 20;            // one-off → keep startup snappy
    tc.photoPixelStride       = 2;
    tc.contourItersTrack      = 3;             // warm-started → converges fast
    tc.trackPhotoIterations   = 1;
    tc.trackPhotoOptimizePose = false;         // lighting = pure linear estimate
    tc.lightingEvery          = 10;
    tc.photoRefine            = lo.photoRefine;
    tc.optimizeFocal          = lo.optimizeFocal;
    FaceTracker tracker(bfm, K0, tc);

    const Renderer renderer(H, W, bfm.faces());
    const std::string liveDir = outDir("live");
    double fpsEma = 0.0;
    long processed = 0, written = 0;

    for (;;) {
        if (!cap.read(raw) || raw.empty()) break;
        const auto t0 = std::chrono::steady_clock::now();
        frame = prep(raw);

        std::vector<LandmarkObservation> obs;
        { ScopedTimer t("live/detect"); obs = detector.detect(frame); }

        bool tracked = false;
        if (!tracker.personalised()) {
            if (!obs.empty()) {
                ScopedTimer t("live/personalise");
                tracked = tracker.personalise(frame, obs, lo.initZ);
            }
        } else {
            ScopedTimer t("live/track");
            tracked = tracker.track(frame, obs, lo.initZ);
        }

        cv::Mat vis;
        if (tracker.personalised()) {
            ScopedTimer t("live/render");
            const RenderInput in{
                .shape  = tracker.currentShape(),
                .albedo = tracker.currentAlbedo(),
                .R      = tracker.pose().rotationMatrix(),
                .t      = tracker.pose().translation.cast<float>(),
                .K      = tracker.K(),
                .sh     = tracker.sh() };
            vis = blendRenderOnPhoto(renderer.render(in), frame);
        } else {
            vis = frame.clone();
        }

        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        fpsEma = fpsEma <= 0.0 ? 1000.0 / ms : 0.9 * fpsEma + 0.1 * (1000.0 / ms);
        std::ostringstream hud;
        hud << (tracker.personalised() ? (tracked ? "track" : "hold")
                                       : "looking for a face...")
            << "  " << std::fixed << std::setprecision(1) << fpsEma << " fps"
            << "  f=" << static_cast<int>(tracker.K()(0, 0)) << "px";
        cv::putText(vis, hud.str(), {8, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    {0, 220, 0}, 2, cv::LINE_AA);

        ++processed;
        if (lo.maxFrames > 0) {                 // headless test: dump frames
            std::ostringstream n;
            n << liveDir << "/live_" << std::setw(4) << std::setfill('0')
              << written++ << ".png";
            cv::imwrite(n.str(), vis);
        }
        if (lo.display) {
            cv::imshow("face2face live", vis);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) break;
            if (key == 'p') {                   // re-personalise from scratch
                tracker.reset(K0);
                std::cout << "live: re-personalising…\n";
            }
            if (key == 's')
                cv::imwrite(liveDir + "/snapshot.png", vis);
        }
        if (lo.maxFrames > 0 && processed >= lo.maxFrames) break;
    }
    std::cout << "live: processed " << processed << " frames @ "
              << std::fixed << std::setprecision(1) << fpsEma << " fps";
    if (written > 0) std::cout << ", wrote " << written << " → " << liveDir;
    std::cout << '\n';
}

// ─────────────────────────────────────────────────────────────────────────────
// ENTRY POINT: live-gpu — realtime reconstruction on the GPU (colleagues' WIP)
// ─────────────────────────────────────────────────────────────────────────────
// Stub. The intended design (see PLAN_REALTIME.md §4): a GPU rasteriser + an
// analytic-Jacobian photometric solve, driven by the SAME FaceTracker
// personalise/track split as runLiveCpu(). Wire the GPU renderer + solver in
// here and reuse the camera/HUD loop from runLiveCpu().
static void runLiveGpu(const BFMLoader& bfm, double sparseReg, const LiveOptions& lo)
{
    (void)bfm; (void)sparseReg; (void)lo;
    std::cerr << "live-gpu: not implemented yet — under development by the GPU "
                 "team. Use --mode live-cpu for the CPU tracker.\n";
}

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

// ─────────────────────────────────────────────────────────────────────────────
// Biwi: sparse + dense in the SAME camera system
// ─────────────────────────────────────────────────────────────────────────────

// Sparse landmark fit on the Biwi RGB frame, using the real rgb.cal intrinsics.
// The pose is metric and transfers into the depth camera via the extrinsics
// (see fitDenseOnBiwi). Landmarks per frame (once, in the container):
//   python3 python/gen_landmarks.py --set dense
//           data/biwi/01/frame_00003_rgb.png data/biwi/01/landmarks_00003.txt
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

        saveCurrentModel(outDir("biwi_sparse") + "/fitted_face.obj",
                         fittedShape, bfm.faces(), albedo);

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

// Dense ICP fit on the Biwi depth frame:
//   1. head centre = ground truth from *_pose.txt.
//   2. sparseInit is transformed into the depth camera via the extrinsics
//      (only possible with registered cameras).
//   3. the result is also rendered into the RGB camera and blended over the
//      photo — the "face copy on the image".
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

        // Scale the regulariser WITH the point count: E_data grows linearly
        // with the number of target points, E_reg does not. Biwi yields ~8700
        // head points (vs ~600 before); a constant reg would rail ~10 coeffs.
        // Base 150 (not 500): 500·N/600 flattens the shape back to ~1.4 mm from
        // the mean; 150·N/600 gives ~5 mm real personalisation without railing.
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

        saveCurrentModel(outDir("biwi_dense") + "/fitted_face.obj",
                         fitted, bfm.faces(), albedo);

        const Eigen::Matrix3f R = dense.pose.rotationMatrix();
        const Eigen::Vector3f t = dense.pose.translation.cast<float>();
        const Eigen::MatrixX3f posedCam = proj::toCameraFrame(fitted, R, t);
        saveCurrentModel(outDir("biwi_dense") + "/fitted_face_camframe.obj",
                         posedCam, bfm.faces(), albedo);
        savePointCloud(outDir("biwi_dense") + "/head_cloud.obj", head);

        // ─── projective texture: sample the real photo colours onto the mesh ─
        // Geometry alone is not recognisable — the mean albedo always looks like
        // "some face". With the extrinsics we project every camera-facing vertex
        // into the RGB photo and sample its colour there; averted/occluded
        // vertices keep the mean albedo.
        Eigen::MatrixX3f photoAlbedo = albedo;
        {
            const Eigen::MatrixX3f camNormals =
                Renderer::computeNormals(posedCam, bfm.faces());
            int sampled = 0;
            for (int i = 0; i < posedCam.rows(); ++i) {
                const Eigen::Vector3d pDepth = posedCam.row(i).cast<double>();
                const Eigen::Vector3d pRgb = cal.R_rgb * pDepth + cal.t_rgb;
                if (pRgb.z() <= 1.0) continue;
                // visible ≈ normal faces the camera (the face is frontally
                // convex, so no real occlusion test is needed here).
                const Eigen::Vector3d n = camNormals.row(i).cast<double>();
                if (n.dot(pDepth.normalized()) > -0.25) continue;
                const float u = static_cast<float>(
                    cal.K_rgb(0, 0) * pRgb.x() / pRgb.z() + cal.K_rgb(0, 2));
                const float v = static_cast<float>(
                    cal.K_rgb(1, 1) * pRgb.y() / pRgb.z() + cal.K_rgb(1, 2));
                if (u < 1.0f || v < 1.0f || u >= frame.rgb.cols - 2.0f ||
                    v >= frame.rgb.rows - 2.0f)
                    continue;
                cv::Mat patch;                          // 1x1 patch = bilinear
                cv::getRectSubPix(frame.rgb, {1, 1}, {u, v}, patch);
                const cv::Vec3b bgr = patch.at<cv::Vec3b>(0, 0);
                photoAlbedo.row(i) = Eigen::RowVector3f(
                    bgr[2] / 255.0f, bgr[1] / 255.0f, bgr[0] / 255.0f);
                ++sampled;
            }
            std::cout << "Projective texture: " << sampled << " / "
                      << posedCam.rows() << " vertices sampled from the photo\n";
        }
        saveCurrentModel(outDir("biwi_dense") + "/fitted_face_textured.obj",
                         fitted, bfm.faces(), photoAlbedo);

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

        // The same copy with the photo texture — the most convincing view:
        // when geometry AND colour match, the (green) boundary is nearly
        // invisible, i.e. the reconstruction is right.
        rgbInput.albedo = photoAlbedo;
        const RenderOutput texturedRender = rgbRenderer.render(rgbInput);
        overlayRenderOnPhoto(texturedRender, frame.rgb,
                             outDir("biwi_dense") + "/rgb_overlay_textured.png");

        // 100% variant: NO blend, NO contour — inside the render mask the
        // render fully replaces the photo, so everything there is purely the
        // reconstructed mesh with its texture.
        {
            cv::Mat renderBgr;
            cv::cvtColor(texturedRender.image, renderBgr, cv::COLOR_RGB2BGR);
            renderBgr.convertTo(renderBgr, CV_8UC3, 255.0f);
            cv::Mat hard = frame.rgb.clone();
            renderBgr.copyTo(hard, texturedRender.mask);
            cv::imwrite(outDir("biwi_dense") + "/rgb_overlay_textured_100.png", hard);
            std::cout << "render overlay (100%) → "
                      << outDir("biwi_dense") + "/rgb_overlay_textured_100.png" << '\n';
        }

        // 3-panel mask visualisation (overlay | recon @ pose | recon frontal),
        // in the RGB camera, with the photo-projected texture.
        {
            const Eigen::AngleAxisd rgbAA(Rrgb.cast<double>());
            PoseParameters rgbPose;
            rgbPose.angleAxis   = rgbAA.angle() * rgbAA.axis();
            rgbPose.translation = trgb.cast<double>();
            const cv::Mat maskPanels = renderMaskPanels(
                frame.rgb, fitted, photoAlbedo, cal.K_rgb, rgbPose, bfm.faces());
            const std::string maskPath = outDir("biwi_dense") + "/mask_panels.png";
            cv::imwrite(maskPath, maskPanels);
            std::cout << "Wrote mask panels: " << maskPath << '\n';
        }

        // Frontal portrait of the fitted identity: once with the mean albedo
        // (shows pure GEOMETRY) and once with the projected photo texture
        // (shows the PERSON — colour comes from the real image).
        {
            constexpr int RH = 480, RW = 640;
            RenderInput portrait{
                .shape  = fitted,
                .albedo = albedo,
                .R      = Eigen::Matrix3f::Identity(),
                .t      = Eigen::Vector3f(0.0f, 0.0f, 350.0f),
                .K      = proj::defaultIntrinsics(RW, RH),
                .sh     = light::defaultWhite(),
            };
            const Renderer portraitRenderer(RH, RW, bfm.faces());
            writeColourImage(portraitRenderer.render(portrait),
                             outDir("biwi_dense") + "/face_render.png");
            portrait.albedo = photoAlbedo;
            writeColourImage(portraitRenderer.render(portrait),
                             outDir("biwi_dense") + "/face_render_textured.png");
        }

        const Eigen::VectorXf disp = (fitted - meanShape).rowwise().norm();
        std::cout << "Biwi dense mean displacement: " << disp.mean()
                  << " mm, max " << disp.maxCoeff() << " mm\n";
    }
    catch (const std::exception& exception) {
        std::cerr << "Biwi dense fit skipped: " << exception.what() << '\n';
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI usage (Biwi is the only dataset).
// ─────────────────────────────────────────────────────────────────────────────
static void printUsage()
{
    std::cerr <<
      "Usage: face_recon --mode <MODE> [options]   (dataset: Biwi RGB-D only)\n\n"
      "Video reconstruction (personalise frame 0, then track the sequence):\n"
      "  rgb        landmarks + jaw contour + photometric      (no depth term)\n"
      "  rgbd       + the metric Kinect depth ICP term\n"
      "Realtime (Mac camera, HOST only — Docker has no camera):\n"
      "  live-cpu   the CPU tracker (this build)\n"
      "  live-gpu   GPU tracker — stub, under development\n"
      "Single-frame geometry (kept for reports):\n"
      "  dense      depth-only ICP\n"
      "  full       sparse landmark fit → dense ICP\n\n"
      "Options:\n"
      "  --biwi-dir <path> | --biwi-seq <NN>   Biwi subject folder\n"
      "  --frames <n>       frame count for rgb/rgbd (default 30)\n"
      "  --sparse-reg <λ>   identity/expression reg (default 100)\n"
      "  --icp-iters <n>    dense/full ICP rounds (default 30)\n"
      "  --detector <yunet|lbf|mediapipe>   landmark backend (default yunet)\n"
      "  --camera <i> | --live-source <path> | --live-width <px>\n"
      "  --live-frames <n> --live-nodisplay   headless live test\n"
      "  --photo-refine     pyramid photometric pose refinement (live)\n"
      "  --optimize-focal   solve fx=fy during personalise (experimental)\n"
      "  --timers           print per-stage timings\n";
}

int main(int argc, char** argv)
{
    // ── CLI ──
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
        else if (arg == "--detector" && i + 1 < argc) kDetector = argv[++i];  // yunet|lbf|mediapipe
        // ── live / realtime flags ──
        else if (arg == "--camera" && i + 1 < argc) live.camera = std::stoi(argv[++i]);
        else if (arg == "--live-source" && i + 1 < argc) live.source = argv[++i];
        else if (arg == "--live-width" && i + 1 < argc) live.width = std::stoi(argv[++i]);
        else if (arg == "--live-frames" && i + 1 < argc) live.maxFrames = std::stoi(argv[++i]);
        else if (arg == "--live-nodisplay") live.display = false;
        else if (arg == "--photo-refine") live.photoRefine = true;
        else if (arg == "--optimize-focal") live.optimizeFocal = true;
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

    // ── mode dispatch: one clear entry point per mode ──
    if (mode == "rgb" || mode == "rgbd") {
        runVideoReconstruction(bfm, albedo, numFrames, sparseReg,
                               /*useDepth=*/mode == "rgbd");
    } else if (mode == "live-cpu") {
        runLiveCpu(bfm, sparseReg, live);
    } else if (mode == "live-gpu") {
        runLiveGpu(bfm, sparseReg, live);
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
