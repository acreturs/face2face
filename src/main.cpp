#include "BFMLoader.h"
#include "BiwiLoader.h"
#include "IPhoneLoader.h"
#include "Renderer.h"
#include "ProjectionUtils.h"
#include "Lighting.h"
#include "CeresFitter.h"
#include "LandmarkDetector.h"

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

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
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Config — every path and tunable lives here, grouped by concern.
// ─────────────────────────────────────────────────────────────────────────────
namespace cfg {

// ── input paths ──
const std::string kBfmPath   = "data/bfm/model2017-1_bfm_nomouth.h5";
const std::string kIPhoneDir = "data/iphone/default";
std::string       kBiwiDir   = "data/biwi/01";   // set via --biwi-seq / --biwi-dir
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

// ── output layout ──  each run writes into data/out/<tag>/ (iphone, biwi_rgb,
// biwi_dense, debug) so datasets never clobber each other.
const std::string kOutDir = "data/out";
std::string outDir(const std::string& tag) {
    const std::string d = kOutDir + "/" + tag;
    std::filesystem::create_directories(d);
    return d;
}

// ── initial face depth (mm) ──  a plausible pose init; the fit refines it.
// iPhone selfies are at arm's length; Biwi seeds from the GT head centre.
constexpr float kIphoneDepthMM = 350.0f;

// ── sparse / contour landmark fit ──
constexpr double kDefaultSparseReg  = 100.0;  // interior-only fit (--sparse-reg)
constexpr int    kContourOuterIters = 40;
// Expression prior for the SINGLE-FRAME fit. Kept much stiffer than the identity
// reg: only 5 interior landmarks (2 mouth corners) drive expression here, which
// cannot reliably determine 30 coeffs — so a neutral face must stay neutral
// instead of over-articulating (jaw-open) to absorb landmark noise. Video
// tracking uses a lower value (identity frozen → expression must move).
constexpr double kExprRegWeight = 500.0;

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
constexpr double kDepthCropRadiusMM       = 100.0;
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
// straight-on). Mirrors the panels in the video output (fitBiwiVideo) so single
// -image runs (iPhone, Biwi sparse/RGB/full) get the same "just the mask" views.
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
    frontal.translation = Eigen::Vector3d(0, 0, kIphoneDepthMM);
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
    const std::string& tag = "iphone",  // filename prefix for the wireframe overlay
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

// Project the BFM mesh onto the first iPhone photo and save a wireframe overlay.
// No-op (with a message) if the session or its frames are missing.

// Full RGB analysis-by-synthesis fit on ONE image (no depth). Dataset-agnostic:
// the iPhone and Biwi-RGB entry points both funnel through here with their own
// image, intrinsics K, landmark file, output tag and initial face depth.
//   Stage 1  fitPose                 -> pose
//   Stage 2  fitPoseAndShapeContour  -> pose + identity + expression (+ jaw)
//   Stage 3  fitPhotometric          -> lighting + albedo + pose (geometry frozen)
static void fitRgbFrame(
    const BFMLoader&        bfm,
    const Eigen::MatrixX3f& meanShape,
    const Eigen::MatrixX3f& albedo,
    const cv::Mat&          photo,
    const Eigen::Matrix3f&  K,
    const std::string&      tag,            // output subfolder, e.g. "iphone"
    double                  initZ,          // initial face depth mm (iPhone ~350, Biwi ~880)
    double                  sparseReg,
    bool                    refinePhotometric,
    // Optional depth cloud (points already in THIS camera frame). When present,
    // Stage 2 becomes the FULL fit: landmarks + contour + depth solved jointly.
    const std::vector<Eigen::Vector3d>* depthCloud = nullptr)
{
    std::cout << "\n=== RGB fit [" << tag << "] : " << photo.cols << "x" << photo.rows
              << ", initZ=" << initZ << "mm ===\n";

    // Landmarks are detected IN-PROCESS (YuNet face box + pose-robust 5 points,
    // LBF jaw contour) — no Python round-trip / landmark file needed.
    LandmarkDetector detector(kLbfModelPath, kYuNetPath);
    if (!detector.ok()) {
        std::cerr << "RGB fit skipped: landmark detector unavailable "
                     "(is models/lbfmodel.yaml present?)\n";
        return;
    }
    const std::vector<LandmarkObservation> observations = detector.detect(photo);
    if (observations.empty()) {
        std::cerr << "RGB fit skipped: no face detected in the frame\n";
        return;
    }

    // Interior-only subset (drop the -1 contour points) — Stage 1 needs fixed
    // model vertices.
    std::vector<LandmarkObservation> interiorObs;
    std::copy_if(observations.begin(), observations.end(),
                 std::back_inserter(interiorObs),
                 [](const LandmarkObservation& o) { return o.vertexIndex >= 0; });

    PoseParameters initialPose;
    initialPose.translation = Eigen::Vector3d(0.0, 0.0, initZ);

    // z-bounds scaled around the initial depth so the same code works at iPhone
    // (~350 mm) and Biwi (~880 mm) distances.
    const double zMin = 0.4 * initZ, zMax = 2.5 * initZ;

    // Stage 1: pose only (interior points).
    const PoseParameters poseOnly = CeresFitter::fitPose(
        meanShape, interiorObs, K, initialPose, zMin, zMax);

    // Stage 2: pose + identity + expression, with the sliding jaw-contour term
    // if the landmark file carries -1 contour points.
    const bool hasContour = std::any_of(
        observations.begin(), observations.end(),
        [](const LandmarkObservation& o) { return o.vertexIndex < 0; });

    const FitParameters poseAndShape = hasContour
        ? CeresFitter::fitPoseAndShapeContour(
              meanShape, bfm.shape_basis_raw(), bfm.shape_sigma(),
              bfm.expr_basis_raw(), bfm.expr_sigma(), bfm.faces(),
              observations, K, poseOnly, sparseReg, /*exprReg=*/kExprRegWeight,
              zMin, zMax, kContourOuterIters,
              depthCloud, kDepthPointToPlaneWeight, kDepthWeight, kDepthVertexStride)
        : CeresFitter::fitPoseAndShape(
              meanShape, bfm.shape_basis_raw(), bfm.shape_sigma(),
              observations, K, poseOnly, sparseReg, zMin, zMax);
    std::cout << tag << " sparse: " << observations.size() << " landmarks ("
              << (hasContour ? "with contour+expr" : "interior only")
              << (depthCloud ? " + depth" : "") << "), reg=" << sparseReg << '\n';

    const Eigen::VectorXf alpha = poseAndShape.shapeCoefficients.cast<float>();
    const Eigen::MatrixX3f fittedShape = poseAndShape.exprCoefficients.size()
        ? bfm.shape(alpha, poseAndShape.exprCoefficients.cast<float>())
        : bfm.shape(alpha);

    saveCurrentModel(outDir(tag) + "/fitted_face.obj",
                     fittedShape, bfm.faces(), albedo);
    const Eigen::VectorXf disp = (fittedShape - meanShape).rowwise().norm();
    std::cout << "Mean shape displacement: " << disp.mean()
              << " mm, max " << disp.maxCoeff() << " mm\n";

    writeFitOutputs("pose_only", photo, bfm, meanShape, albedo, K, poseOnly,
                    tag, &observations);
    writeFitOutputs("pose_and_shape", photo, bfm, fittedShape, albedo, K,
                    poseAndShape.pose, tag, &observations);

    if (!refinePhotometric) return;

    // Stage 3: PHOTOMETRIC — estimate SH lighting + BFM albedo, refine pose;
    // geometry (identity + expression) frozen from Stage 2.
    const std::string progressDir = outDir(tag) + "/photo_progress";
    std::filesystem::create_directories(progressDir);
    const Renderer photoRenderer(photo.rows, photo.cols, bfm.faces());

    const auto albedoOf = [&](const FitParameters& fp) -> Eigen::MatrixX3f {
        if (fp.albedoCoefficients.size() == 0) return albedo;
        Eigen::VectorXf betaFull = Eigen::VectorXf::Zero(bfm.color_sigma().size());
        const int n = std::min<int>(fp.albedoCoefficients.size(), betaFull.size());
        betaFull.head(n) = fp.albedoCoefficients.head(n).cast<float>();
        return bfm.albedo(betaFull);
    };

    const DenseIterationCallback writeProgress =
        [&](int iteration, const FitParameters& current, double rmse) {
            const RenderInput in{
                .shape  = fittedShape,
                .albedo = albedoOf(current),
                .R      = current.pose.rotationMatrix(),
                .t      = current.pose.translation.cast<float>(),
                .K      = K,
                .sh     = current.sh,
            };
            cv::Mat prog = blendRenderOnPhoto(photoRenderer.render(in), photo);
            std::ostringstream label;
            label << "photo " << std::setw(2) << std::setfill('0') << iteration
                  << " | colRMSE " << std::fixed << std::setprecision(3) << rmse;
            cv::putText(prog, label.str(), {8, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        {0, 200, 0}, 2, cv::LINE_AA);
            std::ostringstream name;
            name << progressDir << "/photo_" << std::setw(2) << std::setfill('0')
                 << iteration << ".png";
            cv::imwrite(name.str(), prog);
        };

    FitParameters photoInit = poseAndShape;
    photoInit.shapeCoefficients = Eigen::VectorXd::Zero(kShapeCoefficientCount);

    const FitParameters photoFit = CeresFitter::fitPhotometric(
        fittedShape, bfm.shape_basis_raw(), bfm.shape_sigma(), bfm.faces(),
        albedo, bfm.color_basis_raw(), bfm.color_sigma(),
        photo, K, photoInit,
        /*shapeRegWeight=*/sparseReg,
        kAlbedoRegWeight,
        kPhotoIterations,
        kPhotoPixelStride,
        /*photometricWeight=*/1.0,
        /*optimizeShape=*/false,   // geometry (id+expr) fixed from Stage 2
        /*optimizeLighting=*/true,
        /*optimizeAlbedo=*/true,
        /*optimizePose=*/true,
        writeProgress);

    const Eigen::MatrixX3f photoAlbedo = albedoOf(photoFit);
    saveCurrentModel(outDir(tag) + "/fitted_face_photometric.obj",
                     fittedShape, bfm.faces(), photoAlbedo);

    const RenderInput finalIn{
        .shape  = fittedShape,
        .albedo = photoAlbedo,
        .R      = photoFit.pose.rotationMatrix(),
        .t      = photoFit.pose.translation.cast<float>(),
        .K      = K,
        .sh     = photoFit.sh,
    };
    overlayRenderOnPhoto(photoRenderer.render(finalIn), photo,
        outDir(tag) + "/render_photometric_appearance.png");
    writeFitOutputs("photometric", photo, bfm, fittedShape, photoAlbedo, K,
                    photoFit.pose, tag, &observations, photoFit.sh);
}

// iPhone entry point: load frame `frameIndex` + its landmarks, then fitRgbFrame.
static void overlayMeshOnPhoto(
    const BFMLoader& bfm,
    const Eigen::MatrixX3f& meanShape,
    const Eigen::MatrixX3f& albedo,
    double sparseReg = 100.0,
    bool refinePhotometric = false,
    int frameIndex = 0)
{
    try {
        IPhoneLoader iphone(kIPhoneDir, frameIndex + 1);
        const auto frames = iphone.getFrames();
        if (static_cast<int>(frames.size()) <= frameIndex) {
            std::cout << "iPhone: frame " << frameIndex << " not found in "
                      << kIPhoneDir << " (have " << frames.size() << ")\n";
            return;
        }
        fitRgbFrame(bfm, meanShape, albedo, frames[frameIndex].rgb, iphone.K(),
                    "iphone", kIphoneDepthMM, sparseReg, refinePhotometric);
    } catch (const std::exception& e) {
        std::cerr << "iPhone fit skipped: " << e.what() << '\n';
    }
}

// Biwi RGB entry point: the same analysis-by-synthesis pipeline on a Biwi RGB
// frame, using the rgb.cal intrinsics and the GT head depth as the init.
//   useDepth   = false → RGB-only fit (landmarks + contour + photometric).
//   useDepth   = true  → FULL fit: the Kinect depth cloud, transformed into the
//                        RGB camera via the extrinsics, joins Stage 2 as an ICP
//                        term, so geometry is anchored by metric depth.
//   frameIndex          → which frame to use, as an index into the SORTED list
//                        of frames found in kBiwiDir (0 = the first/smallest
//                        frame number present, e.g. frame_00003 for subject 01).
static void fitBiwiRgb(
    const BFMLoader& bfm,
    const Eigen::MatrixX3f& meanShape,
    const Eigen::MatrixX3f& albedo,
    double sparseReg,
    bool refinePhotometric,
    bool useDepth,
    int frameIndex = 0)
{
    try {
        BiwiLoader biwi(kBiwiDir, frameIndex + 1);
        const auto frames = biwi.getFrames();
        if (static_cast<int>(frames.size()) <= frameIndex) {
            std::cout << "Biwi: frame index " << frameIndex << " not found in "
                      << kBiwiDir << " (have " << frames.size() << ")\n";
            return;
        }
        const BiwiFrame& f = frames[frameIndex];
        const BiwiCalibration cal = biwi.getCalibration();
        const std::string tag = useDepth ? "biwi_full" : "biwi_rgb";
        std::cout << "Biwi " << tag << ": subject " << kBiwiDir
                  << ", frame " << f.frameNumber << " (index " << frameIndex << ")\n";

        // Head centre in the RGB camera frame → init depth + cloud crop centre.
        const Eigen::Vector3d headRgb = cal.R_rgb * f.headCenter + cal.t_rgb;

        // Build the depth cloud in the RGB camera frame (only when requested).
        std::vector<Eigen::Vector3d> headCloud;
        if (useDepth) {
            const std::vector<Eigen::Vector3d> cloudDepth = backprojectDepth(
                f.depth, cal.K_depth, kDepthBackprojStride);
            const std::vector<Eigen::Vector3d> headDepth = cropHead(
                cloudDepth, f.headCenter, kDepthCropRadiusMM, kDepthCropFrontSlabMM);
            headCloud.reserve(headDepth.size());
            for (const Eigen::Vector3d& p : headDepth)
                headCloud.push_back(cal.R_rgb * p + cal.t_rgb);  // depth → rgb frame
            std::cout << "Biwi depth: " << headCloud.size()
                      << " head points into the RGB fit\n";
        }

        fitRgbFrame(bfm, meanShape, albedo, f.rgb, cal.K_rgb, tag,
                    headRgb.z(), sparseReg, refinePhotometric,
                    useDepth ? &headCloud : nullptr);
    } catch (const std::exception& e) {
        std::cerr << "Biwi RGB fit skipped: " << e.what() << '\n';
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Offline VIDEO tracking (Biwi frame sequence)
// ─────────────────────────────────────────────────────────────────────────────
// Personalise identity + albedo on frame 0 (the expensive full fit), then TRACK
// only pose + expression (+ lighting) on the rest, warm-started from the previous
// frame with identity/albedo frozen. This is the split real-time reconstruction
// needs. Adaptable:
//   useDepth = true  → track via the depth cloud (robust to head rotation; only
//                      frame 0 needs landmarks).
//   useDepth = false → RGB-only: track via per-frame landmark files.
static void fitBiwiVideo(
    const BFMLoader& bfm,
    const Eigen::MatrixX3f& meanShape,
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
        // In-process landmark detection — no per-frame Python round-trip.
        LandmarkDetector detector(kLbfModelPath, kYuNetPath);
        if (!detector.ok()) {
            std::cerr << "video: landmark detector unavailable "
                         "(is models/lbfmodel.yaml present?)\n";
            return;
        }

        // Mean 2D position of the interior landmarks — used for gating.
        const auto centroid = [](const std::vector<LandmarkObservation>& obs) {
            Eigen::Vector2d c(0, 0); int n = 0;
            for (const LandmarkObservation& o : obs)
                if (o.vertexIndex >= 0) { c += o.imagePoint; ++n; }
            return n ? Eigen::Vector2d(c / n) : Eigen::Vector2d(-1, -1);
        };
        // Reject a detection that jumps far from the tracked head (a false
        // positive on the background). Tuned tight — Haar's false positives on
        // Biwi sit only ~85 px from the real face, while real inter-frame head
        // motion is smaller; ~12% of image width separates them.
        const double kGateDist = 0.12 * sz.width;
        Eigen::Vector2d prevCentroid(-1, -1);
        bool haveCentroid = false;

        // Personalised (constant) parameters, filled on frame 0.
        Eigen::VectorXd identity, betaVec;
        light::SHCoeffs prevSh = light::defaultWhite();
        // Per-frame history for constant-velocity motion PREDICTION: warm-start
        // each frame from where the head is *heading* (prev + (prev − prev2)),
        // not where it *was* — otherwise the tracker starts a frame behind and
        // lags during motion.
        PoseParameters  prevPose,  prev2Pose;
        Eigen::VectorXd prevExpr,  prev2Expr;
        bool            havePrev2 = false;

        for (size_t i = 0; i < frames.size(); ++i) {
            const BiwiFrame& f = frames[i];
            const Eigen::Vector3d headRgb = cal.R_rgb * f.headCenter + cal.t_rgb;
            const double zMin = 0.4 * headRgb.z(), zMax = 2.5 * headRgb.z();
            std::vector<Eigen::Vector3d> cloud;
            if (useDepth) cloud = headCloudRgb(f);
            const std::vector<Eigen::Vector3d>* cloudPtr = useDepth ? &cloud : nullptr;

            FitParameters geo;
            if (i == 0) {
                // ── PERSONALISE: full fit (identity + albedo + everything) ──
                const std::vector<LandmarkObservation> obs = detector.detect(f.rgb);
                if (obs.empty()) {
                    std::cerr << "video: no face detected on the first frame — "
                                 "cannot personalise\n";
                    return;
                }
                prevCentroid = centroid(obs); haveCentroid = true;  // seed the gate
                std::vector<LandmarkObservation> interior;
                std::copy_if(obs.begin(), obs.end(), std::back_inserter(interior),
                             [](const LandmarkObservation& o) { return o.vertexIndex >= 0; });
                PoseParameters init; init.translation = Eigen::Vector3d(0, 0, headRgb.z());
                const PoseParameters poseOnly = CeresFitter::fitPose(
                    meanShape, interior, cal.K_rgb, init, zMin, zMax);

                geo = CeresFitter::fitPoseAndShapeContour(
                    meanShape, bfm.shape_basis_raw(), bfm.shape_sigma(),
                    bfm.expr_basis_raw(), bfm.expr_sigma(), bfm.faces(),
                    obs, cal.K_rgb, poseOnly, sparseReg, sparseReg, zMin, zMax,
                    kContourOuterIters, cloudPtr, kDepthPointToPlaneWeight,
                    kDepthWeight, kDepthVertexStride);
                identity = geo.shapeCoefficients;

                const Eigen::MatrixX3f fitted = bfm.shape(
                    identity.cast<float>(), geo.exprCoefficients.cast<float>());
                FitParameters photoInit;
                photoInit.pose = geo.pose;
                photoInit.shapeCoefficients = Eigen::VectorXd::Zero(kShapeCoefficientCount);

                const FitParameters photo = CeresFitter::fitPhotometric(
                    fitted, bfm.shape_basis_raw(), bfm.shape_sigma(), bfm.faces(),
                    albedo, bfm.color_basis_raw(), bfm.color_sigma(),
                    f.rgb, cal.K_rgb, photoInit, sparseReg, kAlbedoRegWeight,
                    kPhotoIterations, kPhotoPixelStride, 1.0,
                    /*optimizeShape=*/true, /*optimizeLighting=*/true,
                    /*optimizeAlbedo=*/true, /*optimizePose=*/true);
                betaVec  = photo.albedoCoefficients;
                prevPose = geo.pose;       // depth pose = the tracking reference
                prevSh   = photo.sh;
                prevExpr = geo.exprCoefficients;
                std::cout << "[personalise] frame " << f.frameNumber
                          << " — identity + albedo fixed for the rest\n";
            } else {
                // ── TRACK: pose + expression only, identity/albedo frozen ──
                // RGB-only tracks from per-frame landmarks (detected in C++);
                // depth tracks from the cloud (no landmarks needed).
                std::vector<LandmarkObservation> obs;
                if (!useDepth) {
                    obs = detector.detect(f.rgb);
                    if (obs.empty()) {
                        std::cout << "  skip frame " << f.frameNumber
                                  << " (no face detected)\n";
                        continue;
                    }
                    // Gate: reject a detection that jumped far from the last one.
                    const Eigen::Vector2d c = centroid(obs);
                    if (haveCentroid && (c - prevCentroid).norm() > kGateDist) {
                        std::cout << "  reject frame " << f.frameNumber
                                  << " (detection jumped " << (c - prevCentroid).norm()
                                  << " px)\n";
                        continue;
                    }
                    prevCentroid = c; haveCentroid = true;
                }
                // Constant-velocity prediction of pose + expression → warm-start
                // where the head is heading, not where it was (kills motion lag).
                PoseParameters  initPose = prevPose;
                Eigen::VectorXd initExpr = prevExpr;
                if (havePrev2) {
                    initPose.angleAxis   = 2.0 * prevPose.angleAxis   - prev2Pose.angleAxis;
                    initPose.translation = 2.0 * prevPose.translation - prev2Pose.translation;
                    initExpr = (2.0 * prevExpr - prev2Expr).cwiseMax(-3.0).cwiseMin(3.0);
                }

                geo = CeresFitter::fitPoseAndShapeContour(
                    meanShape, bfm.shape_basis_raw(), bfm.shape_sigma(),
                    bfm.expr_basis_raw(), bfm.expr_sigma(), bfm.faces(),
                    obs, cal.K_rgb, initPose, sparseReg, sparseReg, zMin, zMax,
                    /*numOuterIterations=*/10, cloudPtr, kDepthPointToPlaneWeight,
                    kDepthWeight, kDepthVertexStride,
                    identity, initExpr, /*optimizeIdentity=*/false);

                const Eigen::MatrixX3f fitted = bfm.shape(
                    identity.cast<float>(), geo.exprCoefficients.cast<float>());
                FitParameters photoInit;
                photoInit.pose = geo.pose;
                photoInit.shapeCoefficients  = Eigen::VectorXd::Zero(kShapeCoefficientCount);
                photoInit.albedoCoefficients = betaVec;
                photoInit.sh = prevSh;
                // Lighting only — pose stays FIXED at the depth-fit result, which
                // is the authoritative tracking pose. Refining pose photometrically
                // here (unreliable on low-res Biwi) is what drifted the tracker.
                const FitParameters photo = CeresFitter::fitPhotometric(
                    fitted, bfm.shape_basis_raw(), bfm.shape_sigma(), bfm.faces(),
                    albedo, bfm.color_basis_raw(), bfm.color_sigma(),
                    f.rgb, cal.K_rgb, photoInit, sparseReg, kAlbedoRegWeight,
                    /*numIterations=*/2, kPhotoPixelStride, 1.0,
                    /*optimizeShape=*/false, /*optimizeLighting=*/true,
                    /*optimizeAlbedo=*/false, /*optimizePose=*/true);
                // Temporal smoothing (EMA) on pose + expression to damp jitter.
                PoseParameters smoothPose;
                smoothPose.angleAxis =
                    kSmoothAlpha * geo.pose.angleAxis + (1 - kSmoothAlpha) * prevPose.angleAxis;
                smoothPose.translation =
                    kSmoothAlpha * geo.pose.translation + (1 - kSmoothAlpha) * prevPose.translation;
                const Eigen::VectorXd smoothExpr =
                    kSmoothAlpha * geo.exprCoefficients + (1 - kSmoothAlpha) * prevExpr;

                prev2Pose = prevPose;    prev2Expr = prevExpr;   // shift history
                prevPose  = smoothPose;  prevExpr  = smoothExpr;
                prevSh    = photo.sh;
                havePrev2 = true;
            }

            // 3-panel composite: overlay | reconstruction @ tracked pose |
            // reconstruction frontal (both recon panels on a black background).
            const Eigen::MatrixX3f fitted = bfm.shape(
                identity.cast<float>(), prevExpr.cast<float>());
            const RenderInput in{
                .shape = fitted, .albedo = albedoOf(betaVec),
                .R = prevPose.rotationMatrix(), .t = prevPose.translation.cast<float>(),
                .K = cal.K_rgb, .sh = prevSh };
            cv::Mat overlay = blendRenderOnPhoto(renderer.render(in), f.rgb);
            label(overlay, i == 0 ? "personalise" : "track");

            cv::Mat reconPose = reconBgr(fitted, betaVec, prevPose, prevSh, cal.K_rgb);
            label(reconPose, "reconstruction @ pose");

            PoseParameters frontal;                     // identity rotation, centred
            frontal.translation = Eigen::Vector3d(0, 0, kIphoneDepthMM);
            cv::Mat reconFront = reconBgr(fitted, betaVec, frontal, prevSh, frontalK);
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
        .t      = Eigen::Vector3f(0.0f, 0.0f, kIphoneDepthMM),
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

int main(int argc, char** argv)
{
    // --mode sparse (landmarks) | dense (depth) | full (sparse → dense).
    // --dataset biwi | iphone  (iphone: sparse only — no depth).
    // --sparse-reg <λ>: shape regulariser of the iPhone sparse fit (default 100,
    // tuned for the 9-landmark set).
    // --biwi-seq <NN>: Biwi sequence (01..24, default 01). Generate landmarks
    // once: python3 python/gen_landmarks.py --set dense
    //   data/biwi/NN/frame_XXXXX_rgb.png data/biwi/NN/landmarks_XXXXX.txt
    // ── CLI ──
    std::string mode        = "sparse";  // sparse | dense | full | photometric
    std::string dataset     = "biwi";    // biwi | iphone
    double      sparseReg   = kDefaultSparseReg;   // --sparse-reg
    int         icpIters    = 30;        // --icp-iters: dense-fit outer rounds
    int         iphoneFrame = 0;         // --iphone-frame: which iPhone frame
    int         biwiFrame   = 0;         // --biwi-frame: which Biwi frame (index)
    int         numFrames   = 30;        // --frames: video-mode frame count
    bool        useDepth    = false;     // --depth: add the depth term
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--dataset" && i + 1 < argc) dataset = argv[++i];
        else if (arg == "--sparse-reg" && i + 1 < argc) sparseReg = std::stod(argv[++i]);
        else if (arg == "--biwi-seq" && i + 1 < argc) {
            std::string seq = argv[++i];
            if (seq.size() == 1) seq = "0" + seq;
            kBiwiDir = "data/biwi/" + seq;
        }
        else if (arg == "--biwi-dir" && i + 1 < argc) kBiwiDir = argv[++i];  // e.g. data/BK-1/01
        else if (arg == "--icp-iters" && i + 1 < argc) icpIters = std::stoi(argv[++i]);
        else if (arg == "--iphone-frame" && i + 1 < argc) iphoneFrame = std::stoi(argv[++i]);
        else if (arg == "--biwi-frame" && i + 1 < argc) biwiFrame = std::stoi(argv[++i]);
        else if (arg == "--frames" && i + 1 < argc) numFrames = std::stoi(argv[++i]);
        else if (arg == "--depth") useDepth = true;
    }

    std::filesystem::create_directories(kOutDir);

    // Load the Basel Face Model and take the mean identity face.
    BFMLoader bfm(kBfmPath);
    bfm.summariseBFM(kBfmPath);

    const Eigen::MatrixX3f meanShape = bfm.mean_shape();
    const Eigen::MatrixX3f albedo    = bfm.albedo();   // mean albedo (β = 0)
    std::cout << "BFM: " << meanShape.rows() << " vertices, "
              << bfm.faces().rows() << " triangles\n";
    std::cout << "Mode: " << mode << "  Dataset: " << dataset
              << (useDepth ? "  (+depth)" : "") << '\n';

    saveCurrentModel(outDir("debug") + "/current_face.obj", meanShape,
                     bfm.faces(), albedo);

    if (mode == "video") {
        // Offline video: personalise on frame 0, then track the sequence.
        fitBiwiVideo(bfm, meanShape, albedo, numFrames, sparseReg, useDepth);
    } else if (mode == "dense") {
        fitDenseOnBiwi(bfm, meanShape, albedo,           // Biwi depth → data/out/biwi_dense/
                       nullptr, icpIters);
    } else if (mode == "full") {
        // Two-stage pipeline: first the cheap, robust landmark fit (pins the
        // pose), then the dense ICP fit (recovers identity from depth) — linked
        // through the real RGB↔depth extrinsics.
        std::cout << "\n== Stage 1/2: sparse (landmarks, Biwi RGB) ==\n";
        const std::optional<FitParameters> sparse =
            fitSparseOnBiwi(bfm, meanShape, albedo);
        if (!sparse)
            std::cout << "full: Stage 1 skipped — dense starts at identity rotation\n";

        std::cout << "\n== Stage 2/2: dense (ICP, Biwi depth) ==\n";
        fitDenseOnBiwi(bfm, meanShape, albedo,
                       sparse ? &*sparse : nullptr, icpIters);
    } else if (mode == "photometric") {
        // Analysis-by-synthesis: landmark+contour+expr init → photometric.
        // On Biwi, --depth adds the metric depth term (the FULL fit).
        if (dataset == "biwi")
            fitBiwiRgb(bfm, meanShape, albedo, sparseReg,
                       /*refinePhotometric=*/true, useDepth, biwiFrame);
        else
            overlayMeshOnPhoto(bfm, meanShape, albedo, sparseReg,
                               /*refinePhotometric=*/true, iphoneFrame);
    } else if (dataset == "biwi") {
        fitSparseOnBiwi(bfm, meanShape, albedo);         // Biwi RGB landmarks → fitted_face_biwi_sparse.obj
    } else {
        overlayMeshOnPhoto(bfm, meanShape, albedo, sparseReg,   // iPhone RGB landmarks → fitted_face.obj
                           /*refinePhotometric=*/false, iphoneFrame);
    }

    renderMeanFace(bfm, meanShape, albedo);

    return 0;
}
