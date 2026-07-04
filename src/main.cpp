#include "BFMLoader.h"
#include "BiwiLoader.h"
#include "IPhoneLoader.h"
#include "Renderer.h"
#include "ProjectionUtils.h"
#include "Lighting.h"
#include "CeresFitter.h"

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────
namespace {

const std::string kBfmPath      = "data/bfm/model2017-1_bfm_nomouth.h5";
const std::string kIPhoneDir    = "data/iphone/default";
std::string       kBiwiDir      = "data/biwi/01";   // selectable via --biwi-seq
const std::string kOutDir       = "data/out";

// Default debug pose: identity rotation, face this many mm in front of the
// camera. Closer = Bigger in Image, Further = Smaller in Image. We want this
// to be a reasonable initialisation close to my big head in the Selfie!! (Leon)
constexpr float kFaceDepthMM = 350.0f;

Eigen::Vector3f defaultTranslation() { return {0.0f, 0.0f, kFaceDepthMM}; }

}  // namespace

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
    const std::vector<LandmarkObservation>* observations = nullptr
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
        for (const LandmarkObservation& obs : *observations) {
            const cv::Point detected(
                static_cast<int>(obs.imagePoint.x() + 0.5),
                static_cast<int>(obs.imagePoint.y() + 0.5));
            const cv::Point model(
                static_cast<int>(projectedVertices(obs.vertexIndex, 0) + 0.5f),
                static_cast<int>(projectedVertices(obs.vertexIndex, 1) + 0.5f));
            cv::circle(wireframe, detected, radius, {0, 255, 0}, -1, cv::LINE_AA);
            cv::drawMarker(wireframe, model, {0, 0, 255}, cv::MARKER_CROSS,
                           2 * radius, std::max(2, radius / 2), cv::LINE_AA);
            const double du = projectedVertices(obs.vertexIndex, 0) - obs.imagePoint.x();
            const double dv = projectedVertices(obs.vertexIndex, 1) - obs.imagePoint.y();
            sumSquaredError += du * du + dv * dv;
        }
        const double rms =
            std::sqrt(sumSquaredError / static_cast<double>(observations->size()));
        std::cout << tag << ' ' << suffix << ": landmark reprojection RMS "
                  << rms << " px (gruen = detektiert, rot = Modell)\n";
    }

    const std::string wireframePath =
        kOutDir + "/" + tag + "_overlay_" + suffix + ".png";

    cv::imwrite(wireframePath, wireframe);

    std::cout << "Wrote wireframe overlay: "
              << wireframePath << '\n';

    RenderInput renderInput{
        .shape = shape,
        .albedo = albedo,
        .R = rotation,
        .t = translation,
        .K = intrinsics,
        .sh = light::defaultWhite(),
    };

    const RenderOutput renderOutput =
        Renderer(
            photo.rows,
            photo.cols,
            bfm.faces()
        ).render(renderInput);

    const std::string renderPath =
        kOutDir + "/render_overlay_" + tag + "_" + suffix + ".png";

    overlayRenderOnPhoto(
        renderOutput,
        photo,
        renderPath
    );
}

// Project the BFM mesh onto the first iPhone photo and save a wireframe overlay.
// No-op (with a message) if the session or its frames are missing.

static void overlayMeshOnPhoto(
    const BFMLoader& bfm,
    const Eigen::MatrixX3f& meanShape,
    const Eigen::MatrixX3f& albedo,
    // Shape regulariser λ of the sparse fit (--sparse-reg). Default 100 suits
    // the 9-landmark set (0.9 mm displacement, plausible); the 25-point set on a
    // high-res photo needs a much higher λ or it depth-overfits — see the study
    // in report_figures/leon_landmark_reg_study/.
    double sparseReg = 100.0,
    // When true, refine the sparse result with the dense PHOTOMETRIC term
    // (RGB-only, no depth) and write an extra "photometric" overlay + mesh.
    bool refinePhotometric = false
)
{
    try {
        IPhoneLoader iphone(kIPhoneDir, 1);

        const auto frames = iphone.getFrames();

        if (frames.empty()) {
            std::cout
                << "iPhone: no frames in "
                << kIPhoneDir
                << '\n';

            return;
        }

        const cv::Mat& photo = frames[0].rgb;

        const std::string landmarkPath =
            kIPhoneDir + "/landmarks_000000.txt";

        const std::vector<LandmarkObservation> observations =
            loadLandmarkObservations(landmarkPath);

        PoseParameters initialPose;
        initialPose.translation =
            defaultTranslation().cast<double>();

        // Stage 1: optimize only rotation and translation.
        const PoseParameters poseOnly =
            CeresFitter::fitPose(
                meanShape,
                observations,
                iphone.K(),
                initialPose
            );

        // Stage 2: refine pose and optimize shape coefficients.
        // Strong regulariser on purpose (100): only 9 landmarks (18 residuals)
        // for 6 pose + 30 shape params, and 2D landmarks say nothing about depth.
        // A weak reg improves the 2D reprojection but distorts the 3D geometry at
        // the unobserved vertices; reg=100 keeps the shape near the mean (<1 mm).
        // Real identity comes from the DENSE term, not from 9 landmarks.
        const FitParameters poseAndShape =
            CeresFitter::fitPoseAndShape(
                meanShape,
                bfm.shape_basis_raw(),
                bfm.shape_sigma(),
                observations,
                iphone.K(),
                poseOnly,
                sparseReg
            );
        std::cout << "iPhone sparse: " << observations.size()
                  << " Landmarks, reg=" << sparseReg << '\n';

        const Eigen::VectorXf alpha =
            poseAndShape.shapeCoefficients.cast<float>();

        const Eigen::MatrixX3f fittedShape =
            bfm.shape(alpha);

        // Save the personalized 3D mesh.
        saveCurrentModel(
            kOutDir + "/fitted_face.obj",
            fittedShape,
            bfm.faces(),
            albedo
        );

        // Print how much the optimized shape differs from the mean shape.
        const Eigen::VectorXf vertexDisplacement =
            (fittedShape - meanShape).rowwise().norm();

        std::cout
            << "Mean shape displacement: "
            << vertexDisplacement.mean()
            << " mm\n";

        std::cout
            << "Maximum shape displacement: "
            << vertexDisplacement.maxCoeff()
            << " mm\n";

        // Output 1: mean shape with pose-only optimization.
        writeFitOutputs(
            "pose_only",
            photo,
            bfm,
            meanShape,
            albedo,
            iphone.K(),
            poseOnly,
            "iphone",
            &observations
        );

        // Output 2: fitted shape with pose-and-shape optimization.
        writeFitOutputs(
            "pose_and_shape",
            photo,
            bfm,
            fittedShape,
            albedo,
            iphone.K(),
            poseAndShape.pose,
            "iphone",
            &observations
        );

        // Output 3 (optional): dense PHOTOMETRIC refinement — RGB only, no depth.
        // Alternates estimating SH lighting + BFM albedo (both linear) with a
        // per-pixel geometry refinement, so the render matches the selfie's
        // colour AND lighting. Progress overlays → data/out/iphone_photo_progress/.
        if (refinePhotometric) {
            const std::string progressDir = kOutDir + "/iphone_photo_progress";
            std::filesystem::create_directories(progressDir);
            const Renderer photoRenderer(photo.rows, photo.cols, bfm.faces());

            // Rebuild the per-vertex albedo from a fit's estimated β coeffs
            // (empty ⇒ fall back to the mean albedo).
            const auto albedoOf = [&](const FitParameters& fp) -> Eigen::MatrixX3f {
                if (fp.albedoCoefficients.size() == 0) return albedo;
                Eigen::VectorXf betaFull =
                    Eigen::VectorXf::Zero(bfm.color_sigma().size());
                const int n = std::min<int>(fp.albedoCoefficients.size(),
                                            betaFull.size());
                betaFull.head(n) = fp.albedoCoefficients.head(n).cast<float>();
                return bfm.albedo(betaFull);
            };

            const DenseIterationCallback writeProgress =
                [&](int iteration, const FitParameters& current, double rmse) {
                    const RenderInput in{
                        .shape  = bfm.shape(current.shapeCoefficients.cast<float>()),
                        .albedo = albedoOf(current),        // estimated albedo
                        .R      = current.pose.rotationMatrix(),
                        .t      = current.pose.translation.cast<float>(),
                        .K      = iphone.K(),
                        .sh     = current.sh,               // estimated lighting
                    };
                    cv::Mat prog = blendRenderOnPhoto(photoRenderer.render(in), photo);
                    std::ostringstream label;
                    label << "photo " << std::setw(2) << std::setfill('0')
                          << iteration << " | colRMSE " << std::fixed
                          << std::setprecision(3) << rmse;
                    cv::putText(prog, label.str(), {8, 28},
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 200, 0}, 2,
                                cv::LINE_AA);
                    std::ostringstream name;
                    name << progressDir << "/photo_" << std::setw(2)
                         << std::setfill('0') << iteration << ".png";
                    cv::imwrite(name.str(), prog);
                };

            const FitParameters photoFit = CeresFitter::fitPhotometric(
                meanShape, bfm.shape_basis_raw(), bfm.shape_sigma(), bfm.faces(),
                albedo, bfm.color_basis_raw(), bfm.color_sigma(),
                photo, iphone.K(), poseAndShape,
                /*shapeRegWeight=*/sparseReg,
                /*albedoRegWeight=*/100.0,
                /*numIterations=*/10,
                /*pixelStride=*/1,
                /*photometricWeight=*/1.0,
                /*optimizeShape=*/true,
                /*optimizeLighting=*/true,
                /*optimizeAlbedo=*/true,
                writeProgress);

            const Eigen::MatrixX3f photoShape =
                bfm.shape(photoFit.shapeCoefficients.cast<float>());
            const Eigen::MatrixX3f photoAlbedo = albedoOf(photoFit);
            saveCurrentModel(kOutDir + "/fitted_face_photometric.obj",
                             photoShape, bfm.faces(), photoAlbedo);

            // Final overlay rendered with the ESTIMATED lighting + albedo.
            const RenderInput finalIn{
                .shape  = photoShape,
                .albedo = photoAlbedo,
                .R      = photoFit.pose.rotationMatrix(),
                .t      = photoFit.pose.translation.cast<float>(),
                .K      = iphone.K(),
                .sh     = photoFit.sh,
            };
            overlayRenderOnPhoto(
                photoRenderer.render(finalIn), photo,
                kOutDir + "/render_overlay_iphone_photometric_appearance.png");

            writeFitOutputs("photometric", photo, bfm, photoShape, photoAlbedo,
                            iphone.K(), photoFit.pose, "iphone", &observations);
        }
    }
    catch (const std::exception& exception) {
        std::cerr
            << "iPhone projection skipped: "
            << exception.what()
            << '\n';
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
        .t      = defaultTranslation(),
        .K      = proj::defaultIntrinsics(RW, RH),
        .sh     = light::defaultWhite(),
    };

    const RenderOutput r = Renderer(RH, RW, bfm.faces()).render(input);

    const int covered = cv::countNonZero(r.mask);
    std::cout << "render: " << covered << " / " << (RW * RH) << " pixels covered\n";

    writeDepthVis  (r, kOutDir + "/render_depth.png");
    writeColourImage(r, kOutDir + "/render_image.png");
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

        saveCurrentModel(kOutDir + "/fitted_face_biwi_sparse.obj",
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
                             kOutDir + "/biwi_sparse_face_render.png");
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
        const std::string progressDir = kOutDir + "/biwi_icp_progress";
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

        saveCurrentModel(kOutDir + "/fitted_face_biwi_dense.obj",
                         fitted, bfm.faces(), albedo);

        const Eigen::Matrix3f R = dense.pose.rotationMatrix();
        const Eigen::Vector3f t = dense.pose.translation.cast<float>();
        const Eigen::MatrixX3f posedCam = proj::toCameraFrame(fitted, R, t);
        saveCurrentModel(kOutDir + "/fitted_face_biwi_dense_camframe.obj",
                         posedCam, bfm.faces(), albedo);
        savePointCloud(kOutDir + "/biwi_head_cloud.obj", head);

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
        saveCurrentModel(kOutDir + "/fitted_face_biwi_textured.obj",
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
                             kOutDir + "/biwi_depth_overlay.png");

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
                             kOutDir + "/biwi_rgb_overlay_dense.png");

        // The same copy with the photo texture — the most convincing view:
        // when geometry AND colour match, the (green) boundary is nearly
        // invisible, i.e. the reconstruction is right.
        rgbInput.albedo = photoAlbedo;
        const RenderOutput texturedRender = rgbRenderer.render(rgbInput);
        overlayRenderOnPhoto(texturedRender, frame.rgb,
                             kOutDir + "/biwi_rgb_overlay_textured.png");

        // 100% variant: NO blend, NO contour — inside the render mask the
        // render fully replaces the photo, so everything there is purely the
        // reconstructed mesh with its texture.
        {
            cv::Mat renderBgr;
            cv::cvtColor(texturedRender.image, renderBgr, cv::COLOR_RGB2BGR);
            renderBgr.convertTo(renderBgr, CV_8UC3, 255.0f);
            cv::Mat hard = frame.rgb.clone();
            renderBgr.copyTo(hard, texturedRender.mask);
            cv::imwrite(kOutDir + "/biwi_rgb_overlay_textured_100.png", hard);
            std::cout << "render overlay (100%) → "
                      << kOutDir + "/biwi_rgb_overlay_textured_100.png" << '\n';
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
                             kOutDir + "/biwi_face_render.png");
            portrait.albedo = photoAlbedo;
            writeColourImage(portraitRenderer.render(portrait),
                             kOutDir + "/biwi_face_render_textured.png");
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
    std::string mode = "sparse";
    std::string dataset = "biwi";
    double sparseReg = 100.0;
    int icpIters = 30;          // --icp-iters: outer ICP rounds of the dense fit
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
        else if (arg == "--icp-iters" && i + 1 < argc) icpIters = std::stoi(argv[++i]);
    }

    std::filesystem::create_directories(kOutDir);

    // Load the Basel Face Model and take the mean identity face.
    BFMLoader bfm(kBfmPath);
    bfm.summariseBFM(kBfmPath);

    const Eigen::MatrixX3f meanShape = bfm.mean_shape();
    const Eigen::MatrixX3f albedo    = bfm.albedo();   // mean albedo (β = 0)
    std::cout << "BFM: " << meanShape.rows() << " vertices, "
              << bfm.faces().rows() << " triangles\n";
    std::cout << "Mode: " << mode << "  Dataset: " << dataset << '\n';

    saveCurrentModel(kOutDir + "/current_face.obj", meanShape, bfm.faces(), albedo);

    if (mode == "dense") {
        fitDenseOnBiwi(bfm, meanShape, albedo,           // Biwi depth → fitted_face_biwi_dense.obj
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
        // RGB-only analysis-by-synthesis: sparse landmark init → photometric
        // refinement. No depth needed, so it runs on the iPhone selfie.
        overlayMeshOnPhoto(bfm, meanShape, albedo, sparseReg,
                           /*refinePhotometric=*/true);
    } else if (dataset == "biwi") {
        fitSparseOnBiwi(bfm, meanShape, albedo);         // Biwi RGB landmarks → fitted_face_biwi_sparse.obj
    } else {
        overlayMeshOnPhoto(bfm, meanShape, albedo, sparseReg);  // iPhone RGB landmarks → fitted_face.obj
    }

    renderMeanFace(bfm, meanShape, albedo);

    return 0;
}
