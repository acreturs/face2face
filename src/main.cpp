#include "BFMLoader.h"
#include "IPhoneLoader.h"
#include "Renderer.h"
#include "ProjectionUtils.h"
#include "Lighting.h"
#include "CeresFitter.h"

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────
namespace {

const std::string kBfmPath      = "data/bfm/model2017-1_bfm_nomouth.h5";
const std::string kIPhoneDir    = "data/iphone/default";
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
static void overlayRenderOnPhoto(const RenderOutput& r,
                                 const cv::Mat&      photo,
                                 const std::string&  path)
{
    // Rendered image: float RGB [0,1] → 8-bit BGR to match the photo.
    cv::Mat bgr;
    cv::cvtColor(r.image, bgr, cv::COLOR_RGB2BGR);
    bgr.convertTo(bgr, CV_8UC3, 255.0f);

    // Blend 50/50 everywhere, then restore the photo where nothing was drawn.
    cv::Mat overlay;
    cv::addWeighted(photo, 0.5, bgr, 0.5, 0.0, overlay);
    photo.copyTo(overlay, ~r.mask);

    cv::imwrite(path, overlay);
    std::cout << "iPhone: render overlay → " << path
              << "  (" << overlay.cols << "x" << overlay.rows << ")\n";
}

static void writeFitOutputs(
    const std::string& suffix,
    const cv::Mat& photo,
    const BFMLoader& bfm,
    const Eigen::MatrixX3f& shape,
    const Eigen::MatrixX3f& albedo,
    const Eigen::Matrix3f& intrinsics,
    const PoseParameters& pose
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

    const cv::Mat wireframe =
        overlayWireframe(
            photo,
            projectedVertices,
            bfm.faces(),
            albedo,
            2
        );

    const std::string wireframePath =
        kOutDir + "/iphone_overlay_" + suffix + ".png";

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
        kOutDir + "/render_overlay_" + suffix + ".png";

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
    const Eigen::MatrixX3f& albedo
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
        const FitParameters poseAndShape =
            CeresFitter::fitPoseAndShape(
                meanShape,
                bfm.shape_basis_raw(),
                bfm.shape_sigma(),
                observations,
                iphone.K(),
                poseOnly,
                10.0
            );

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
            poseOnly
        );

        // Output 2: fitted shape with pose-and-shape optimization.
        writeFitOutputs(
            "pose_and_shape",
            photo,
            bfm,
            fittedShape,
            albedo,
            iphone.K(),
            poseAndShape.pose
        );
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

int main()
{
    std::filesystem::create_directories(kOutDir);

    // Load the Basel Face Model and take the mean identity face.
    BFMLoader bfm(kBfmPath);
    bfm.summariseBFM(kBfmPath);

    const Eigen::MatrixX3f meanShape = bfm.mean_shape();
    const Eigen::MatrixX3f albedo    = bfm.albedo();   // mean albedo (β = 0)
    std::cout << "BFM: " << meanShape.rows() << " vertices, "
              << bfm.faces().rows() << " triangles\n";

    saveCurrentModel(kOutDir + "/current_face.obj", meanShape, bfm.faces(), albedo);
    overlayMeshOnPhoto(bfm, meanShape, albedo);
    renderMeanFace(bfm, meanShape, albedo);

    return 0;
}
