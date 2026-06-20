#include "BFMLoader.h"
#include "PandoraLoader.h"
#include "IPhoneLoader.h"
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// BFM is +Y-up / -Z-forward, OpenCV is +Y-down / +Z-forward. This static
// alignment composes with any head pose R to put the model in the camera frame.
static const Eigen::Matrix3f BFM_TO_CAM = (Eigen::Matrix3f() <<
    1,  0,  0,
    0, -1,  0,
    0,  0, -1).finished();

// Project an (N×3) BFM mesh onto the image with pose (R, t) and intrinsics K.
// Returns the (N×2) pixel coordinates; vertices with z ≤ 0 get (-1,-1).
static Eigen::Matrix<float, Eigen::Dynamic, 2>
projectBFM(const Eigen::MatrixX3f& V,
           const Eigen::Matrix3f&  R,
           const Eigen::Vector3f&  t,
           const Eigen::Matrix3f&  K)
{
    const Eigen::Matrix3f R_total = R * BFM_TO_CAM;
    Eigen::MatrixX3f V_cam = (V * R_total.transpose()).rowwise() + t.transpose();

    Eigen::Matrix<float, Eigen::Dynamic, 2> uv(V_cam.rows(), 2);
    for (int i = 0; i < V_cam.rows(); ++i) {
        const float z = V_cam(i, 2);
        if (z <= 1e-3f) { uv.row(i) << -1.f, -1.f; continue; }
        uv(i, 0) = K(0, 0) * V_cam(i, 0) / z + K(0, 2);
        uv(i, 1) = K(1, 1) * V_cam(i, 1) / z + K(1, 2);
    }
    return uv;
}

// Draw the projected vertices on top of the input image as small coloured dots.
static cv::Mat overlayProjection(const cv::Mat& bgr,
                                 const Eigen::Matrix<float, Eigen::Dynamic, 2>& uv,
                                 const Eigen::MatrixX3f& albedo,
                                 int stride = 5)
{
    cv::Mat out = bgr.clone();
    const int H = out.rows, W = out.cols;
    for (int i = 0; i < uv.rows(); i += stride) {
        const int u = static_cast<int>(uv(i, 0) + 0.5f);
        const int v = static_cast<int>(uv(i, 1) + 0.5f);
        if (u < 0 || v < 0 || u >= W || v >= H) continue;
        // OpenCV is BGR; albedo is RGB in [0,1]
        const cv::Vec3b c(
            static_cast<uchar>(albedo(i, 2) * 255.0f),
            static_cast<uchar>(albedo(i, 1) * 255.0f),
            static_cast<uchar>(albedo(i, 0) * 255.0f));
        cv::circle(out, {u, v}, 1, c, cv::FILLED);
    }
    return out;
}

// Draw triangle edges of the BFM mesh on top of the input image (wireframe).
// Each edge is coloured by the average albedo of its two endpoints. Triangles
// with any vertex behind the camera (uv == -1) are skipped.
static cv::Mat overlayWireframe(const cv::Mat& bgr,
                                const Eigen::Matrix<float, Eigen::Dynamic, 2>& uv,
                                const Eigen::MatrixX3i& triangles,
                                const Eigen::MatrixX3f& albedo,
                                int stride = 1)
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

    for (int t = 0; t < triangles.rows(); t += stride) {
        const int i0 = triangles(t, 0);
        const int i1 = triangles(t, 1);
        const int i2 = triangles(t, 2);

        const cv::Point p0(static_cast<int>(uv(i0, 0) + 0.5f),
                           static_cast<int>(uv(i0, 1) + 0.5f));
        const cv::Point p1(static_cast<int>(uv(i1, 0) + 0.5f),
                           static_cast<int>(uv(i1, 1) + 0.5f));
        const cv::Point p2(static_cast<int>(uv(i2, 0) + 0.5f),
                           static_cast<int>(uv(i2, 1) + 0.5f));
        if (uv(i0, 0) < 0 || uv(i1, 0) < 0 || uv(i2, 0) < 0) continue;
        if (!inImage(p0) && !inImage(p1) && !inImage(p2))     continue;

        const cv::Vec3b c0 = vertColour(i0);
        const cv::Vec3b c1 = vertColour(i1);
        const cv::Vec3b c2 = vertColour(i2);

        // Average colour for each of the three edges
        auto mix = [](cv::Vec3b a, cv::Vec3b b) {
            return cv::Vec3b((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2);
        };
        cv::line(out, p0, p1, mix(c0, c1), 1, cv::LINE_AA);
        cv::line(out, p1, p2, mix(c1, c2), 1, cv::LINE_AA);
        cv::line(out, p2, p0, mix(c2, c0), 1, cv::LINE_AA);
    }
    return out;
}


//schreibt in objekt was in meshview angeguckt werden kann
static void writeObj(const std::string& path,
                     const Eigen::MatrixX3f& V, const Eigen::MatrixX3i& F,
                     const Eigen::MatrixX3f& C)
{
    std::ofstream out(path);
    for (int i = 0; i < V.rows(); ++i)
        out << "v " << V(i, 0) << ' ' << V(i, 1) << ' ' << V(i, 2)
            << ' ' << C(i, 0) << ' ' << C(i, 1) << ' ' << C(i, 2) << '\n';
    for (int i = 0; i < F.rows(); ++i)             // OBJ indices are 1-based -> +1
        out << "f " << F(i, 0) + 1 << ' ' << F(i, 1) + 1 << ' ' << F(i, 2) + 1 << '\n';
}

int main(int argc, char** argv)
{
    // ---- runtime options (set via the Makefile: make run NUM_IMAGES=20 RGBD=false) ----
    int  num_images = 1;       // how many Pandora frames to process
    bool rgbd       = true;    // true: RGB + depth   |   false: RGB only
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--images" && i + 1 < argc) num_images = std::stoi(argv[++i]);
        else if (a == "--rgbd"   && i + 1 < argc) rgbd       = (std::string(argv[++i]) == "true");
    }
    std::cout << "options: num_images=" << num_images
              << ", rgbd=" << (rgbd ? "true" : "false") << "\n";

    // Paths relative to the repo root.
    const std::string model   = "data/bfm/model2017-1_bfm_nomouth.h5";
    const std::string pandora = "data/pandora/base_1_ID01";
    std::filesystem::create_directories("data/out");

    // ---- BFM: load + generate faces ----
    BFMLoader bfm(model);
    bfm.summariseBFM(model);                         // full overview of the .h5 contents
    Eigen::MatrixX3f meanV  = bfm.mean_shape();
    Eigen::MatrixX3f albedo = bfm.albedo();          // mean per-vertex colour (no beta yet)
    // std::cout << "BFM: " << meanV.rows() << " vertices, " << bfm.faces().rows() << " triangles\n";
    // writeObj("data/out/mean_face.obj", meanV, bfm.faces(), albedo);

    // // tweak a few identity coefficients -> a different face
    // Eigen::VectorXf alpha = Eigen::VectorXf::Zero(40);
    // alpha(0) =  200.5f;
    // alpha(1) = -200.5f;
    // writeObj("data/out/tweaked_face.obj", bfm.shape(alpha), bfm.faces(), albedo);
    // std::cout << "wrote data/out/{mean_face,tweaked_face}.obj\n";

    // // ---- Pandora: load frames via the loader ----
    // PandoraLoader loader(pandora, num_images, rgbd);
    // std::vector<PandoraFrame> frames = loader.getFrames();
    // if (frames.empty()) { std::cerr << "no Pandora frames found at " << pandora << "\n"; return 1; }
    // std::cout << "Pandora: loaded " << frames.size() << " frame(s)\n";

    // for (size_t i = 0; i < frames.size(); ++i) {
    //     const cv::Mat& rgb = frames[i].rgb;
    //     std::cout << "  frame " << i << ": RGB " << rgb.cols << "x" << rgb.rows
    //               << " ch=" << rgb.channels();
    //     if (rgbd && !frames[i].depth.empty()) {
    //         double dmin, dmax;
    //         cv::minMaxLoc(frames[i].depth, &dmin, &dmax);
    //         std::cout << " | DEPTH " << frames[i].depth.cols << "x" << frames[i].depth.rows
    //                   << " range " << dmin << ".." << dmax << " mm";
    //     }
    //     std::cout << "\n";
    // }

    // ---- iPhone session: project the BFM onto the first photo ---------------
    const std::string iphoneSession = "data/iphone/default";
    try {
        IPhoneLoader iphone(iphoneSession, 1);
        auto iframes = iphone.getFrames();
        if (iframes.empty()) {
            std::cout << "iPhone: no frames in " << iphoneSession << "\n";
        } else {
            const cv::Mat& photo = iframes[0].rgb;

            // Default pose: identity rotation, place face ~500 mm in front of
            // the camera. The BFM mean is roughly centred at the origin so a
            // translation of (0, 0, 500 mm) puts it dead-centre at the image
            // principal point.
            const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
            const Eigen::Vector3f t(0.f, 0.f, 350.f);

            auto uv = projectBFM(meanV, R, t, iphone.K());
            cv::Mat overlay = overlayWireframe(photo, uv, bfm.faces(), albedo,
                                               /*stride=*/2);
            const std::string outPath = "data/out/iphone_overlay.png";
            cv::imwrite(outPath, overlay);
            std::cout << "iPhone: drew " << bfm.faces().rows() / 2
                      << " wireframe triangles onto " << photo.cols << "x" << photo.rows
                      << " → " << outPath << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "iPhone projection skipped: " << e.what() << "\n";
    }
    return 0;
}
