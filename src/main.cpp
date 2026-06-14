#include "BFMLoader.h"
#include "PandoraLoader.h"
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


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
    std::cout << "BFM: " << meanV.rows() << " vertices, " << bfm.faces().rows() << " triangles\n";
    writeObj("data/out/mean_face.obj", meanV, bfm.faces(), albedo);

    // tweak a few identity coefficients -> a different face
    Eigen::VectorXf alpha = Eigen::VectorXf::Zero(40);
    alpha(0) =  200.5f;
    alpha(1) = -200.5f;
    writeObj("data/out/tweaked_face.obj", bfm.shape(alpha), bfm.faces(), albedo);
    std::cout << "wrote data/out/{mean_face,tweaked_face}.obj\n";

    // ---- Pandora: load frames via the loader ----
    PandoraLoader loader(pandora, num_images, rgbd);
    std::vector<PandoraFrame> frames = loader.getFrames();
    if (frames.empty()) { std::cerr << "no Pandora frames found at " << pandora << "\n"; return 1; }
    std::cout << "Pandora: loaded " << frames.size() << " frame(s)\n";

    for (size_t i = 0; i < frames.size(); ++i) {
        const cv::Mat& rgb = frames[i].rgb;
        std::cout << "  frame " << i << ": RGB " << rgb.cols << "x" << rgb.rows
                  << " ch=" << rgb.channels();
        if (rgbd && !frames[i].depth.empty()) {
            double dmin, dmax;
            cv::minMaxLoc(frames[i].depth, &dmin, &dmax);
            std::cout << " | DEPTH " << frames[i].depth.cols << "x" << frames[i].depth.rows
                      << " range " << dmin << ".." << dmax << " mm";
        }
        std::cout << "\n";
    }
    return 0;
}
