// loads biwi kinect frames (rgb + depth) and the camera calibration
// just mirrors the biwi.py file
#include "BiwiLoader.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

BiwiLoader::BiwiLoader(std::string seqPath, int numImages)
    : seqPath_(std::move(seqPath)), numImages_(numImages) {}

// frame number -> zero-padded stem, e.g. 3 -> "frame_00003"
static std::string frameStem(int i)
{
    std::ostringstream s;
    s << "frame_" << std::setw(5) << std::setfill('0') << i;
    return s.str();
}

// read all whitespace separated numbers from a text file (cal or pose)
static std::vector<double> readNumbers(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("BiwiLoader: cannot read " + path);
    std::vector<double> values;
    double v;
    while (in >> v) values.push_back(v);
    return values;
}

// .cal layout is 9 K, 4 distortion (all 0), 9 R, 3 t in mm, 2 width/height
static void parseCal(const std::string& path,
                     Eigen::Matrix3f&   K,
                     Eigen::Matrix3d&   R,
                     Eigen::Vector3d&   t)
{
    const std::vector<double> v = readNumbers(path);
    if (v.size() < 27)
        throw std::runtime_error("BiwiLoader: bad .cal format: " + path);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            K(r, c) = static_cast<float>(v[3 * r + c]);
            R(r, c) = v[13 + 3 * r + c];
        }
    t = Eigen::Vector3d(v[22], v[23], v[24]);
}

// decode the run length encoded depth .bin into a uint16 mm image
cv::Mat BiwiLoader::readDepthBin(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("BiwiLoader: cannot read depth " + path);

    int32_t width = 0, height = 0;
    in.read(reinterpret_cast<char*>(&width), 4);
    in.read(reinterpret_cast<char*>(&height), 4);
    if (width <= 0 || height <= 0)
        throw std::runtime_error("BiwiLoader: bad depth header: " + path);

    cv::Mat depth = cv::Mat::zeros(height, width, CV_16U);
    uint16_t* p = depth.ptr<uint16_t>();
    const int total = width * height;

    int filled = 0;
    while (filled < total && in) {
        int32_t numEmpty = 0, numFull = 0;
        in.read(reinterpret_cast<char*>(&numEmpty), 4);
        in.read(reinterpret_cast<char*>(&numFull), 4);
        if (!in) break;
        filled += numEmpty;                       // zeros are already there
        if (numFull > 0) {
            if (filled + numFull > total)
                throw std::runtime_error("BiwiLoader: RLE overflow: " + path);
            in.read(reinterpret_cast<char*>(p + filled),
                    static_cast<std::streamsize>(numFull) * 2);
            filled += numFull;
        }
    }
    return depth;
}

std::vector<BiwiFrame> BiwiLoader::getFrames() const
{
    // scan the directory, frames start at 3 and may have gaps
    std::vector<int> frameNumbers;
    const std::regex rgbName("frame_(\\d+)_rgb\\.png");
    for (const fs::directory_entry& entry : fs::directory_iterator(seqPath_)) {
        std::smatch match;
        const std::string name = entry.path().filename().string();
        if (std::regex_match(name, match, rgbName))
            frameNumbers.push_back(std::stoi(match[1]));
    }
    std::sort(frameNumbers.begin(), frameNumbers.end());

    std::vector<BiwiFrame> frames;
    for (int number : frameNumbers) {
        if (static_cast<int>(frames.size()) >= numImages_) break;
        const std::string stem = seqPath_ + "/" + frameStem(number);

        BiwiFrame f;
        f.frameNumber = number;
        f.rgb   = cv::imread(stem + "_rgb.png", cv::IMREAD_COLOR);
        f.depth = readDepthBin(stem + "_depth.bin");
        if (f.rgb.empty())
            throw std::runtime_error("BiwiLoader: missing RGB: " + stem + "_rgb.png");

        const std::vector<double> pose = readNumbers(stem + "_pose.txt");
        if (pose.size() < 12)
            throw std::runtime_error("BiwiLoader: bad pose: " + stem);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                f.headRotation(r, c) = pose[3 * r + c];
        f.headCenter = Eigen::Vector3d(pose[9], pose[10], pose[11]);

        frames.push_back(std::move(f));
    }
    return frames;
}

BiwiCalibration BiwiLoader::getCalibration() const
{
    BiwiCalibration cal;
    Eigen::Matrix3d Rd;            // depth.cal has R=I, t=0 (depth cam = world)
    Eigen::Vector3d td;
    parseCal(seqPath_ + "/depth.cal", cal.K_depth, Rd, td);
    parseCal(seqPath_ + "/rgb.cal",   cal.K_rgb, cal.R_rgb, cal.t_rgb);
    return cal;
}

// depth geometry helpers

// backproject a uint16 mm depth map into camera frame points (mm), opencv axes
std::vector<Eigen::Vector3d> backprojectDepth(const cv::Mat&         depth,
                                              const Eigen::Matrix3f& K,
                                              int                    stride)
{
    if (depth.empty() || depth.type() != CV_16U)
        throw std::runtime_error("backprojectDepth: expected a 16-bit (mm) depth image");
    if (stride < 1) stride = 1;

    const double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);

    std::vector<Eigen::Vector3d> cloud;
    cloud.reserve(static_cast<size_t>(depth.rows) * depth.cols /
                  static_cast<size_t>(stride * stride));

    for (int r = 0; r < depth.rows; r += stride) {
        const uint16_t* row = depth.ptr<uint16_t>(r);
        for (int c = 0; c < depth.cols; c += stride) {
            const uint16_t z = row[c];
            if (z == 0) continue;                 // no measurement
            const double Z = static_cast<double>(z);
            const double X = (c - cx) / fx * Z;
            const double Y = (r - cy) / fy * Z;   // +Y down (OpenCV), matches BFM_TO_CAM
            cloud.emplace_back(X, Y, Z);
        }
    }
    return cloud;
}

// keep the points within radius of the center then only the front frontSlab mm
std::vector<Eigen::Vector3d> cropHead(const std::vector<Eigen::Vector3d>& cloud,
                                      const Eigen::Vector3d&               center,
                                      double                               radius,
                                      double                               frontSlab)
{
    std::vector<Eigen::Vector3d> sphere;
    for (const Eigen::Vector3d& p : cloud)
        if ((p - center).norm() < radius) sphere.push_back(p);
    if (sphere.empty())
        throw std::runtime_error("cropHead: empty after sphere crop — check head centre");

    double zNear = sphere.front().z();
    for (const Eigen::Vector3d& p : sphere) zNear = std::min(zNear, p.z());

    std::vector<Eigen::Vector3d> head;
    for (const Eigen::Vector3d& p : sphere)
        if (p.z() < zNear + frontSlab) head.push_back(p);
    if (head.empty())
        throw std::runtime_error("cropHead: empty after front-slab crop");

    return head;
}
