#include "LandmarkDetector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <iostream>

namespace {

// dlib/LBF 68-point index → BFM vertex index, the "small" interior set
// (mirrors LBF68_TO_BFM_SMALL resolved through the BFM landmark table). 62 and
// 66 both resolve to 8190 because the nomouth BFM has no inner-mouth vertices.
struct LbfToBfm { int lbf; int vertex; };
const std::array<LbfToBfm, 9> kInterior = {{
    {30,  8156},  // center.nose.tip
    {36,  2736},  // right.eye.corner_outer
    {39,  6219},  // right.eye.corner_inner
    {42,  9892},  // left.eye.corner_inner
    {45, 13360},  // left.eye.corner_outer
    {48,  5779},  // right.lips.corner
    {54, 10598},  // left.lips.corner
    {62,  8190},  // center.lips.upper.inner
    {66,  8190},  // center.lips.lower.inner
}};

// Jawline points emitted as contour observations (vertexIndex -1). Sides only —
// endpoints (near the ears) and the chin are skipped.
const std::array<int, 8> kJawContour = {{1, 3, 5, 7, 9, 11, 13, 15}};

// Common install locations for the Haar cascade, tried if the caller's path
// fails to load.
const char* kCascadeCandidates[] = {
    "models/haarcascade_frontalface_default.xml",                  // stable, preferred
    "/usr/local/lib/python3.12/dist-packages/cv2/data/haarcascade_frontalface_default.xml",
    "/usr/lib/python3/dist-packages/cv2/data/haarcascade_frontalface_default.xml",
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
};

}  // namespace

LandmarkDetector::LandmarkDetector(const std::string& lbfModelPath,
                                   const std::string& cascadePath)
{
    bool cascadeLoaded = !cascadePath.empty() && cascade_.load(cascadePath);
    for (const char* p : kCascadeCandidates) {
        if (cascadeLoaded) break;
        cascadeLoaded = cascade_.load(p);
    }
    if (!cascadeLoaded) {
        std::cerr << "LandmarkDetector: could not load any Haar cascade\n";
        return;
    }

    facemark_ = cv::face::FacemarkLBF::create();
    try {
        facemark_->loadModel(lbfModelPath);
    } catch (const cv::Exception& e) {
        std::cerr << "LandmarkDetector: could not load LBF model '" << lbfModelPath
                  << "' (run the Python landmark tool once to download it): "
                  << e.what() << '\n';
        return;
    }
    ok_ = true;
}

std::vector<LandmarkObservation> LandmarkDetector::detect(const cv::Mat& bgr)
{
    std::vector<LandmarkObservation> obs;
    if (!ok_ || bgr.empty()) return obs;

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    cascade_.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(50, 50));
    if (faces.empty()) return obs;

    // Keep the largest detection (the subject).
    const cv::Rect face = *std::max_element(
        faces.begin(), faces.end(),
        [](const cv::Rect& a, const cv::Rect& b) { return a.area() < b.area(); });

    std::vector<cv::Rect> one{face};
    std::vector<std::vector<cv::Point2f>> landmarks;
    if (!facemark_->fit(bgr, one, landmarks) || landmarks.empty()) return obs;

    const std::vector<cv::Point2f>& p = landmarks[0];
    if (p.size() < 68) return obs;

    for (const LbfToBfm& m : kInterior)
        obs.push_back({m.vertex, Eigen::Vector2d(p[m.lbf].x, p[m.lbf].y)});
    for (int idx : kJawContour)
        obs.push_back({-1, Eigen::Vector2d(p[idx].x, p[idx].y)});
    return obs;
}
