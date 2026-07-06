#pragma once
#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/objdetect.hpp>

#include <string>
#include <vector>

#include "CeresFitter.h"   // LandmarkObservation

// In-process 68-point facial-landmark detection (OpenCV LBF facemark + a Haar
// face detector) — the C++ twin of python/landmarks.py, so RGB video can be
// tracked without a per-frame Python round-trip.
//
// detect() returns solver observations directly: the interior points mapped to
// their BFM vertex indices, plus the jawline as CONTOUR points (vertexIndex -1,
// matched dynamically by the fitter). Empty vector ⇒ no face found.
class LandmarkDetector {
public:
    // lbfModelPath: the pretrained lbfmodel.yaml (downloaded by the Python tool
    // into models/). cascadePath: a Haar frontal-face cascade; if it fails to
    // load, a few standard system locations are tried.
    explicit LandmarkDetector(const std::string& lbfModelPath,
                              const std::string& cascadePath = "");

    std::vector<LandmarkObservation> detect(const cv::Mat& bgr);

    bool ok() const { return ok_; }

private:
    cv::Ptr<cv::face::Facemark> facemark_;
    cv::CascadeClassifier       cascade_;
    bool                        ok_ = false;
};
