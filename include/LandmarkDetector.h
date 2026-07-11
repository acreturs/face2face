#pragma once
#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/objdetect.hpp>

#include <string>
#include <vector>

#include "CeresFitter.h"   // LandmarkObservation

// In-process 68-point facial-landmark detection — the C++ twin of
// python/landmarks.py, so RGB video can be tracked without a per-frame Python
// round-trip.
//
// Face box: prefers YuNet (a CNN detector robust to head pose); falls back to a
// Haar frontal cascade if the YuNet model is unavailable. Landmarks: OpenCV LBF
// facemark on the detected box.
//
// detect() returns solver observations directly: interior points mapped to their
// BFM vertex indices + the jawline as CONTOUR points (vertexIndex -1). Empty ⇒
// no face found.
class LandmarkDetector {
public:
    // lbfModelPath: pretrained lbfmodel.yaml (downloaded by the Python tool).
    // yunetPath: YuNet ONNX model; empty/missing ⇒ use the Haar fallback.
    // cascadePath: optional explicit Haar path (else standard locations tried).
    LandmarkDetector(const std::string& lbfModelPath,
                     const std::string& yunetPath   = "",
                     const std::string& cascadePath = "");

    std::vector<LandmarkObservation> detect(const cv::Mat& bgr);

    bool ok() const { return ok_; }
    bool usingYuNet() const { return static_cast<bool>(yunet_); }

private:
    // Detects the face box. When YuNet is active, also returns its 5 pose-robust
    // landmarks (right eye, left eye, nose, right mouth, left mouth) in `pts`.
    bool faceBox(const cv::Mat& bgr, cv::Rect& box, std::vector<cv::Point2f>* pts);

    cv::Ptr<cv::face::Facemark> facemark_;
    cv::Ptr<cv::FaceDetectorYN> yunet_;
    cv::CascadeClassifier       cascade_;
    bool                        ok_ = false;
};
