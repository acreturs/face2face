#pragma once
#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/objdetect.hpp>

#include <string>
#include <vector>

#include "CeresFitter.h"   // LandmarkObservation

// finds 68 facial landmarks in an image without leaving C++ so RGB video can
// be tracked without calling out to python every frame. this is the twin of
// python/landmarks.py
//
// for the face box it prefers YuNet (a CNN that copes with head pose) and
// falls back to a Haar frontal cascade if the YuNet model is missing. the
// landmarks themselves come from OpenCV's LBF facemark on that box
class LandmarkDetector {
public:
    // lbfModelPath points at the pretrained lbfmodel.yaml
    // yunetPath is the YuNet ONNX model, leave empty to force the Haar fallback
    // cascadePath is an optional explicit Haar path
    LandmarkDetector(const std::string& lbfModelPath,
                     const std::string& yunetPath   = "",
                     const std::string& cascadePath = "");

    // gives back solver-ready observations, interior points carry their BFM
    // vertex index and the jawline comes back as contour points (index -1).
    // empty means no face was found
    std::vector<LandmarkObservation> detect(const cv::Mat& bgr);

    bool ok() const { return ok_; }
    bool usingYuNet() const { return static_cast<bool>(yunet_); }

private:
    // finds the face box. with YuNet active it also fills `pts` with its 5
    // pose-robust points (eyes, nose, mouth corners)
    bool faceBox(const cv::Mat& bgr, cv::Rect& box, std::vector<cv::Point2f>* pts);

    cv::Ptr<cv::face::Facemark> facemark_;
    cv::Ptr<cv::FaceDetectorYN> yunet_;
    cv::CascadeClassifier       cascade_;
    bool                        ok_ = false;
};
