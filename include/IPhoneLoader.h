#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

// One loaded iPhone frame: a single RGB image. Depth is never available.
struct IPhoneFrame {
    cv::Mat rgb;            // BGR, uint8 (cv::imread default)
};

// Loads photos written by the `phone_dataset` Python module:
//
//     data/iphone/<session>/
//         RGB/000000_RGB.png ...
//         intrinsics.json     { fx, fy, cx, cy, width, height, ... }
//
// All frames in a session share the same intrinsic matrix K (the iPhone
// doesn't change its camera between photos in a session).
class IPhoneLoader {
public:
    IPhoneLoader(std::string sessionDir, int numImages);

    std::vector<IPhoneFrame> getFrames() const;
    Eigen::Matrix3f          K() const { return K_; }       // intrinsic matrix
    int                      width()  const { return width_;  }
    int                      height() const { return height_; }

private:
    std::string     sessionDir_;
    int             numImages_;
    Eigen::Matrix3f K_;
    int             width_  = 0;
    int             height_ = 0;
};
