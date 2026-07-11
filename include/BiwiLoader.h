// Implemented with Claude (Anthropic Claude Code) on 2026-07-02.
// Mirrors python/biwi.py; conventions verified by python/check_biwi.py.
#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

// Biwi Kinect Head Pose Database (data/biwi/NN/): RGB + depth from one Kinect,
// both cameras calibrated. Depth cam = world; rgb.cal maps depth points into
// the RGB camera via p_rgb = R*p_depth + t (mm). Axes are OpenCV (+Y down, mm).

// rgb.cal + depth.cal of one sequence.
struct BiwiCalibration {
    Eigen::Matrix3f K_depth;   // fx=fy=575.8, cx=320, cy=240
    Eigen::Matrix3f K_rgb;     // fx=fy=517.7, cx=320, cy=240.5
    Eigen::Matrix3d R_rgb;     // depth cam -> RGB cam
    Eigen::Vector3d t_rgb;     // depth cam -> RGB cam (mm)
};

// One loaded Biwi frame.
struct BiwiFrame {
    int             frameNumber;  // Biwi frames start at 3
    cv::Mat         rgb;          // 640x480, 8-bit BGR
    cv::Mat         depth;        // 640x480, uint16 mm (RLE-decoded)
    Eigen::Matrix3d headRotation; // GT head rotation from *_pose.txt
    Eigen::Vector3d headCenter;   // GT head centre (mm, depth cam)
};

// Loads the first `numImages` frames of a Biwi sequence (e.g. "data/biwi/01").
class BiwiLoader {
public:
    BiwiLoader(std::string seqPath, int numImages);

    std::vector<BiwiFrame> getFrames() const;
    BiwiCalibration getCalibration() const;

private:
    static cv::Mat readDepthBin(const std::string& path);

    std::string seqPath_;
    int         numImages_;
};

// depth geometry helpers for the dense fit (shared with the Biwi pipeline).

// Backproject a uint16 mm depth map to camera-frame points (mm), OpenCV axes.
std::vector<Eigen::Vector3d> backprojectDepth(const cv::Mat&         depth,
                                              const Eigen::Matrix3f& K,
                                              int                    stride = 1);

// Keep the head: points within `radius` of `center`, then the front `frontSlab`
// mm. Tight crop (90/80) is the dominant lever for a clean dense fit.
std::vector<Eigen::Vector3d> cropHead(const std::vector<Eigen::Vector3d>& cloud,
                                      const Eigen::Vector3d&               center,
                                      double                               radius    = 90.0,
                                      double                               frontSlab = 80.0);
