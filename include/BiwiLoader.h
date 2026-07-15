#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

// reads the Biwi Kinect dataset (data/biwi/NN/) which is RGB + depth from one
// Kinect with both cameras calibrated. the depth cam is treated as the world
// and rgb.cal maps depth points into the RGB camera with p_rgb = R*p_depth + t
// everything is in mm and OpenCV axes (+Y down). mirrors python/biwi.py

// the rgb.cal and depth.cal numbers for one sequence
struct BiwiCalibration {
    Eigen::Matrix3f K_depth;   // fx=fy=575.8, cx=320, cy=240
    Eigen::Matrix3f K_rgb;     // fx=fy=517.7, cx=320, cy=240.5
    Eigen::Matrix3d R_rgb;     // depth cam -> RGB cam
    Eigen::Vector3d t_rgb;     // depth cam -> RGB cam (mm)
};

// one loaded frame
struct BiwiFrame {
    int             frameNumber;  // Biwi numbering starts at 3
    cv::Mat         rgb;          // 640x480 8-bit BGR
    cv::Mat         depth;        // 640x480 uint16 mm (RLE-decoded)
    Eigen::Matrix3d headRotation; // ground-truth head rotation
    Eigen::Vector3d headCenter;   // ground-truth head centre (mm, depth cam)
};

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

// depth helpers shared with the dense fit

// turn a uint16 mm depth map into 3D camera-frame points (mm)
std::vector<Eigen::Vector3d> backprojectDepth(const cv::Mat&         depth,
                                              const Eigen::Matrix3f& K,
                                              int                    stride = 1);

// keep only the head, points within `radius` of `center` and then the front
// `frontSlab` mm. a tight crop is the biggest lever for a clean dense fit
std::vector<Eigen::Vector3d> cropHead(const std::vector<Eigen::Vector3d>& cloud,
                                      const Eigen::Vector3d&               center,
                                      double                               radius    = 90.0,
                                      double                               frontSlab = 80.0);
