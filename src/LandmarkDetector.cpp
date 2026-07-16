// finds a face and its landmarks in an image
// YuNet or Haar gives the box, LBF gives the 68 points
#include "LandmarkDetector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>

namespace {

// dlib/lbf 68-point index -> bfm vertex, the small interior set (mirrors
// LBF68_TO_BFM_SMALL through the bfm landmark table). 62 and 66 both map to
// 8190 because the nomouth bfm has no inner-mouth vertices
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

// yunet 5-landmark index -> bfm vertex. these cnn landmarks stay accurate under
// head rotation (unlike lbf's frontal-biased regression) so they drive the
// pose. order matches yunet: right eye, left eye, nose, right/left mouth corner
struct YuToBfm { int yu; int vertex; };
const std::array<YuToBfm, 5> kYuNet = {{
    {0,  4540},  // right.eye.pupil.center
    {1, 11681},  // left.eye.pupil.center
    {2,  8156},  // center.nose.tip
    {3,  5779},  // right.lips.corner
    {4, 10598},  // left.lips.corner
}};

// jawline points emitted as contour observations (vertexIndex -1), sides only
const std::array<int, 8> kJawContour = {{1, 3, 5, 7, 9, 11, 13, 15}};

const char* kCascadeCandidates[] = {
    "models/haarcascade_frontalface_default.xml",
    "/usr/local/lib/python3.12/dist-packages/cv2/data/haarcascade_frontalface_default.xml",
    "/usr/lib/python3/dist-packages/cv2/data/haarcascade_frontalface_default.xml",
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
};

}  // namespace

LandmarkDetector::LandmarkDetector(const std::string& lbfModelPath,
                                   const std::string& yunetPath,
                                   const std::string& cascadePath)
{
    // prefer the yunet cnn detector, it's robust to head pose. its onnx model
    // needs opencv's >= 4.8 dnn backend, on older builds create() works but the
    // first detect() throws "Layer id=-1 not found", so check the version up
    // front and fall back to haar with a clear message instead of a per-frame
    // dnn error each run
    const int ocv = cv::getVersionMajor() * 100 + cv::getVersionMinor();
    if (!yunetPath.empty() && std::filesystem::exists(yunetPath)) {
        if (ocv < 408) {
            std::cerr << "LandmarkDetector: YuNet needs OpenCV >= 4.8 but this "
                         "build links " << cv::getVersionMajor() << '.'
                      << cv::getVersionMinor() << " — using the Haar detector.\n"
                         "  Rebuild the devcontainer (Dev Containers: Rebuild "
                         "Container) to get OpenCV 4.9 + YuNet.\n";
        } else {
            try {
                yunet_ = cv::FaceDetectorYN::create(
                    yunetPath, "", cv::Size(320, 320),
                    /*score=*/0.6f, /*nms=*/0.3f, /*top_k=*/5000);
            } catch (const cv::Exception& e) {
                std::cerr << "LandmarkDetector: YuNet load failed (" << e.what()
                          << ") — falling back to Haar\n";
                yunet_.release();
            }
        }
    }

    // always keep a haar cascade as a fallback, yunet can also fail at inference
    // time on older opencv, in which case faceBox() switches to haar at runtime
    {
        bool loaded = !cascadePath.empty() && cascade_.load(cascadePath);
        for (const char* p : kCascadeCandidates) {
            if (loaded) break;
            loaded = cascade_.load(p);
        }
        if (!yunet_ && !loaded) {
            std::cerr << "LandmarkDetector: no face detector (YuNet nor Haar)\n";
            return;
        }
    }

    facemark_ = cv::face::FacemarkLBF::create();
    try {
        facemark_->loadModel(lbfModelPath);
    } catch (const cv::Exception& e) {
        std::cerr << "LandmarkDetector: could not load LBF model '" << lbfModelPath
                  << "': " << e.what() << '\n';
        return;
    }
    ok_ = true;
    std::cout << "LandmarkDetector: " << (yunet_ ? "YuNet" : "Haar")
              << " face detector + LBF 68-point landmarks\n";
}

// find the face box (and 5 yunet points if we have them), yunet first then haar
bool LandmarkDetector::faceBox(const cv::Mat& bgr, cv::Rect& box,
                               std::vector<cv::Point2f>* pts)
{
    if (pts) pts->clear();
    if (yunet_) {
        try {
            // yunet is tuned for small inputs, on a multi-megapixel frame (a
            // 2316-wide iphone selfie) it misses the face. downscale to ~640 px
            // wide for detection then map the results back to full res
            constexpr int kDetWidth = 640;
            double s = std::min(1.0, static_cast<double>(kDetWidth) / bgr.cols);
            cv::Mat small = bgr;
            if (s < 1.0) cv::resize(bgr, small, cv::Size(), s, s, cv::INTER_AREA);
            else         s = 1.0;

            yunet_->setInputSize(small.size());
            cv::Mat faces;                             // (N × 15): bbox, 5 pts, score
            yunet_->detect(small, faces);
            if (faces.rows == 0) return false;
            int best = 0;
            for (int r = 1; r < faces.rows; ++r)
                if (faces.at<float>(r, 14) > faces.at<float>(best, 14)) best = r;
            const double inv = 1.0 / s;                // small → full-res scale
            box = cv::Rect(cvRound(faces.at<float>(best, 0) * inv),
                           cvRound(faces.at<float>(best, 1) * inv),
                           cvRound(faces.at<float>(best, 2) * inv),
                           cvRound(faces.at<float>(best, 3) * inv));
            box &= cv::Rect(0, 0, bgr.cols, bgr.rows);
            if (pts)
                for (int k = 0; k < 5; ++k)
                    pts->emplace_back(
                        static_cast<float>(faces.at<float>(best, 4 + 2 * k) * inv),
                        static_cast<float>(faces.at<float>(best, 5 + 2 * k) * inv));
            return box.area() > 0;
        } catch (const cv::Exception& e) {
            std::cerr << "LandmarkDetector: YuNet inference failed (" << e.what()
                      << ") — switching to Haar for the rest of the run\n";
            yunet_.release();      // fall through to haar below and stay there
        }
    }
    if (cascade_.empty()) return false;

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);
    std::vector<cv::Rect> faces;
    cascade_.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(50, 50));
    if (faces.empty()) return false;
    box = *std::max_element(
        faces.begin(), faces.end(),
        [](const cv::Rect& a, const cv::Rect& b) { return a.area() < b.area(); });
    return true;
}

// run detection and return interior + jaw-contour observations for one image
std::vector<LandmarkObservation> LandmarkDetector::detect(const cv::Mat& bgr)
{
    std::vector<LandmarkObservation> obs;
    if (!ok_ || bgr.empty()) return obs;

    cv::Rect box;
    std::vector<cv::Point2f> yuPts;
    if (!faceBox(bgr, box, &yuPts)) return obs;

    // lbf 68-point fit on the box (used for the jaw contour, and for the
    // interior when yunet isn't there). lbf was trained on haar-style boxes and
    // yunet's tighter rectangle shifts its regression, a centred square at 1.1x
    // the larger side matched the haar convention best (checked against
    // mediapipe on biwi frames)
    cv::Rect lbfBox = box;
    if (!yuPts.empty()) {
        const int side = static_cast<int>(1.1f * std::max(box.width, box.height));
        lbfBox = cv::Rect(box.x + box.width / 2 - side / 2,
                          box.y + box.height / 2 - side / 2, side, side);
        lbfBox &= cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (lbfBox.area() <= 0) lbfBox = box;
    }
    std::vector<cv::Rect> one{lbfBox};
    std::vector<std::vector<cv::Point2f>> landmarks;
    const bool haveLbf = facemark_->fit(bgr, one, landmarks) &&
                         !landmarks.empty() && landmarks[0].size() >= 68;

    // interior (pose-critical) points, prefer yunet's pose-robust landmarks and
    // fall back to lbf's frontal-biased ones when yunet isn't available
    if (yuPts.size() >= 5) {
        for (const YuToBfm& m : kYuNet)
            obs.push_back({m.vertex, Eigen::Vector2d(yuPts[m.yu].x, yuPts[m.yu].y)});
    } else if (haveLbf) {
        for (const LbfToBfm& m : kInterior)
            obs.push_back({m.vertex,
                           Eigen::Vector2d(landmarks[0][m.lbf].x, landmarks[0][m.lbf].y)});
    }

    // jaw contour (width/silhouette) always comes from lbf, if we have it
    if (haveLbf)
        for (int idx : kJawContour)
            obs.push_back({-1, Eigen::Vector2d(landmarks[0][idx].x, landmarks[0][idx].y)});

    return obs;   // empty if neither YuNet points nor LBF were usable
}
