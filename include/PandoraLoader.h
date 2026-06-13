// Implemented with Claude (Anthropic Claude Code) on 2026-06-13.
#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>

// One loaded Pandora frame (mirrors python/dataset.py PandoraFrame).
//   rgb   : 1920x1080, 8-bit BGRA (4 channels)
//   depth : 512x424,  uint16 millimetres (empty when loaded RGB-only)
struct PandoraFrame {
    cv::Mat rgb;
    cv::Mat depth;
};

// Loads the first `numImages` frames of one Pandora sequence.
// You pass the values in -> getFrames() gives you the list back.
class PandoraLoader {
public:
    PandoraLoader(std::string path, int numImages, bool loadRGBD);
    std::vector<PandoraFrame> getFrames() const;

private:
    std::string path_;
    int         numImages_;
    bool        loadRGBD_;
};
