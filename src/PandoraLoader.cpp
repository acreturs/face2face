// Implemented with Claude (Anthropic Claude Code) on 2026-06-13.
#include "PandoraLoader.h"
#include <opencv2/imgcodecs.hpp>
#include <iomanip>
#include <sstream>
#include <utility>

PandoraLoader::PandoraLoader(std::string path, int numImages, bool loadRGBD)
    : path_(std::move(path)), numImages_(numImages), loadRGBD_(loadRGBD) {}

// frame index -> zero-padded stem, e.g. 0 -> "000000" -> AI fix
static std::string frameStem(int i)
{
    std::ostringstream s;
    s << std::setw(6) << std::setfill('0') << i;
    return s.str();
}

std::vector<PandoraFrame> PandoraLoader::getFrames() const
{
    std::vector<PandoraFrame> frames;
    for (int i = 0; i < numImages_; ++i) {
        const std::string stem = frameStem(i);
        PandoraFrame f;
        f.rgb = cv::imread(path_ + "/RGB/" + stem + "_RGB.png", cv::IMREAD_UNCHANGED);
        if (loadRGBD_)
            f.depth = cv::imread(path_ + "/DEPTH/" + stem + "_DEPTH.png", cv::IMREAD_UNCHANGED);

        if (f.rgb.empty()) break;            // stop at the first missing frame
        frames.push_back(std::move(f));      // collect it into the result list
    }
    return frames;
}
