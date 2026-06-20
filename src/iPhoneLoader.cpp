#include "IPhoneLoader.h"

#include <opencv2/imgcodecs.hpp>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

// ── tiny JSON scalar reader ──────────────────────────────────────────────────
// intrinsics.json has a flat, known structure; rather than pulling in a JSON
// library we just scan for `"key" :  <number>` patterns. Robust enough because
// the file is written by our own Python module.
static float readJsonNumber(const std::string& blob, const std::string& key)
{
    const std::string needle = "\"" + key + "\"";
    auto pos = blob.find(needle);
    if (pos == std::string::npos)
        throw std::runtime_error("intrinsics.json missing key: " + key);

    pos = blob.find(':', pos);
    if (pos == std::string::npos)
        throw std::runtime_error("intrinsics.json malformed near: " + key);
    ++pos;
    while (pos < blob.size() && std::isspace(static_cast<unsigned char>(blob[pos])))
        ++pos;

    size_t end = pos;
    while (end < blob.size() &&
           (std::isdigit(static_cast<unsigned char>(blob[end])) ||
            blob[end] == '.' || blob[end] == '-' || blob[end] == '+' ||
            blob[end] == 'e' || blob[end] == 'E'))
        ++end;

    if (end == pos)
        throw std::runtime_error("intrinsics.json: could not parse number for: " + key);
    return std::stof(blob.substr(pos, end - pos));
}

// frame index → zero-padded stem, e.g. 0 → "000000"
static std::string frameStem(int i)
{
    std::ostringstream s;
    s << std::setw(6) << std::setfill('0') << i;
    return s.str();
}

// ── IPhoneLoader ─────────────────────────────────────────────────────────────

IPhoneLoader::IPhoneLoader(std::string sessionDir, int numImages)
    : sessionDir_(std::move(sessionDir)), numImages_(numImages)
{
    const std::string intrPath = sessionDir_ + "/intrinsics.json";
    std::ifstream in(intrPath);
    if (!in)
        throw std::runtime_error("could not open " + intrPath +
                                 " (run `python3 -m phone_dataset.cli` first)");

    std::stringstream buf;
    buf << in.rdbuf();
    const std::string blob = buf.str();

    const float fx = readJsonNumber(blob, "fx");
    const float fy = readJsonNumber(blob, "fy");
    const float cx = readJsonNumber(blob, "cx");
    const float cy = readJsonNumber(blob, "cy");
    width_  = static_cast<int>(readJsonNumber(blob, "width"));
    height_ = static_cast<int>(readJsonNumber(blob, "height"));

    K_ << fx, 0,  cx,
          0,  fy, cy,
          0,  0,  1;

    std::cout << "iPhone session: " << sessionDir_ << "\n"
              << "  image size : " << width_ << " × " << height_ << "\n"
              << "  fx = " << fx << "  fy = " << fy
              << "  cx = " << cx << "  cy = " << cy << "\n";
}

std::vector<IPhoneFrame> IPhoneLoader::getFrames() const
{
    std::vector<IPhoneFrame> frames;
    frames.reserve(numImages_);

    for (int i = 0; i < numImages_; ++i) {
        const std::string p = sessionDir_ + "/RGB/" + frameStem(i) + "_RGB.png";
        IPhoneFrame f;
        f.rgb = cv::imread(p, cv::IMREAD_COLOR);
        if (f.rgb.empty()) break;       // stop at first missing
        frames.push_back(std::move(f));
    }
    return frames;
}
