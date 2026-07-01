#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace uiparsercv {

struct RectF {
  float x{};
  float y{};
  float width{};
  float height{};
};

struct Detection {
  RectF box{};
  float score{};
  std::string label;
};

struct TextRegion {
  std::vector<cv::Point2f> polygon;
  RectF box{};
  float score{};
};

struct TextRecognition {
  std::string text;
  float confidence{};
};

} // namespace uiparsercv

