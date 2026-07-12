#include "uiparsercv/pipeline/line_rect_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <stdexcept>
#include <vector>

namespace {

using uiparsercv::RectF;
using uiparsercv::pipeline::UiElementCandidate;
using uiparsercv::pipeline::UiElementKind;

UiElementCandidate text_candidate(RectF box) {
  return UiElementCandidate{
    UiElementKind::Text, box, 1.0F, "text", "text", "text", 1.0F, false, "probe"
  };
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  cv::Mat image(180, 420, CV_8UC3, cv::Scalar(248, 248, 248));
  cv::rectangle(image, {20, 25, 170, 120}, cv::Scalar(30, 30, 30), 3, cv::LINE_8);
  cv::rectangle(image, {230, 25, 170, 120}, cv::Scalar(30, 30, 30), 3, cv::LINE_8);

  const std::vector<UiElementCandidate> atoms{
    text_candidate({40, 48, 60, 18}),
    text_candidate({40, 90, 100, 18}),
    text_candidate({250, 48, 60, 18}),
    text_candidate({250, 90, 100, 18})
  };

  const auto rectangles = uiparsercv::pipeline::detect_line_rects(image, atoms);
  require(rectangles.size() == 2, "two disjoint four-side cards should produce two minimal rectangles");
  for (const auto& rectangle : rectangles) {
    require(rectangle.kind == UiElementKind::VisualContainer, "line rectangle kind mismatch");
    require(rectangle.source == "line_rect_4sides", "line rectangle source mismatch");
  }
  return 0;
}
