#include "uiparsercv/detect/uitag_detector.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool near(float a, float b) { return std::abs(a - b) < 0.001F; }

void test_tile_origins() {
  require(uiparsercv::detect::UitagDetector::tile_origins(500, 640, 512) ==
              std::vector<int>{0},
          "small image must use one origin");
  require(uiparsercv::detect::UitagDetector::tile_origins(1280, 640, 512) ==
              (std::vector<int>{0, 512, 640}),
          "last tile must be edge anchored");
}

void test_decode_and_coordinate_restore() {
  uiparsercv::infer::Tensor output;
  output.shape = {1, 13, 2};
  output.data.assign(26, 0.0F);
  // Prediction 0: cx=160, cy=220, w=80, h=40; class 2 wins.
  output.data[0] = 160.0F;
  output.data[2] = 220.0F;
  output.data[4] = 80.0F;
  output.data[6] = 40.0F;
  output.data[(4 + 2) * 2] = 0.9F;
  // Prediction 1 remains below threshold.
  output.data[(4 + 1) * 2 + 1] = 0.1F;

  const auto decoded = uiparsercv::detect::UitagDetector::decode_output(
      output, {.scale = 0.5F, .left = 10, .top = 20},
      {640, 600}, {512, 100}, {1400, 900}, 0.25F);
  require(decoded.size() == 1, "decoder must filter low confidence predictions");
  require(decoded[0].class_id == 2 && near(decoded[0].score, 0.9F),
          "decoder must select the strongest class");
  require(near(decoded[0].box.x, 732.0F) && near(decoded[0].box.y, 460.0F) &&
              near(decoded[0].box.width, 160.0F) && near(decoded[0].box.height, 80.0F),
          "decoder must undo letterbox and restore full-image coordinates");
}

void test_cross_tile_nms() {
  using uiparsercv::detect::UitagDetection;
  std::vector<UitagDetection> boxes = {
      {{100, 100, 100, 100}, 0.9F, 0},
      {{105, 105, 100, 100}, 0.8F, 1},
      {{400, 400, 20, 20}, 0.7F, 0}};
  const auto class_aware =
      uiparsercv::detect::UitagDetector::nms(boxes, 0.5F, true);
  require(class_aware.size() == 3, "per-tile NMS must preserve overlapping classes");
  const auto cross_tile =
      uiparsercv::detect::UitagDetector::nms(std::move(boxes), 0.5F, false);
  require(cross_tile.size() == 2 && near(cross_tile[0].score, 0.9F),
          "cross-tile NMS must be class agnostic and retain the strongest box");
}

}  // namespace

int main() try {
  test_tile_origins();
  test_decode_and_coordinate_restore();
  test_cross_tile_nms();
  std::cout << "uitag_detector_test: passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "uitag_detector_test: " << error.what() << '\n';
  return 1;
}
