#include "uiparsercv/pipeline/tree_grouper.hpp"
#include "uiparsercv/pipeline/ui_element.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool overlaps_partially(const uiparsercv::tree::Rect& a, const uiparsercv::tree::Rect& b) {
  const bool intersects = std::max(a.x, b.x) < std::min(a.x + a.width, b.x + b.width) &&
      std::max(a.y, b.y) < std::min(a.y + a.height, b.y + b.height);
  return intersects && !uiparsercv::tree::contains(a, b) && !uiparsercv::tree::contains(b, a);
}

void assert_laminar(const uiparsercv::tree::TreeNode& node) {
  for (std::size_t i = 0; i < node.children.size(); ++i) {
    for (std::size_t j = i + 1; j < node.children.size(); ++j) {
      require(!overlaps_partially(node.children[i].box.rect, node.children[j].box.rect),
              "tree siblings must not partially overlap");
    }
    assert_laminar(node.children[i]);
  }
}

}  // namespace

int main() try {
  using namespace uiparsercv;
  const std::vector<detect::UitagDetection> model = {
      {{10, 10, 180, 60}, 0.9F, 0},
      {{150, 20, 100, 60}, 0.6F, 8}};
  const std::vector<Detection> icons = {
      {{300, 10, 30, 30}, 0.8F, "icon"},
      {{20, 20, 20, 20}, 0.7F, "duplicate"}};
  const std::vector<TextRegion> regions = {
      {{}, {20, 20, 80, 20}, 0.95F},
      {{}, {400, 20, 60, 20}, 0.9F}};
  const std::vector<TextRecognition> text = {{"Submit", 0.98F}, {"OCR only", 0.9F}};

  const auto combined = pipeline::build_model_ocr_candidates(model, icons, regions, text);
  require(combined.candidates.size() == 5,
          "pipeline must keep model proposals, novel icon, and every OCR box");
  require(combined.associations.size() == 1,
          "only unambiguous contained text should be associated");
  const auto& association = combined.associations.front();
  require(association.proposal_index == 0 && association.text_index == 3,
          "association indices must point to preserved observations");
  require(combined.candidates[3].box.width == 80 && combined.candidates[3].text == "Submit",
          "association must not replace tight OCR geometry or content");

  const auto tree = pipeline::build_grouped_ui_tree(combined.candidates, 500, 200);
  assert_laminar(tree);
  std::cout << "combined_pipeline_test: passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "combined_pipeline_test: " << error.what() << '\n';
  return 1;
}
