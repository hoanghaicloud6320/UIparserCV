#include "uiparsercv/pipeline/tree_grouper.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace uiparsercv::pipeline {
namespace {

tree::Rect to_tree_rect(const RectF& rect) {
  return tree::Rect{
    static_cast<int>(std::round(rect.x)),
    static_cast<int>(std::round(rect.y)),
    std::max(1, static_cast<int>(std::round(rect.width))),
    std::max(1, static_cast<int>(std::round(rect.height)))
  };
}

std::string label_for(const UiElementCandidate& candidate) {
  std::ostringstream label;
  if (candidate.kind == UiElementKind::Icon) {
    label << "icon";
  } else if (candidate.kind == UiElementKind::Text) {
    label << "text";
  } else {
    label << "container";
  }
  if (!candidate.text.empty()) {
    label << ": " << candidate.text;
  } else {
    label << ": " << candidate.source;
  }
  return label.str();
}

} // namespace

tree::TreeNode build_grouped_ui_tree(
    const std::vector<UiElementCandidate>& candidates,
    int image_width,
    int image_height) {
  std::vector<tree::Box> boxes;
  boxes.reserve(candidates.size() + 1);
  boxes.push_back(tree::Box{1, tree::Rect{0, 0, image_width, image_height}, "screen"});

  for (std::size_t i = 0; i < candidates.size(); ++i) {
    boxes.push_back(tree::Box{
      static_cast<int>(i) + 2,
      to_tree_rect(candidates[i].box),
      label_for(candidates[i])
    });
  }

  return tree::build_containment_tree(std::move(boxes));
}

} // namespace uiparsercv::pipeline
