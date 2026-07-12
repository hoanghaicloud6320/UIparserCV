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

bool contains_strict(const RectF& outer, const RectF& inner) {
  const float outer_area = outer.width * outer.height;
  const float inner_area = inner.width * inner.height;
  return outer_area > inner_area && outer.x <= inner.x && outer.y <= inner.y &&
      outer.x + outer.width >= inner.x + inner.width &&
      outer.y + outer.height >= inner.y + inner.height;
}

bool intersects(const RectF& lhs, const RectF& rhs) {
  return std::max(lhs.x, rhs.x) < std::min(lhs.x + lhs.width, rhs.x + rhs.width) &&
      std::max(lhs.y, rhs.y) < std::min(lhs.y + lhs.height, rhs.y + rhs.height);
}

int tree_priority(const UiElementCandidate& candidate) {
  // Tight OCR geometry wins conflicts. Model proposals then outrank optional
  // support sources and legacy diagnostic heuristics.
  if (candidate.kind == UiElementKind::Text) return 4;
  if (candidate.kind == UiElementKind::ModelProposal) return 3;
  if (candidate.kind == UiElementKind::Icon) return 2;
  return 1;
}

std::string label_for(const UiElementCandidate& candidate) {
  std::ostringstream label;
  if (candidate.kind == UiElementKind::ModelProposal) {
    label << "model:" << candidate.model_class;
  } else if (candidate.kind == UiElementKind::Icon) {
    label << "icon";
  } else if (candidate.kind == UiElementKind::Text) {
    label << "text";
  } else if (candidate.kind == UiElementKind::VisualContainer) {
    label << "container";
  } else {
    label << "group";
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

  std::vector<std::size_t> order(candidates.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
    const int lhs_priority = tree_priority(candidates[lhs]);
    const int rhs_priority = tree_priority(candidates[rhs]);
    if (lhs_priority != rhs_priority) return lhs_priority > rhs_priority;
    return candidates[lhs].detection_score > candidates[rhs].detection_score;
  });

  std::vector<std::size_t> selected;
  for (const std::size_t index : order) {
    const auto& box = candidates[index].box;
    const bool partial_overlap = std::any_of(
        selected.begin(), selected.end(), [&](std::size_t kept_index) {
          const auto& kept = candidates[kept_index].box;
          return intersects(box, kept) && !contains_strict(box, kept) &&
              !contains_strict(kept, box);
        });
    if (!partial_overlap) selected.push_back(index);
  }

  std::sort(selected.begin(), selected.end());
  for (const std::size_t i : selected) {
    boxes.push_back(tree::Box{
      static_cast<int>(i) + 2,
      to_tree_rect(candidates[i].box),
      label_for(candidates[i])
    });
  }

  return tree::build_containment_tree(std::move(boxes));
}

} // namespace uiparsercv::pipeline
