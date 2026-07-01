#include "uiparsercv/pipeline/tree_grouper.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>
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
  label << (candidate.kind == UiElementKind::Icon ? "icon" : "text");
  if (!candidate.text.empty()) {
    label << ": " << candidate.text;
  } else {
    label << ": " << candidate.source;
  }
  return label.str();
}

float center_y(const RectF& rect) {
  return rect.y + rect.height * 0.5F;
}

float vertical_overlap_ratio(const RectF& lhs, const RectF& rhs) {
  const float top = std::max(lhs.y, rhs.y);
  const float bottom = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
  const float overlap = std::max(0.0F, bottom - top);
  const float min_height = std::max(1.0F, std::min(lhs.height, rhs.height));
  return overlap / min_height;
}

float horizontal_gap(const RectF& lhs, const RectF& rhs) {
  if (lhs.x <= rhs.x) {
    return rhs.x - (lhs.x + lhs.width);
  }
  return lhs.x - (rhs.x + rhs.width);
}

tree::Rect union_rect(const tree::Rect& lhs, const tree::Rect& rhs) {
  const int left = std::min(lhs.x, rhs.x);
  const int top = std::min(lhs.y, rhs.y);
  const int right = std::max(lhs.x + lhs.width, rhs.x + rhs.width);
  const int bottom = std::max(lhs.y + lhs.height, rhs.y + rhs.height);
  return tree::Rect{left, top, std::max(1, right - left), std::max(1, bottom - top)};
}

tree::Rect padded(tree::Rect rect, int padding, int image_width, int image_height) {
  const int left = std::max(0, rect.x - padding);
  const int top = std::max(0, rect.y - padding);
  const int right = std::min(image_width, rect.x + rect.width + padding);
  const int bottom = std::min(image_height, rect.y + rect.height + padding);
  return tree::Rect{left, top, std::max(1, right - left), std::max(1, bottom - top)};
}

tree::Rect bounds_for_indices(
    const std::vector<UiElementCandidate>& candidates,
    const std::vector<int>& indices,
    int image_width,
    int image_height) {
  tree::Rect bounds = to_tree_rect(candidates[static_cast<std::size_t>(indices.front())].box);
  for (std::size_t i = 1; i < indices.size(); ++i) {
    bounds = union_rect(
        bounds,
        to_tree_rect(candidates[static_cast<std::size_t>(indices[i])].box));
  }
  return padded(bounds, 6, image_width, image_height);
}

tree::TreeNode candidate_node(const std::vector<UiElementCandidate>& candidates, int index) {
  const auto& candidate = candidates[static_cast<std::size_t>(index)];
  return tree::TreeNode{
    tree::Box{index + 2, to_tree_rect(candidate.box), label_for(candidate)},
    {}
  };
}

struct GroupSpec {
  std::string label;
  std::vector<int> indices;
};

std::vector<GroupSpec> find_button_groups(const std::vector<UiElementCandidate>& candidates) {
  std::vector<GroupSpec> groups;
  std::vector<bool> used(candidates.size(), false);

  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (used[i]) {
      continue;
    }
    for (std::size_t j = 0; j < candidates.size(); ++j) {
      if (i == j || used[j]) {
        continue;
      }

      const auto& lhs = candidates[i];
      const auto& rhs = candidates[j];
      if (lhs.kind == rhs.kind || vertical_overlap_ratio(lhs.box, rhs.box) < 0.45F) {
        continue;
      }

      const float gap = horizontal_gap(lhs.box, rhs.box);
      const float row_height = std::max(lhs.box.height, rhs.box.height);
      if (gap >= -4.0F && gap <= std::max(18.0F, row_height * 0.8F)) {
        std::vector<int> indices{static_cast<int>(i), static_cast<int>(j)};
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
          return candidates[static_cast<std::size_t>(a)].box.x <
                 candidates[static_cast<std::size_t>(b)].box.x;
        });
        groups.push_back(GroupSpec{"group: button", indices});
        used[i] = true;
        used[j] = true;
        break;
      }
    }
  }

  return groups;
}

std::vector<int> find_input_bar_group(
    const std::vector<UiElementCandidate>& candidates,
    const std::vector<bool>& already_grouped,
    int image_width,
    int image_height) {
  std::vector<int> best;

  for (std::size_t seed = 0; seed < candidates.size(); ++seed) {
    if (already_grouped[seed]) {
      continue;
    }
    if (center_y(candidates[seed].box) > static_cast<float>(image_height) * 0.60F) {
      continue;
    }

    std::vector<int> row;
    const float y = center_y(candidates[seed].box);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      if (already_grouped[i]) {
        continue;
      }
      const auto& box = candidates[i].box;
      if (std::abs(center_y(box) - y) < std::max(18.0F, box.height * 0.75F)) {
        row.push_back(static_cast<int>(i));
      }
    }

    if (row.size() < best.size()) {
      continue;
    }
    std::sort(row.begin(), row.end(), [&](int a, int b) {
      return candidates[static_cast<std::size_t>(a)].box.x <
             candidates[static_cast<std::size_t>(b)].box.x;
    });

    const auto& first = candidates[static_cast<std::size_t>(row.front())].box;
    const auto& last = candidates[static_cast<std::size_t>(row.back())].box;
    const float span = last.x + last.width - first.x;
    if (row.size() >= 4 && span > static_cast<float>(image_width) * 0.45F) {
      best = row;
    }
  }

  return best;
}

} // namespace

tree::TreeNode build_grouped_ui_tree(
    const std::vector<UiElementCandidate>& candidates,
    int image_width,
    int image_height) {
  tree::TreeNode root{tree::Box{0, tree::Rect{0, 0, 0, 0}, "root"}, {}};
  tree::TreeNode screen{tree::Box{1, tree::Rect{0, 0, image_width, image_height}, "screen"}, {}};

  std::vector<bool> grouped(candidates.size(), false);
  std::vector<GroupSpec> groups;

  if (const auto input_indices = find_input_bar_group(candidates, grouped, image_width, image_height);
      !input_indices.empty()) {
    groups.push_back(GroupSpec{"group: input_bar", input_indices});
    for (const int index : input_indices) {
      grouped[static_cast<std::size_t>(index)] = true;
    }
  }

  auto button_groups = find_button_groups(candidates);
  for (auto& group : button_groups) {
    const bool available = std::all_of(group.indices.begin(), group.indices.end(), [&](int index) {
      return !grouped[static_cast<std::size_t>(index)];
    });
    if (!available) {
      continue;
    }
    for (const int index : group.indices) {
      grouped[static_cast<std::size_t>(index)] = true;
    }
    groups.push_back(std::move(group));
  }

  int group_id = static_cast<int>(candidates.size()) + 2;
  for (const auto& group : groups) {
    tree::TreeNode group_node{
      tree::Box{
        group_id++,
        bounds_for_indices(candidates, group.indices, image_width, image_height),
        group.label
      },
      {}
    };
    for (const int index : group.indices) {
      group_node.children.push_back(candidate_node(candidates, index));
    }
    screen.children.push_back(std::move(group_node));
  }

  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (!grouped[i]) {
      screen.children.push_back(candidate_node(candidates, static_cast<int>(i)));
    }
  }

  std::stable_sort(screen.children.begin(), screen.children.end(), [](const auto& lhs, const auto& rhs) {
    if (std::abs(lhs.box.rect.y - rhs.box.rect.y) < 10) {
      return lhs.box.rect.x < rhs.box.rect.x;
    }
    return lhs.box.rect.y < rhs.box.rect.y;
  });

  root.children.push_back(std::move(screen));
  return root;
}

} // namespace uiparsercv::pipeline
