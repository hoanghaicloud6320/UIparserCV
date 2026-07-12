#include "uiparsercv/pipeline/boundaryless_group_detector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace uiparsercv::pipeline {
namespace {

float area(const RectF& box) {
  return std::max(0.0F, box.width) * std::max(0.0F, box.height);
}

float intersection_area(const RectF& lhs, const RectF& rhs) {
  const float left = std::max(lhs.x, rhs.x);
  const float top = std::max(lhs.y, rhs.y);
  const float right = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
  const float bottom = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
  return std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
}

bool strict_contains(const RectF& outer, const RectF& inner) {
  return area(outer) > area(inner) &&
         outer.x <= inner.x && outer.y <= inner.y &&
         outer.x + outer.width >= inner.x + inner.width &&
         outer.y + outer.height >= inner.y + inner.height;
}

bool mostly_contains(const RectF& outer, const RectF& inner) {
  const float inner_area = area(inner);
  return area(outer) > inner_area && inner_area > 0.0F &&
         intersection_area(outer, inner) / inner_area >= 0.90F;
}

bool already_explained(
    const std::vector<UiElementCandidate>& candidates,
    int first,
    int second) {
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (static_cast<int>(i) == first || static_cast<int>(i) == second) {
      continue;
    }
    if (mostly_contains(candidates[i].box, candidates[static_cast<std::size_t>(first)].box) &&
        mostly_contains(candidates[i].box, candidates[static_cast<std::size_t>(second)].box)) {
      return true;
    }
  }
  return false;
}

float axis_overlap(float a0, float a1, float b0, float b1) {
  return std::max(0.0F, std::min(a1, b1) - std::max(a0, b0));
}

RectF union_box(const RectF& lhs, const RectF& rhs, int image_width, int image_height) {
  const float padding = 4.0F;
  const float left = std::max(0.0F, std::min(lhs.x, rhs.x) - padding);
  const float top = std::max(0.0F, std::min(lhs.y, rhs.y) - padding);
  const float right = std::min(
      static_cast<float>(image_width),
      std::max(lhs.x + lhs.width, rhs.x + rhs.width) + padding);
  const float bottom = std::min(
      static_cast<float>(image_height),
      std::max(lhs.y + lhs.height, rhs.y + rhs.height) + padding);
  return RectF{left, top, right - left, bottom - top};
}

enum class PatternType {
  HorizontalIconText,
  VerticalTextStack
};

struct Proposal {
  int first{};
  int second{};
  PatternType type{};
  RectF box{};
  float normalized_gap{};
  float size_ratio{};
  float score{};
};

float external_gap(
    const std::vector<UiElementCandidate>& candidates,
    int first,
    int second,
    const RectF& group,
    PatternType type) {
  float nearest = std::numeric_limits<float>::max();
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (static_cast<int>(i) == first || static_cast<int>(i) == second) {
      continue;
    }
    const auto& box = candidates[i].box;
    if (type == PatternType::HorizontalIconText) {
      const float overlap = axis_overlap(group.y, group.y + group.height, box.y, box.y + box.height);
      if (overlap < std::min(group.height, box.height) * 0.35F) {
        continue;
      }
      const float gap = box.x >= group.x + group.width
          ? box.x - (group.x + group.width)
          : (group.x >= box.x + box.width ? group.x - (box.x + box.width) : 0.0F);
      nearest = std::min(nearest, gap);
    } else {
      const float overlap = axis_overlap(group.x, group.x + group.width, box.x, box.x + box.width);
      if (overlap < std::min(group.width, box.width) * 0.35F) {
        continue;
      }
      const float gap = box.y >= group.y + group.height
          ? box.y - (group.y + group.height)
          : (group.y >= box.y + box.height ? group.y - (box.y + box.height) : 0.0F);
      nearest = std::min(nearest, gap);
    }
  }
  return nearest;
}

bool similar(const Proposal& lhs, const Proposal& rhs) {
  if (lhs.type != rhs.type) {
    return false;
  }
  const float gap_delta = std::abs(lhs.normalized_gap - rhs.normalized_gap);
  const float ratio = lhs.size_ratio / std::max(0.01F, rhs.size_ratio);
  const float width_ratio = lhs.box.width / std::max(1.0F, rhs.box.width);
  const float height_ratio = lhs.box.height / std::max(1.0F, rhs.box.height);
  return gap_delta <= 0.28F &&
         ratio >= 0.55F && ratio <= 1.80F &&
         width_ratio >= 0.55F && width_ratio <= 1.80F &&
         height_ratio >= 0.55F && height_ratio <= 1.80F;
}

bool partial_overlap(const RectF& lhs, const RectF& rhs) {
  const float intersection = intersection_area(lhs, rhs);
  if (intersection <= 0.0F) {
    return false;
  }
  return !strict_contains(lhs, rhs) && !strict_contains(rhs, lhs);
}

} // namespace

std::vector<UiElementCandidate> detect_boundaryless_groups(
    const std::vector<UiElementCandidate>& candidates,
    int image_width,
    int image_height,
    BoundarylessGroupOptions options) {
  std::vector<UiElementCandidate> result;
  if (!options.enabled || candidates.size() < 4) {
    return result;
  }

  std::vector<Proposal> proposals;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    for (std::size_t j = i + 1; j < candidates.size(); ++j) {
      const auto& lhs = candidates[i];
      const auto& rhs = candidates[j];
      if (lhs.kind == UiElementKind::VisualContainer || lhs.kind == UiElementKind::InferredGroup ||
          rhs.kind == UiElementKind::VisualContainer || rhs.kind == UiElementKind::InferredGroup ||
          strict_contains(lhs.box, rhs.box) || strict_contains(rhs.box, lhs.box) ||
          already_explained(candidates, static_cast<int>(i), static_cast<int>(j))) {
        continue;
      }

      Proposal proposal;
      proposal.first = static_cast<int>(i);
      proposal.second = static_cast<int>(j);
      proposal.box = union_box(lhs.box, rhs.box, image_width, image_height);

      const bool icon_text =
          (lhs.kind == UiElementKind::Icon && rhs.kind == UiElementKind::Text) ||
          (lhs.kind == UiElementKind::Text && rhs.kind == UiElementKind::Icon);
      if (icon_text) {
        const auto& left = lhs.box.x <= rhs.box.x ? lhs.box : rhs.box;
        const auto& right = lhs.box.x <= rhs.box.x ? rhs.box : lhs.box;
        const float gap = right.x - (left.x + left.width);
        const float scale = std::max(1.0F, std::max(lhs.box.height, rhs.box.height));
        const float overlap = axis_overlap(
            lhs.box.y, lhs.box.y + lhs.box.height, rhs.box.y, rhs.box.y + rhs.box.height);
        const float overlap_ratio = overlap / std::max(1.0F, std::min(lhs.box.height, rhs.box.height));
        if (gap < -2.0F || gap / scale > options.max_normalized_gap ||
            overlap_ratio < options.min_axis_overlap) {
          continue;
        }
        proposal.type = PatternType::HorizontalIconText;
        proposal.normalized_gap = gap / scale;
        proposal.size_ratio = lhs.box.height / std::max(1.0F, rhs.box.height);
        proposal.score = 0.55F + overlap_ratio * 0.25F;
        const float outside = external_gap(
            candidates, proposal.first, proposal.second, proposal.box, proposal.type);
        if (outside != std::numeric_limits<float>::max() &&
            outside < std::max(4.0F, gap * options.isolation_ratio)) {
          continue;
        }
        proposals.push_back(proposal);
        continue;
      }

      if (lhs.kind != UiElementKind::Text || rhs.kind != UiElementKind::Text) {
        continue;
      }
      const auto& top = lhs.box.y <= rhs.box.y ? lhs.box : rhs.box;
      const auto& bottom = lhs.box.y <= rhs.box.y ? rhs.box : lhs.box;
      const float gap = bottom.y - (top.y + top.height);
      const float scale = std::max(1.0F, std::max(lhs.box.height, rhs.box.height));
      const float left_delta = std::abs(lhs.box.x - rhs.box.x);
      const float align_scale = std::max(1.0F, std::min(lhs.box.width, rhs.box.width));
      if (gap < -2.0F || gap / scale > options.max_normalized_gap ||
          left_delta / align_scale > options.alignment_tolerance) {
        continue;
      }
      proposal.type = PatternType::VerticalTextStack;
      proposal.normalized_gap = gap / scale;
      proposal.size_ratio = lhs.box.width / std::max(1.0F, rhs.box.width);
      proposal.score = 0.55F + (1.0F - std::min(1.0F, left_delta / align_scale)) * 0.25F;
      const float outside = external_gap(
          candidates, proposal.first, proposal.second, proposal.box, proposal.type);
      if (outside != std::numeric_limits<float>::max() &&
          outside < std::max(4.0F, gap * options.isolation_ratio)) {
        continue;
      }
      proposals.push_back(proposal);
    }
  }

  std::vector<bool> selected(proposals.size(), false);
  std::vector<bool> member_used(candidates.size(), false);
  for (std::size_t seed = 0; seed < proposals.size(); ++seed) {
    std::vector<std::size_t> cluster;
    for (std::size_t i = seed; i < proposals.size(); ++i) {
      if (similar(proposals[seed], proposals[i])) {
        cluster.push_back(i);
      }
    }
    if (cluster.size() < static_cast<std::size_t>(options.min_repetitions)) {
      continue;
    }
    for (const auto index : cluster) {
      const auto& proposal = proposals[index];
      if (selected[index] || member_used[proposal.first] || member_used[proposal.second]) {
        continue;
      }
      const bool conflicts = std::any_of(result.begin(), result.end(), [&](const auto& group) {
        return partial_overlap(group.box, proposal.box);
      });
      if (conflicts) {
        continue;
      }
      selected[index] = true;
      member_used[proposal.first] = true;
      member_used[proposal.second] = true;
      result.push_back(UiElementCandidate{
        UiElementKind::InferredGroup,
        proposal.box,
        std::min(0.95F, proposal.score + 0.08F * static_cast<float>(cluster.size() - 1)),
        "",
        "",
        "",
        0.0F,
        false,
        proposal.type == PatternType::HorizontalIconText
            ? "boundaryless_repeated_icon_text"
            : "boundaryless_repeated_text_stack"
      });
    }
  }
  return result;
}

} // namespace uiparsercv::pipeline
