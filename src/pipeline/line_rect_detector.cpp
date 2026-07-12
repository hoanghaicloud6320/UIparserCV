#include "uiparsercv/pipeline/line_rect_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace uiparsercv::pipeline {
namespace {

struct AxisLine {
  int coordinate{};
  int start{};
  int end{};
};

struct DividerCell {
  RectF box{};
  float score{};
};

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

float iou(const RectF& lhs, const RectF& rhs) {
  const float intersection = intersection_area(lhs, rhs);
  return intersection / (area(lhs) + area(rhs) - intersection + 1e-6F);
}

bool strict_contains(const RectF& outer, const RectF& inner) {
  return area(outer) > area(inner) &&
         outer.x <= inner.x && outer.y <= inner.y &&
         outer.x + outer.width >= inner.x + inner.width &&
         outer.y + outer.height >= inner.y + inner.height;
}

bool partial_overlap(const RectF& lhs, const RectF& rhs) {
  const float intersection = intersection_area(lhs, rhs);
  return intersection > 0.0F && !strict_contains(lhs, rhs) && !strict_contains(rhs, lhs);
}

bool conflicts_with_visual_container(
    const RectF& box,
    const std::vector<UiElementCandidate>& existing) {
  for (const auto& candidate : existing) {
    if (candidate.kind != UiElementKind::VisualContainer) {
      continue;
    }
    if (strict_contains(box, candidate.box) || partial_overlap(box, candidate.box)) {
      return true;
    }
  }
  return false;
}

int atomic_content_count(
    const RectF& box,
    const std::vector<UiElementCandidate>& existing) {
  int count = 0;
  for (const auto& candidate : existing) {
    if (candidate.kind == UiElementKind::VisualContainer ||
        candidate.kind == UiElementKind::InferredGroup) {
      continue;
    }
    const float center_x = candidate.box.x + candidate.box.width * 0.5F;
    const float center_y = candidate.box.y + candidate.box.height * 0.5F;
    if (center_x >= box.x && center_x <= box.x + box.width &&
        center_y >= box.y && center_y <= box.y + box.height) {
      ++count;
    }
  }
  return count;
}

float interval_support(const AxisLine& line, int begin, int end) {
  const int overlap = std::max(0, std::min(line.end, end) - std::max(line.start, begin));
  return static_cast<float>(overlap) / static_cast<float>(std::max(1, end - begin));
}

std::vector<AxisLine> merge_axis_lines(
    std::vector<AxisLine> lines,
    int coordinate_tolerance,
    int merge_gap,
    int max_lines) {
  std::sort(lines.begin(), lines.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.coordinate != rhs.coordinate) {
      return lhs.coordinate < rhs.coordinate;
    }
    return lhs.start < rhs.start;
  });

  std::vector<AxisLine> merged;
  for (const auto& line : lines) {
    int best = -1;
    int best_distance = coordinate_tolerance + 1;
    for (std::size_t i = 0; i < merged.size(); ++i) {
      const int distance = std::abs(merged[i].coordinate - line.coordinate);
      const bool intervals_touch =
          line.start <= merged[i].end + merge_gap && merged[i].start <= line.end + merge_gap;
      if (distance <= coordinate_tolerance && intervals_touch && distance < best_distance) {
        best = static_cast<int>(i);
        best_distance = distance;
      }
    }
    if (best < 0) {
      merged.push_back(line);
      continue;
    }
    auto& target = merged[static_cast<std::size_t>(best)];
    target.coordinate = (target.coordinate + line.coordinate) / 2;
    target.start = std::min(target.start, line.start);
    target.end = std::max(target.end, line.end);
  }

  std::stable_sort(merged.begin(), merged.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.end - lhs.start > rhs.end - rhs.start;
  });
  if (merged.size() > static_cast<std::size_t>(max_lines)) {
    merged.resize(static_cast<std::size_t>(max_lines));
  }
  std::sort(merged.begin(), merged.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.coordinate < rhs.coordinate;
  });
  return merged;
}

bool duplicate(
    const RectF& box,
    const std::vector<UiElementCandidate>& existing,
    const std::vector<UiElementCandidate>& proposals,
    float threshold) {
  const auto near = [&](const UiElementCandidate& candidate) {
    return iou(box, candidate.box) >= threshold;
  };
  return std::any_of(existing.begin(), existing.end(), near) ||
         std::any_of(proposals.begin(), proposals.end(), near);
}

float edge_density(const cv::Mat& edges, const cv::Rect& rect) {
  const int inset_x = std::max(2, rect.width / 20);
  const int inset_y = std::max(2, rect.height / 20);
  const cv::Rect inner{
    rect.x + inset_x,
    rect.y + inset_y,
    rect.width - inset_x * 2,
    rect.height - inset_y * 2
  };
  if (inner.width <= 0 || inner.height <= 0) {
    return 1.0F;
  }
  return static_cast<float>(cv::countNonZero(edges(inner))) /
         static_cast<float>(inner.area());
}

std::vector<UiElementCandidate> repeated_vertical_divider_cells(
    const std::vector<AxisLine>& vertical,
    const std::vector<UiElementCandidate>& existing,
    int image_width,
    const LineRectOptions& options) {
  std::vector<DividerCell> cells;
  for (std::size_t left = 0; left < vertical.size(); ++left) {
    for (std::size_t right = left + 1; right < vertical.size(); ++right) {
      const int width = vertical[right].coordinate - vertical[left].coordinate;
      if (width < options.min_side || width > image_width / 2) {
        continue;
      }
      const int top = std::max(vertical[left].start, vertical[right].start);
      const int bottom = std::min(vertical[left].end, vertical[right].end);
      const int height = bottom - top;
      if (height < options.min_side * 2) {
        continue;
      }
      const float left_support = interval_support(vertical[left], top, bottom);
      const float right_support = interval_support(vertical[right], top, bottom);
      const RectF box{
        static_cast<float>(vertical[left].coordinate),
        static_cast<float>(top),
        static_cast<float>(width),
        static_cast<float>(height)
      };
      if (atomic_content_count(box, existing) < 2) {
        continue;
      }
      cells.push_back(DividerCell{box, (left_support + right_support) * 0.5F});
    }
  }

  std::vector<UiElementCandidate> selected;
  std::vector<bool> used(cells.size(), false);
  for (std::size_t seed = 0; seed < cells.size(); ++seed) {
    if (used[seed]) {
      continue;
    }
    std::vector<std::size_t> family;
    for (std::size_t i = 0; i < cells.size(); ++i) {
      const float width_ratio = cells[i].box.width / std::max(1.0F, cells[seed].box.width);
      const float height_ratio = cells[i].box.height / std::max(1.0F, cells[seed].box.height);
      const float y_tolerance = std::max(10.0F, cells[seed].box.height * 0.08F);
      if (width_ratio >= 0.78F && width_ratio <= 1.28F &&
          height_ratio >= 0.82F && height_ratio <= 1.22F &&
          std::abs(cells[i].box.y - cells[seed].box.y) <= y_tolerance &&
          std::abs((cells[i].box.y + cells[i].box.height) -
                   (cells[seed].box.y + cells[seed].box.height)) <= y_tolerance) {
        family.push_back(i);
      }
    }
    std::sort(family.begin(), family.end(), [&](std::size_t lhs, std::size_t rhs) {
      return cells[lhs].box.x < cells[rhs].box.x;
    });

    std::vector<std::size_t> chain;
    for (const auto index : family) {
      if (chain.empty()) {
        chain.push_back(index);
        continue;
      }
      const auto& previous = cells[chain.back()].box;
      const float gap = cells[index].box.x - (previous.x + previous.width);
      if (gap >= -options.coordinate_tolerance * 2.0F &&
          gap <= std::max(24.0F, previous.width * 0.22F)) {
        chain.push_back(index);
      } else if (chain.size() < 3) {
        chain.clear();
        chain.push_back(index);
      }
    }
    if (chain.size() < 3) {
      continue;
    }
    for (const auto index : chain) {
      if (used[index] || duplicate(cells[index].box, existing, selected, options.duplicate_iou)) {
        continue;
      }
      used[index] = true;
      selected.push_back(UiElementCandidate{
        UiElementKind::VisualContainer,
        cells[index].box,
        std::min(0.95F, 0.72F + cells[index].score * 0.18F),
        "",
        "",
        "",
        0.0F,
        false,
        "line_grid_cell_vertical_dividers",
        ""
      });
    }
  }
  return selected;
}

} // namespace

std::vector<UiElementCandidate> detect_line_rects(
    const cv::Mat& bgr_image,
    const std::vector<UiElementCandidate>& existing,
    LineRectOptions options) {
  std::vector<UiElementCandidate> proposals;
  if (!options.enabled || bgr_image.empty()) {
    return proposals;
  }

  cv::Mat smooth;
  cv::bilateralFilter(bgr_image, smooth, 7, 30.0, 30.0);
  std::vector<cv::Mat> channels;
  cv::split(smooth, channels);
  cv::Mat edges = cv::Mat::zeros(bgr_image.size(), CV_8UC1);
  for (const auto& channel : channels) {
    cv::Mat channel_edges;
    cv::Canny(channel, channel_edges, 32.0, 96.0);
    cv::bitwise_or(edges, channel_edges, edges);
  }

  std::vector<cv::Vec4i> raw_lines;
  const int min_dimension = std::min(bgr_image.cols, bgr_image.rows);
  cv::HoughLinesP(
      edges,
      raw_lines,
      1.0,
      CV_PI / 180.0,
      24,
      std::max(20.0, static_cast<double>(min_dimension) * 0.045),
      static_cast<double>(options.merge_gap));

  std::vector<AxisLine> horizontal_raw;
  std::vector<AxisLine> vertical_raw;
  for (const auto& line : raw_lines) {
    const int dx = line[2] - line[0];
    const int dy = line[3] - line[1];
    if (std::abs(dx) >= std::abs(dy) * 5) {
      horizontal_raw.push_back(AxisLine{
        (line[1] + line[3]) / 2,
        std::min(line[0], line[2]),
        std::max(line[0], line[2])
      });
    } else if (std::abs(dy) >= std::abs(dx) * 5) {
      vertical_raw.push_back(AxisLine{
        (line[0] + line[2]) / 2,
        std::min(line[1], line[3]),
        std::max(line[1], line[3])
      });
    }
  }

  const auto horizontal = merge_axis_lines(
      std::move(horizontal_raw),
      options.coordinate_tolerance,
      options.merge_gap,
      options.max_axis_lines);
  const auto vertical = merge_axis_lines(
      std::move(vertical_raw),
      options.coordinate_tolerance,
      options.merge_gap,
      options.max_axis_lines);

  std::vector<UiElementCandidate> grid_cells = repeated_vertical_divider_cells(
      vertical, existing, bgr_image.cols, options);

  const float image_area = static_cast<float>(bgr_image.cols * bgr_image.rows);
  for (std::size_t top_index = 0; top_index < horizontal.size(); ++top_index) {
    for (std::size_t bottom_index = top_index + 1; bottom_index < horizontal.size(); ++bottom_index) {
      const int top = horizontal[top_index].coordinate;
      const int bottom = horizontal[bottom_index].coordinate;
      const int height = bottom - top;
      if (height < options.min_side) {
        continue;
      }

      std::vector<std::size_t> spanning_verticals;
      for (std::size_t i = 0; i < vertical.size(); ++i) {
        if (interval_support(vertical[i], top, bottom) >= options.min_side_support) {
          spanning_verticals.push_back(i);
        }
      }

      for (std::size_t left_pos = 0; left_pos < spanning_verticals.size(); ++left_pos) {
        for (std::size_t right_pos = left_pos + 1; right_pos < spanning_verticals.size(); ++right_pos) {
          const auto& left_line = vertical[spanning_verticals[left_pos]];
          const auto& right_line = vertical[spanning_verticals[right_pos]];
          const int left = left_line.coordinate;
          const int right = right_line.coordinate;
          const int width = right - left;
          if (width < options.min_side) {
            continue;
          }

          const float top_support = interval_support(horizontal[top_index], left, right);
          const float bottom_support = interval_support(horizontal[bottom_index], left, right);
          const float left_support = interval_support(left_line, top, bottom);
          const float right_support = interval_support(right_line, top, bottom);
          if (std::min({top_support, bottom_support, left_support, right_support}) <
              options.min_side_support) {
            continue;
          }

          const float area_ratio = static_cast<float>(width * height) / image_area;
          if (area_ratio < options.min_area_ratio || area_ratio > options.max_area_ratio) {
            continue;
          }

          const cv::Rect rect{left, top, width, height};
          if (rect.x < 0 || rect.y < 0 || rect.x + rect.width > bgr_image.cols ||
              rect.y + rect.height > bgr_image.rows ||
              edge_density(edges, rect) > options.max_interior_edge_density) {
            continue;
          }

          const RectF box{
            static_cast<float>(left),
            static_cast<float>(top),
            static_cast<float>(width),
            static_cast<float>(height)
          };
          if (duplicate(box, existing, proposals, options.duplicate_iou)) {
            continue;
          }

          const float score =
              (top_support + bottom_support + left_support + right_support) * 0.25F;
          proposals.push_back(UiElementCandidate{
            UiElementKind::VisualContainer,
            box,
            score,
            "",
            "",
            "",
            0.0F,
            false,
            "line_rect_4sides",
            ""
          });
        }
      }
    }
  }

  std::stable_sort(proposals.begin(), proposals.end(), [](const auto& lhs, const auto& rhs) {
    if (std::abs(lhs.detection_score - rhs.detection_score) > 0.02F) {
      return lhs.detection_score > rhs.detection_score;
    }
    return area(lhs.box) < area(rhs.box);
  });
  if (proposals.size() > static_cast<std::size_t>(options.max_proposals)) {
    proposals.resize(static_cast<std::size_t>(options.max_proposals));
  }

  std::vector<UiElementCandidate> selected = std::move(grid_cells);
  for (std::size_t i = 0; i < proposals.size(); ++i) {
    const auto& proposal = proposals[i];
    if (atomic_content_count(proposal.box, existing) < 2 ||
        conflicts_with_visual_container(proposal.box, existing)) {
      continue;
    }
    const bool contains_smaller = std::any_of(
        proposals.begin(), proposals.end(), [&](const auto& other) {
          return strict_contains(proposal.box, other.box) &&
                 atomic_content_count(other.box, existing) >= 2;
        });
    if (contains_smaller) {
      continue;
    }
    const bool conflicts = std::any_of(selected.begin(), selected.end(), [&](const auto& other) {
      return partial_overlap(proposal.box, other.box);
    });
    if (!conflicts) {
      selected.push_back(proposal);
    }
  }
  return selected;
}

} // namespace uiparsercv::pipeline
