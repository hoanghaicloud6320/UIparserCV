#include "uiparsercv/pipeline/ui_element.hpp"

#include "uiparsercv/ocr/text_normalizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

float intersection_over_union(const RectF& lhs, const RectF& rhs) {
  const float intersection = intersection_area(lhs, rhs);
  const float total = area(lhs) + area(rhs) - intersection;
  return total > 0.0F ? intersection / total : 0.0F;
}

float omni_overlap(const RectF& lhs, const RectF& rhs) {
  const float inter = intersection_area(lhs, rhs);
  const float lhs_area = area(lhs);
  const float rhs_area = area(rhs);
  const float union_area = lhs_area + rhs_area - inter + 1e-6F;
  const float iou = inter / union_area;
  const float lhs_cover = lhs_area > 0.0F ? inter / lhs_area : 0.0F;
  const float rhs_cover = rhs_area > 0.0F ? inter / rhs_area : 0.0F;
  return std::max({iou, lhs_cover, rhs_cover});
}

bool is_inside(const RectF& inner, const RectF& outer) {
  const float inner_area = area(inner);
  if (inner_area <= 0.0F) {
    return false;
  }
  return intersection_area(inner, outer) / inner_area > 0.80F;
}

} // namespace

CombinedCandidates build_model_ocr_candidates(
    const std::vector<detect::UitagDetection>& model_detections,
    const std::vector<Detection>& legacy_icons,
    const std::vector<TextRegion>& text_regions,
    const std::vector<TextRecognition>& recognized_text,
    ModelOcrMergeOptions options) {
  if (text_regions.size() != recognized_text.size()) {
    throw std::runtime_error("text region count does not match recognition count");
  }

  CombinedCandidates result;
  result.candidates.reserve(
      model_detections.size() + legacy_icons.size() + text_regions.size());

  for (const auto& detection : model_detections) {
    result.candidates.push_back(UiElementCandidate{
        UiElementKind::ModelProposal,
        {detection.box.x, detection.box.y, detection.box.width, detection.box.height},
        detection.score,
        {}, {}, {}, 0.0F,
        detection.class_id == 0 || detection.class_id == 1 || detection.class_id == 2 ||
            detection.class_id == 3,
        "uitag_yolo11",
        std::string(detect::UitagDetector::class_name(detection.class_id))});
  }

  // OmniParser is supporting evidence only: keep boxes that add spatial coverage
  // instead of duplicating an existing UITag proposal.
  for (const auto& icon : legacy_icons) {
    bool novel = true;
    for (const auto& candidate : result.candidates) {
      if (intersection_over_union(icon.box, candidate.box) >= options.legacy_icon_novelty_iou ||
          is_inside(icon.box, candidate.box)) {
        novel = false;
        break;
      }
    }
    if (!novel) continue;
    result.candidates.push_back(UiElementCandidate{
        UiElementKind::Icon, icon.box, icon.score, {}, {}, {}, 0.0F, true,
        "omniparser_icon_support", icon.label});
  }

  const std::size_t proposal_count = result.candidates.size();
  for (std::size_t i = 0; i < text_regions.size(); ++i) {
    const auto text = ocr::normalize_text(recognized_text[i].text);
    result.candidates.push_back(UiElementCandidate{
        UiElementKind::Text, text_regions[i].box, text_regions[i].score,
        text.normalized, text.raw, text.normalized, recognized_text[i].confidence,
        false, "ocr_box_and_content", "Text"});
  }

  for (std::size_t text_offset = 0; text_offset < text_regions.size(); ++text_offset) {
    const std::size_t text_index = proposal_count + text_offset;
    const RectF& text_box = result.candidates[text_index].box;
    std::size_t best = proposal_count;
    float best_area = 0.0F;
    float best_cover = 0.0F;
    for (std::size_t proposal = 0; proposal < proposal_count; ++proposal) {
      const auto& proposal_box = result.candidates[proposal].box;
      const float text_area = area(text_box);
      const float cover = text_area > 0.0F
          ? intersection_area(text_box, proposal_box) / text_area
          : 0.0F;
      if (cover < options.text_containment_threshold) continue;
      const float proposal_area = area(proposal_box);
      if (best == proposal_count || proposal_area < best_area) {
        best = proposal;
        best_area = proposal_area;
        best_cover = cover;
      }
    }
    if (best == proposal_count) continue;

    const float iou = intersection_over_union(result.candidates[best].box, text_box);
    result.associations.push_back(CandidateAssociation{
        best, text_index, std::max(best_cover, iou),
        iou >= options.near_identical_iou ? "same_observation" : "text_in_control"});
  }

  return result;
}

std::vector<UiElementCandidate> build_candidates(
    const std::vector<Detection>& icons,
    const std::vector<TextRegion>& text_regions,
    const std::vector<TextRecognition>& recognized_text,
    CandidateMergeOptions options) {
  if (text_regions.size() != recognized_text.size()) {
    throw std::runtime_error("text region count does not match recognition count");
  }

  std::vector<UiElementCandidate> text_candidates;
  text_candidates.reserve(text_regions.size());
  for (std::size_t i = 0; i < text_regions.size(); ++i) {
    const auto text = ocr::normalize_text(recognized_text[i].text);
    text_candidates.push_back(UiElementCandidate{
      UiElementKind::Text,
      text_regions[i].box,
      text_regions[i].score,
      text.normalized,
      text.raw,
      text.normalized,
      recognized_text[i].confidence,
      false,
      "box_ocr_content_ocr",
      "Text"
    });
  }

  std::vector<UiElementCandidate> result = text_candidates;
  std::vector<bool> text_removed(text_candidates.size(), false);

  for (std::size_t i = 0; i < icons.size(); ++i) {
    const RectF& icon_box = icons[i].box;
    bool keep_icon = true;

    for (std::size_t j = 0; j < icons.size(); ++j) {
      if (i == j) {
        continue;
      }
      if (omni_overlap(icon_box, icons[j].box) > options.overlap_threshold &&
          area(icon_box) > area(icons[j].box)) {
        keep_icon = false;
        break;
      }
    }

    if (!keep_icon) {
      continue;
    }

    std::string absorbed_text;
    std::string absorbed_raw_text;
    float absorbed_confidence = 0.0F;
    int absorbed_count = 0;
    bool icon_inside_text = false;

    for (std::size_t t = 0; t < text_candidates.size(); ++t) {
      const RectF& text_box = text_candidates[t].box;
      if (is_inside(text_box, icon_box)) {
        text_removed[t] = !options.preserve_text_inside_icons;
        if (!text_candidates[t].text.empty()) {
          if (!absorbed_text.empty()) {
            absorbed_text += ' ';
          }
          if (!absorbed_raw_text.empty()) {
            absorbed_raw_text += ' ';
          }
          absorbed_text += text_candidates[t].text;
          absorbed_raw_text += text_candidates[t].raw_text;
          absorbed_confidence += text_candidates[t].text_confidence;
          ++absorbed_count;
        }
      } else if (is_inside(icon_box, text_box)) {
        icon_inside_text = true;
        break;
      }
    }

    if (icon_inside_text) {
      continue;
    }

    result.push_back(UiElementCandidate{
      UiElementKind::Icon,
      icon_box,
      icons[i].score,
      absorbed_text,
      absorbed_raw_text,
      absorbed_text,
      absorbed_count > 0 ? absorbed_confidence / static_cast<float>(absorbed_count) : 0.0F,
      true,
      absorbed_text.empty() ? "box_yolo_content_yolo" : "box_yolo_content_ocr",
      icons[i].label
    });
  }

  result.erase(
      std::remove_if(
          result.begin(),
          result.end(),
          [&](const UiElementCandidate& candidate) {
            if (candidate.kind != UiElementKind::Text) {
              return false;
            }
            for (std::size_t i = 0; i < text_candidates.size(); ++i) {
              if (text_removed[i] &&
                  candidate.box.x == text_candidates[i].box.x &&
                  candidate.box.y == text_candidates[i].box.y &&
                  candidate.box.width == text_candidates[i].box.width &&
                  candidate.box.height == text_candidates[i].box.height) {
                return true;
              }
            }
            return false;
          }),
      result.end());

  std::stable_sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.text.empty() != rhs.text.empty()) {
      return !lhs.text.empty();
    }
    if (std::abs(lhs.box.y - rhs.box.y) < 10.0F) {
      return lhs.box.x < rhs.box.x;
    }
    return lhs.box.y < rhs.box.y;
  });

  return result;
}

} // namespace uiparsercv::pipeline
