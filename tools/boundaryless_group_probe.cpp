#include "uiparsercv/pipeline/boundaryless_group_detector.hpp"

#include <stdexcept>
#include <vector>

namespace {

using uiparsercv::RectF;
using uiparsercv::pipeline::UiElementCandidate;
using uiparsercv::pipeline::UiElementKind;

UiElementCandidate candidate(UiElementKind kind, RectF box, const char* text = "") {
  return UiElementCandidate{kind, box, 1.0F, text, text, text, 1.0F, false, "probe"};
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  using uiparsercv::pipeline::detect_boundaryless_groups;

  const std::vector<UiElementCandidate> repeated_stacks{
    candidate(UiElementKind::Text, {20, 20, 100, 20}, "Title A"),
    candidate(UiElementKind::Text, {20, 44, 160, 18}, "Description A"),
    candidate(UiElementKind::Text, {220, 20, 100, 20}, "Title B"),
    candidate(UiElementKind::Text, {220, 44, 160, 18}, "Description B")
  };
  const auto groups = detect_boundaryless_groups(repeated_stacks, 420, 120);
  require(groups.size() == 2, "two repeated text stacks should form two groups");
  require(groups[0].kind == UiElementKind::InferredGroup, "proposal kind should be inferred group");

  std::vector<UiElementCandidate> already_contained = repeated_stacks;
  already_contained.push_back(candidate(UiElementKind::Icon, {12, 12, 176, 58}));
  already_contained.push_back(candidate(UiElementKind::Icon, {212, 12, 176, 58}));
  const auto redundant = detect_boundaryless_groups(already_contained, 420, 120);
  require(redundant.empty(), "groups already explained by containers must be rejected");

  return 0;
}
