#pragma once

#include "uiparsercv/pipeline/ui_element.hpp"
#include "uiparsercv/tree/box_tree.hpp"

#include <vector>

namespace uiparsercv::pipeline {

[[nodiscard]] tree::TreeNode build_grouped_ui_tree(
    const std::vector<UiElementCandidate>& candidates,
    int image_width,
    int image_height);

} // namespace uiparsercv::pipeline
