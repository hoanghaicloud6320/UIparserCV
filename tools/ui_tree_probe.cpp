#include "uiparsercv/tree/box_tree.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

const uiparsercv::tree::TreeNode* find_by_id(
    const uiparsercv::tree::TreeNode& node,
    int id) {
  if (node.box.id == id) {
    return &node;
  }

  for (const auto& child : node.children) {
    if (const auto* found = find_by_id(child, id)) {
      return found;
    }
  }

  return nullptr;
}

bool has_direct_child(const uiparsercv::tree::TreeNode& node, int id) {
  for (const auto& child : node.children) {
    if (child.box.id == id) {
      return true;
    }
  }
  return false;
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  using namespace uiparsercv::tree;

  std::vector<Box> boxes{
    Box{1, Rect{0, 0, 320, 640}, "screen"},
    Box{2, Rect{16, 16, 288, 56}, "top bar"},
    Box{3, Rect{24, 96, 272, 160}, "card"},
    Box{4, Rect{40, 128, 96, 32}, "title text"},
    Box{5, Rect{40, 176, 224, 48}, "button"},
    Box{6, Rect{24, 288, 272, 80}, "footer row"},
    Box{7, Rect{48, 184, 80, 24}, "button label"},
    Box{8, Rect{24, 400, 120, 40}, "equal box a"},
    Box{9, Rect{24, 400, 120, 40}, "equal box b"},
  };

  const TreeNode root = build_containment_tree(boxes);
  const TreeNode* screen = find_by_id(root, 1);
  const TreeNode* card = find_by_id(root, 3);
  const TreeNode* button = find_by_id(root, 5);
  const TreeNode* title = find_by_id(root, 4);
  const TreeNode* label = find_by_id(root, 7);

  require(screen != nullptr, "screen node was not created");
  require(card != nullptr, "card node was not created");
  require(button != nullptr, "button node was not created");
  require(title != nullptr, "title node was not created");
  require(label != nullptr, "button label node was not created");
  require(card->children.size() == 2, "card should contain title and button");
  require(has_direct_child(card->children[1], 7) || has_direct_child(*button, 7),
          "button should directly contain its label");
  require(!has_direct_child(*find_by_id(root, 8), 9),
          "equal boxes must not contain each other");
  require(!has_direct_child(*find_by_id(root, 9), 8),
          "equal boxes must remain siblings");
  require(contains(card->box.rect, button->box.rect), "card should contain button");
  require(contains(card->box.rect, title->box.rect), "card should contain title");

  write_json(std::cout, root);
  std::cout << '\n';

  return 0;
}
