#pragma once

#include <string>

namespace uiparsercv::ocr {

struct NormalizedText {
  std::string raw;
  std::string normalized;
};

[[nodiscard]] NormalizedText normalize_text(std::string raw_text);

} // namespace uiparsercv::ocr
