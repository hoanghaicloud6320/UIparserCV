#include <filesystem>
#include <iostream>
#include <string_view>

namespace {

bool check_file(std::string_view name, const std::filesystem::path& path) {
  const bool exists = std::filesystem::is_regular_file(path);
  std::cout << name << ": " << path.string() << " -> "
            << (exists ? "found" : "missing") << '\n';
  return exists;
}

} // namespace

int main() {
  bool ok = true;
  ok = check_file("icon model", UIPARSERCV_ICON_MODEL) && ok;
  ok = check_file("ocr detection model", UIPARSERCV_OCR_DET_MODEL) && ok;
  ok = check_file("ocr recognition model", UIPARSERCV_OCR_REC_MODEL) && ok;

  if (!ok) {
    std::cerr << "One or more configured model files are missing.\n";
    return 1;
  }

  return 0;
}

