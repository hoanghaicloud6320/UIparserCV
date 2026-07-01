#pragma once

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <string>
#include <vector>

namespace uiparsercv::infer {

struct Tensor {
  std::vector<int64_t> shape;
  std::vector<float> data;
};

class OnnxSession {
public:
  explicit OnnxSession(const std::filesystem::path& model_path);

  [[nodiscard]] const std::string& input_name() const;
  [[nodiscard]] const std::string& output_name() const;
  [[nodiscard]] std::vector<int64_t> input_shape() const;
  [[nodiscard]] std::vector<int64_t> output_shape() const;

  [[nodiscard]] Tensor run(const std::vector<float>& input, const std::vector<int64_t>& shape);

private:
  Ort::Env env_;
  Ort::SessionOptions options_;
  Ort::Session session_{nullptr};
  Ort::AllocatorWithDefaultOptions allocator_;
  std::string input_name_;
  std::string output_name_;
};

} // namespace uiparsercv::infer

