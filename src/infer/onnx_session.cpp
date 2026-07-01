#include "uiparsercv/infer/onnx_session.hpp"

#include <numeric>
#include <stdexcept>

namespace uiparsercv::infer {
namespace {

std::size_t element_count(const std::vector<int64_t>& shape) {
  return static_cast<std::size_t>(std::accumulate(
      shape.begin(),
      shape.end(),
      int64_t{1},
      [](int64_t lhs, int64_t rhs) { return lhs * rhs; }));
}

} // namespace

OnnxSession::OnnxSession(const std::filesystem::path& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "uiparsercv"),
      options_{} {
  if (!std::filesystem::is_regular_file(model_path)) {
    throw std::runtime_error("ONNX model not found: " + model_path.string());
  }

  options_.SetIntraOpNumThreads(1);
  options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
  session_ = Ort::Session(env_, model_path.c_str(), options_);

  input_name_ = session_.GetInputNameAllocated(0, allocator_).get();
  output_name_ = session_.GetOutputNameAllocated(0, allocator_).get();
}

const std::string& OnnxSession::input_name() const {
  return input_name_;
}

const std::string& OnnxSession::output_name() const {
  return output_name_;
}

std::vector<int64_t> OnnxSession::input_shape() const {
  return session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
}

std::vector<int64_t> OnnxSession::output_shape() const {
  return session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
}

Tensor OnnxSession::run(const std::vector<float>& input, const std::vector<int64_t>& shape) {
  if (input.size() != element_count(shape)) {
    throw std::runtime_error("input tensor size does not match shape");
  }

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator,
      OrtMemTypeDefault);

  auto tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      const_cast<float*>(input.data()),
      input.size(),
      shape.data(),
      shape.size());

  const char* input_names[] = {input_name_.c_str()};
  const char* output_names[] = {output_name_.c_str()};

  std::vector<Ort::Value> outputs = session_.Run(
      Ort::RunOptions{nullptr},
      input_names,
      &tensor,
      1,
      output_names,
      1);

  auto info = outputs[0].GetTensorTypeAndShapeInfo();
  Tensor result;
  result.shape = info.GetShape();
  const std::size_t output_count = info.GetElementCount();
  const float* output_data = outputs[0].GetTensorData<float>();
  result.data.assign(output_data, output_data + output_count);
  return result;
}

} // namespace uiparsercv::infer

