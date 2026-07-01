#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace {

void print_shape(const std::vector<int64_t>& shape) {
  std::cout << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << shape[i];
  }
  std::cout << ']';
}

void print_io(
    const Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    bool input) {
  const std::size_t count = input ? session.GetInputCount() : session.GetOutputCount();
  std::cout << (input ? "inputs" : "outputs") << ": " << count << '\n';

  for (std::size_t i = 0; i < count; ++i) {
    auto name = input
        ? session.GetInputNameAllocated(i, allocator)
        : session.GetOutputNameAllocated(i, allocator);
    auto type_info = input
        ? session.GetInputTypeInfo(i)
        : session.GetOutputTypeInfo(i);

    std::cout << "  [" << i << "] " << name.get();
    if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
      const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      std::cout << " shape=";
      print_shape(tensor_info.GetShape());
      std::cout << " elem_type=" << tensor_info.GetElementType();
    }
    std::cout << '\n';
  }
}

std::string lower_copy(std::string value) {
  for (char& ch : value) {
    if (ch >= 'A' && ch <= 'Z') {
      ch = static_cast<char>(ch - 'A' + 'a');
    }
  }
  return value;
}

std::vector<int64_t> dummy_shape_for(
    const Ort::Session& session,
    const std::filesystem::path& model_path) {
  const std::string path = lower_copy(model_path.string());

  auto type_info = session.GetInputTypeInfo(0);
  std::vector<int64_t> shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
  if (shape.empty()) {
    return {1};
  }

  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] > 0) {
      continue;
    }

    if (i == 0) {
      shape[i] = 1;
    } else if (i == 1) {
      shape[i] = 3;
    } else if (path.find("rec") != std::string::npos && i == 2) {
      shape[i] = 48;
    } else if (path.find("rec") != std::string::npos && i == 3) {
      shape[i] = 160;
    } else {
      shape[i] = 32;
    }
  }

  return shape;
}

std::size_t element_count(const std::vector<int64_t>& shape) {
  return static_cast<std::size_t>(std::accumulate(
      shape.begin(),
      shape.end(),
      int64_t{1},
      [](int64_t lhs, int64_t rhs) { return lhs * rhs; }));
}

void run_dummy_inference(Ort::Session& session, const std::filesystem::path& model_path) {
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator,
      OrtMemTypeDefault);

  std::vector<Ort::AllocatedStringPtr> input_names_storage;
  std::vector<Ort::AllocatedStringPtr> output_names_storage;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
  std::vector<Ort::Value> input_tensors;
  std::vector<std::vector<float>> input_buffers;

  const std::vector<int64_t> shape = dummy_shape_for(session, model_path);

  for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
    input_names_storage.push_back(session.GetInputNameAllocated(i, allocator));
    input_names.push_back(input_names_storage.back().get());

    input_buffers.emplace_back(element_count(shape), 0.0F);
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info,
        input_buffers.back().data(),
        input_buffers.back().size(),
        shape.data(),
        shape.size()));
  }

  for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
    output_names_storage.push_back(session.GetOutputNameAllocated(i, allocator));
    output_names.push_back(output_names_storage.back().get());
  }

  std::vector<Ort::Value> outputs = session.Run(
      Ort::RunOptions{nullptr},
      input_names.data(),
      input_tensors.data(),
      input_tensors.size(),
      output_names.data(),
      output_names.size());

  std::cout << "dummy_run: ok input_shape=";
  print_shape(shape);
  std::cout << '\n';

  for (std::size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i].IsTensor()) {
      auto info = outputs[i].GetTensorTypeAndShapeInfo();
      std::cout << "  output[" << i << "] shape=";
      print_shape(info.GetShape());
      std::cout << '\n';
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "uiparsercv-onnx-probe");
    std::cout << "onnxruntime: " << OrtGetApiBase()->GetVersionString() << '\n';

    if (argc == 1) {
      std::cout << "usage: onnxruntime_probe [--run-dummy] <model.onnx> [more-models.onnx ...]\n";
      return 0;
    }

    int first_model_arg = 1;
    bool run_dummy = false;
    if (std::string_view(argv[1]) == "--run-dummy") {
      run_dummy = true;
      first_model_arg = 2;
    }

    if (first_model_arg >= argc) {
      std::cerr << "usage: onnxruntime_probe [--run-dummy] <model.onnx> [more-models.onnx ...]\n";
      return 2;
    }

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    Ort::AllocatorWithDefaultOptions allocator;

    for (int arg = first_model_arg; arg < argc; ++arg) {
      const std::filesystem::path model_path(argv[arg]);
      if (!std::filesystem::is_regular_file(model_path)) {
        std::cerr << "model not found: " << model_path.string() << '\n';
        return 1;
      }

      Ort::Session session(env, model_path.c_str(), session_options);

      std::cout << "model: " << model_path.string() << '\n';
      print_io(session, allocator, true);
      print_io(session, allocator, false);
      if (run_dummy) {
        run_dummy_inference(session, model_path);
      }
    }

    return 0;
  } catch (const Ort::Exception& error) {
    std::cerr << "onnxruntime error: " << error.what() << '\n';
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
