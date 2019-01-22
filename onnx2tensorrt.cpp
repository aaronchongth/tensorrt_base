// STL
#include <iostream>
#include <string>
#include <vector>

// TensorRT
#include "tensorrt_base.hpp"

// OpenCV
#include "opencv2/opencv.hpp"

// CXXOPTS
#include "cxxopts/cxxopts.hpp"
namespace opt = cxxopts;

struct Onnx2tensorrtConfig {
  std::string model_path = "model.onnx";
  std::string output_path = "model.engine";
  uint32_t batch_size = 1;
  uint32_t max_workspace = 1 << 20;
};

Onnx2tensorrtConfig collect_config(const opt::ParseResult &args) {
  Onnx2tensorrtConfig config;
  config.model_path = std::string(args["model-path"].as<std::string>());
  config.output_path = std::string(args["output-path"].as<std::string>());
  config.batch_size = args["batch-size"].as<uint32_t>();
  config.max_workspace = args["max-workspace"].as<uint32_t>();
  return config;
}

void onnx2tensorrt(const Onnx2tensorrtConfig &config) {
  TensorRTModule trt_module(config.model_path, config.batch_size,
                            config.max_workspace);
  std::cout << "<STATUS> Preparing TensorRT engine for saving now."
            << std::endl;
  trt_module.save_engine(config.output_path);
  return;
}

int main(int argc, char **argv) {
  Onnx2tensorrtConfig config;
  opt::Options options("onnx2tensorrt",
                       "Serializes an Onnx model into an Nvidia TensorRT "
                       "engine ready for inference for this machine.");
  options.add_options()(
      "model-path", "Path to ONNX model",
      opt::value<std::string>()->default_value(config.model_path))(
      "output-path", "Path to save Engine file to",
      opt::value<std::string>()->default_value(config.output_path))(
      "batch-size", "Batch size for inference",
      opt::value<uint32_t>()->default_value(std::to_string(config.batch_size)))(
      "max-workspace",
      "Maximum device workspace size in MB for use during optimization",
      opt::value<uint32_t>()->default_value(
          std::to_string(config.max_workspace)))(
      "h, help", "Displays help message and lists arguments",
      opt::value<std::string>()->default_value("false"));

  bool help = false;
  try {
    auto args = options.parse(argc, argv);
    if (args["help"].as<std::string>() == "true")
      help = true;
    else {
      Onnx2tensorrtConfig init_config = collect_config(args);
      onnx2tensorrt(init_config);
    }
  } catch (const opt::OptionException &e) {
    std::cout << "<ERROR> Option exception." << std::endl;
    help = true;
  }

  if (help) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  std::cout << "<STATUS> Onnx2TensorRT done." << std::endl;
  return 0;
}