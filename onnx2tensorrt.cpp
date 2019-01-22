// STL
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// TensorRT
#include "tensorrt_base.hpp"

// OpenCV
#include "opencv2/opencv.hpp"

int main(int argc, char **argv) {
  std::string onnx_path = "/home/aaron/Data/onnx_models/resnet50v1.onnx";
  int batch_size = 1;
  int max_workspace_size = 1 << 20;
  TensorRTModule trt_module(onnx_path, batch_size, max_workspace_size);

  // std::string engine_path = "/home/aaron/Data/engines/resnet50v1.engine";
  // TensorRTModule trt_module(engine_path);

  // prepare data
  cv::Mat img =
      cv::imread("/home/aaron/projects/ros2_tensorrt/data/cat_224.jpg",
                 CV_LOAD_IMAGE_COLOR);
  cv::Mat float_mat;
  img.convertTo(float_mat, CV_32FC3, 1.0 / 255.0);
  cv::Mat resized_mat;
  cv::resize(float_mat, resized_mat, cv::Size(224, 224), 0, 0,
             cv::INTER_LINEAR);

  // splitting into BGR channels, then normalize
  std::vector<cv::Mat> channels;
  cv::split(resized_mat, channels);
  // channels[0] = channels[0] - BLUE_CHANNEL_MEAN;
  // channels[0] = channels[0] / BLUE_CHANNEL_STD;
  // channels[1] = channels[1] - GREEN_CHANNEL_MEAN;
  // channels[1] = channels[1] / GREEN_CHANNEL_STD;
  // channels[2] = channels[2] - RED_CHANNEL_MEAN;
  // channels[2] = channels[2] / RED_CHANNEL_STD;

  // vectorize
  cv::Mat vect_mat;
  vect_mat.push_back(channels[2]);
  vect_mat.push_back(channels[1]);
  vect_mat.push_back(channels[0]);
  vect_mat = vect_mat.reshape(1, 1);

  std::vector<float> tmp_input;
  tmp_input.assign((float *)vect_mat.datastart, (float *)vect_mat.dataend);

  // cv::Mat test_img;

  // prepare dummy data
  std::vector<std::vector<float>> dummy_input;
  dummy_input.push_back(tmp_input);

  // int n_inputs = trt_module.get_n_inputs();
  // for (int i = 0; i < n_inputs; i++) {
  //   auto input_dim = trt_module.get_input_dimensions(i);
  //   std::cout << "Input size is: ";
  //   for (int j = 0; j < input_dim.nbDims; j++)
  //     std::cout << input_dim.d[j] << " ";
  //   std::cout << std::endl;

  //   int64_t curr_input_volume =
  //       tensorrt_common::volume(trt_module.get_input_dimensions(i));
  //   std::cout << "Allocating " << curr_input_volume << " for dummy input."
  //             << std::endl;
  //   std::vector<float> tmp_input(static_cast<int>(curr_input_volume), 1);
  //   dummy_input.push_back(tmp_input);
  // }

  // inference
  if (!trt_module.inference(dummy_input))
    std::cout << "Inference failed." << std::endl;
  int n_outputs = trt_module.get_n_outputs();
  for (int i = 0; i < n_outputs; i++) {
    auto output_dim = trt_module.get_output_dimensions(i);
    std::cout << "Output size is: ";
    for (int j = 0; j < output_dim.nbDims; j++)
      std::cout << output_dim.d[j] << " ";
    std::cout << std::endl;
  }
  // try to get topK results
  std::vector<float> output = trt_module.get_output(0);
  auto top = tensorrt_common::topK<float>(output, 1);
  std::cout << "Classification is: " << top[0] << std::endl;

  // // timing
  // auto t0 = std::chrono::system_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
  //     std::chrono::system_clock::now() - t0);
  // for (int i = 0; i < 100; i++) {
  //   if (!trt_module.inference(dummy_input))
  //     std::cout << "Inference failed." << std::endl;

  //   // std::vector<float> tmp_output = trt_module.get_output(0);
  //   auto top = tensorrt_common::topK<float>(trt_module.get_output(0), 1);
  //   // std::cout << "Classification is: " << top[0] << std::endl;

  //   duration = std::chrono::duration_cast<std::chrono::microseconds>(
  //       std::chrono::system_clock::now() - t0);
  //   std::cout << "Inference time: " << static_cast<int>(duration.count())
  //             << " microseconds: " << top[0] << std::endl;
  //   t0 = std::chrono::system_clock::now();
  // }

  return 0;
}