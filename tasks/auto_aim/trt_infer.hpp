#ifndef AUTO_AIM__TRT_INFER_HPP
#define AUTO_AIM__TRT_INFER_HPP

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace auto_aim
{

// TensorRT Logger
class TrtLogger : public nvinfer1::ILogger
{
  void log(Severity severity, const char * msg) noexcept override;
};

// TensorRT 推理引擎封装
class TrtInfer
{
public:
  TrtInfer(const std::string & engine_path, bool use_fp16 = false);
  ~TrtInfer();

  // 禁止拷贝
  TrtInfer(const TrtInfer &) = delete;
  TrtInfer & operator=(const TrtInfer &) = delete;

  // 推理接口：输入 cv::Mat (uint8 BGR format)，返回输出 cv::Mat (float32)
  cv::Mat infer(const cv::Mat & input);

  // 批量推理接口
  std::vector<cv::Mat> infer_batch(const std::vector<cv::Mat> & inputs);

  // 获取输入/输出形状
  std::vector<int> get_input_shape() const { return input_shape_; }
  std::vector<int> get_output_shape() const { return output_shape_; }

  // 检查引擎是否有效
  bool is_valid() const { return engine_ != nullptr && context_ != nullptr; }

private:
  void load_engine(const std::string & engine_path);
  void allocate_buffers();
  void free_buffers();

  TrtLogger logger_;
  nvinfer1::IRuntime * runtime_ = nullptr;
  nvinfer1::ICudaEngine * engine_ = nullptr;
  nvinfer1::IExecutionContext * context_ = nullptr;

  cudaStream_t stream_ = nullptr;

  // Binding 信息
  int input_index_ = 0;
  int output_index_ = 1;
  std::vector<int> input_shape_;   // [N, C, H, W] or [N, H, W, C]
  std::vector<int> output_shape_;  // [N, ...]

  // Device buffers
  void * d_input_ = nullptr;
  void * d_output_ = nullptr;

  size_t input_size_ = 0;
  size_t output_size_ = 0;

  bool use_fp16_ = false;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRT_INFER_HPP
