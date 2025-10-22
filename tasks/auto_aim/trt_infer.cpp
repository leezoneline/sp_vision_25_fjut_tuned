#include "trt_infer.hpp"

#include <fstream>
#include <iostream>

#include "tools/logger.hpp"

namespace auto_aim
{

void TrtLogger::log(Severity severity, const char * msg) noexcept
{
  // 只记录警告和错误
  if (severity <= Severity::kWARNING) {
    tools::logger()->warn("[TensorRT] {}", msg);
  }
}

TrtInfer::TrtInfer(const std::string & engine_path, bool use_fp16)
: use_fp16_(use_fp16)
{
  load_engine(engine_path);
  if (!is_valid()) {
    throw std::runtime_error("Failed to load TensorRT engine: " + engine_path);
  }
  allocate_buffers();
  cudaStreamCreate(&stream_);
  tools::logger()->info("TensorRT engine loaded: {}", engine_path);
}

TrtInfer::~TrtInfer()
{
  free_buffers();
  if (stream_) cudaStreamDestroy(stream_);
  delete context_;
  delete engine_;
  delete runtime_;
}

void TrtInfer::load_engine(const std::string & engine_path)
{
  // 读取 engine 文件
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    tools::logger()->error("Cannot open engine file: {}", engine_path);
    return;
  }

  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  file.close();

  // 创建 runtime
  runtime_ = nvinfer1::createInferRuntime(logger_);
  if (!runtime_) {
    tools::logger()->error("Failed to create TensorRT runtime");
    return;
  }

  // 反序列化 engine
  engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
  if (!engine_) {
    tools::logger()->error("Failed to deserialize TensorRT engine");
    return;
  }

  // 创建执行上下文
  context_ = engine_->createExecutionContext();
  if (!context_) {
    tools::logger()->error("Failed to create TensorRT execution context");
    return;
  }

  // 获取输入输出索引和形状
  // TensorRT 10+ uses tensor names instead of binding indices
  const char* input_name = engine_->getIOTensorName(0);
  const char* output_name = engine_->getIOTensorName(1);
  
  input_index_ = 0;
  output_index_ = 1;

  // 获取输入形状
  auto input_dims = engine_->getTensorShape(input_name);
  input_shape_.resize(input_dims.nbDims);
  for (int i = 0; i < input_dims.nbDims; ++i) {
    input_shape_[i] = input_dims.d[i];
  }

  // 获取输出形状
  auto output_dims = engine_->getTensorShape(output_name);
  output_shape_.resize(output_dims.nbDims);
  for (int i = 0; i < output_dims.nbDims; ++i) {
    output_shape_[i] = output_dims.d[i];
  }

  tools::logger()->info(
    "TensorRT bindings: input_index={}, output_index={}", input_index_, output_index_);
}

void TrtInfer::allocate_buffers()
{
  if (!engine_) return;

  // 计算输入输出大小
  input_size_ = 1;
  for (auto dim : input_shape_) {
    input_size_ *= dim;
  }
  input_size_ *= sizeof(float);  // 假设输入为 float32

  output_size_ = 1;
  for (auto dim : output_shape_) {
    output_size_ *= dim;
  }
  output_size_ *= sizeof(float);  // 假设输出为 float32

  // 分配 GPU 内存
  cudaMalloc(&d_input_, input_size_);
  cudaMalloc(&d_output_, output_size_);

  tools::logger()->debug(
    "TensorRT buffers allocated: input={}MB, output={}MB", input_size_ / 1024 / 1024,
    output_size_ / 1024 / 1024);
}

void TrtInfer::free_buffers()
{
  if (d_input_) cudaFree(d_input_);
  if (d_output_) cudaFree(d_output_);
  d_input_ = nullptr;
  d_output_ = nullptr;
}

cv::Mat TrtInfer::infer(const cv::Mat & input)
{
  if (!is_valid()) {
    tools::logger()->error("TensorRT engine is not valid");
    return cv::Mat();
  }

  // 预处理：将 cv::Mat 转换为 float32 并归一化
  cv::Mat float_input;
  input.convertTo(float_input, CV_32F, 1.0 / 255.0);

  // 将数据拷贝到 GPU
  const char* input_name = engine_->getIOTensorName(0);
  const char* output_name = engine_->getIOTensorName(1);
  
  context_->setTensorAddress(input_name, d_input_);
  context_->setTensorAddress(output_name, d_output_);
  
  cudaMemcpyAsync(
    d_input_, float_input.data, input_size_, cudaMemcpyHostToDevice, stream_);

  // 执行推理
  bool status = context_->enqueueV3(stream_);
  if (!status) {
    tools::logger()->error("TensorRT inference failed");
    return cv::Mat();
  }

  // 将结果拷贝回 CPU
  std::vector<float> output_data(output_size_ / sizeof(float));
  cudaMemcpyAsync(
    output_data.data(), d_output_, output_size_, cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  // 将输出转换为 cv::Mat
  cv::Mat output;
  if (output_shape_.size() == 3) {
    // [1, H, W] -> [H, W]
    output = cv::Mat(output_shape_[1], output_shape_[2], CV_32F, output_data.data()).clone();
  } else if (output_shape_.size() == 2) {
    // [N, C] -> [N, C]
    output = cv::Mat(output_shape_[0], output_shape_[1], CV_32F, output_data.data()).clone();
  } else {
    // 其他情况：展平
    output = cv::Mat(1, output_data.size(), CV_32F, output_data.data()).clone();
  }

  return output;
}

std::vector<cv::Mat> TrtInfer::infer_batch(const std::vector<cv::Mat> & inputs)
{
  std::vector<cv::Mat> outputs;
  for (const auto & input : inputs) {
    outputs.push_back(infer(input));
  }
  return outputs;
}

}  // namespace auto_aim
