#ifndef AUTO_BUFF__YOLO11_BUFF_HPP
#define AUTO_BUFF__YOLO11_BUFF_HPP
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>

#include "tools/logger.hpp"
#include "../auto_aim/trt_infer.hpp"

namespace auto_buff
{

using auto_aim::TrtInfer;

const std::vector<std::string> class_names = {"buff", "r"};

class YOLO11_BUFF
{
public:
  struct Object
  {
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<cv::Point2f> kpt;
  };

  YOLO11_BUFF(const std::string & config);

  // 使用NMS，用来获取多个框
  std::vector<Object> get_multicandidateboxes(cv::Mat & image);

  // 寻找置信度最高的框
  std::vector<Object> get_onecandidatebox(cv::Mat & image);

private:
  std::unique_ptr<TrtInfer> trt_infer_;
  const int NUM_POINTS = 6;

  // 将image保存为"../result/$${programName}.jpg"
  void save(const std::string & programName, const cv::Mat & image);
};
}  // namespace auto_buff
#endif