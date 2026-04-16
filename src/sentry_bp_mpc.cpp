#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "io/ros2/ros2.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                     | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  // IO 设备
  io::ROS2 ros2;
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  io::Camera back_camera("configs/camera.yaml");

  // 自瞄模块
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);
  auto_aim::Shooter shooter(config_path);

  // 全向感知决策器
  omniperception::Decider decider(config_path);

  // 目标队列：主线程 → Planner 线程
  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  std::atomic<bool> quit = false;
  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};

  // 用于保存发送的命令（用于 PlotJuggler 可视化）
  struct SendCommand {
    float yaw = 0;
    float pitch = 0;
  } send_cmd;

  // ========== Planner 线程 ==========
  // 10ms 周期运行 MPC，计算轨迹并发送控制命令
  auto plan_thread = std::thread([&]() {
    while (!quit) {
      if (!target_queue.empty() && mode == io::GimbalMode::AUTO_AIM) {
        auto target = target_queue.front();
        auto gs = gimbal.state();
        auto plan = planner.plan(target, gs.bullet_speed);

        send_cmd.yaw = plan.yaw;
        send_cmd.pitch = plan.pitch;

        gimbal.send(
          plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
          plan.pitch_acc);

        std::this_thread::sleep_for(10ms);
      } else {
        std::this_thread::sleep_for(50ms);
      }
    }
  });

  // ========== 主循环 ==========
  while (!exiter.exit()) {
    mode = gimbal.mode();

    // 模式切换日志
    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      last_mode = mode.load();
    }

    // 读取图像和 IMU
    camera.read(img, t);
    q = gimbal.q(t);
    auto gs = gimbal.state();
    // recorder.record(img, q, t);
    solver.set_R_gimbal2world(q);

    // 计算云台欧拉角（用于全向感知）
    Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    // PlotJuggler 可视化
    nlohmann::json plot_data;
    plot_data["gimbal/yaw_recv"] = gs.yaw;
    plot_data["gimbal/pitch_recv"] = gs.pitch;
    plot_data["gimbal/yaw_send"] = send_cmd.yaw;
    plot_data["gimbal/pitch_send"] = send_cmd.pitch;
    plotter.plot(plot_data);

    // ========== 自瞄模式 ==========
    if (mode.load() == io::GimbalMode::AUTO_AIM) {
      // YOLO 检测
      auto armors = yolo.detect(img);

      // ROS2 订阅无敌状态并过滤
      decider.get_invincible_armor(ros2.subscribe_enemy_status());
      decider.armor_filter(armors);
      decider.set_priority(armors);

      // 目标跟踪
      auto targets = tracker.track(armors, t);

      // 全向感知逻辑：tracker lost 时使用后置 USB 相机
      if (tracker.state() == "lost") {
        // 清空 planner 队列
        target_queue.push(std::nullopt);

        // 使用 decider 的全向感知（后置相机）
        auto omni_command = decider.decide(yolo, gimbal_pos, back_camera);

        // 发送全向感知命令（无 MPC，直接位置控制）
        gimbal.send(omni_command.control, omni_command.shoot, omni_command.yaw, 0, 0,
                    omni_command.pitch, 0, 0);

        send_cmd.yaw = omni_command.yaw;
        send_cmd.pitch = omni_command.pitch;
      } else {
        // 有目标时，推入队列供 planner 线程使用
        target_queue.push(targets.front());
        // 发射逻辑由 planner 线程中的 plan.fire 决定
      }

      // ROS2 发布目标信息
      Eigen::Vector4d target_info = decider.get_target_info(armors, targets);
      ros2.publish(target_info);
    }

    // ========== 空闲模式 ==========
    else {
      target_queue.push(std::nullopt);
      send_cmd.yaw = 0;
      send_cmd.pitch = 0;
      gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
    }
  }

  // ========== 退出清理 ==========
  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
