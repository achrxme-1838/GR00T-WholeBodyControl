/**
 * @file odom_subscriber.hpp
 * @brief ROS 2 subscriber for FAST-LIO-style /Odometry messages.
 *
 * Maintains the latest base pose (position + quaternion) received on the
 * `/Odometry` topic on its own background spin thread.  Used by G1Deploy
 * as a substitute for a real odometry source so that ComputeRobotFK and
 * the diff-body observation gatherers can express both the robot and the
 * reference motion in a common "origin" frame anchored at the robot's
 * world pose at the moment the operator pressed T (`operator_state.play`
 * false→true transition).
 */

#ifndef ODOM_SUBSCRIBER_HPP
#define ODOM_SUBSCRIBER_HPP

#if HAS_ROS2

#include <array>
#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <nav_msgs/msg/odometry.hpp>

class OdomSubscriber {
 public:
  explicit OdomSubscriber(const std::string& topic = "/Odometry",
                          const std::string& node_name = "g1_odom_subscriber")
      : topic_(topic), node_name_(node_name) {}

  ~OdomSubscriber() { stop(); }

  bool start() {
    if (running_.load()) return true;

    if (!rclcpp::ok()) {
      try {
        rclcpp::init(0, nullptr);
      } catch (const std::exception& e) {
        std::cerr << "[OdomSubscriber] rclcpp::init failed: " << e.what() << std::endl;
        return false;
      }
    }

    try {
      node_ = rclcpp::Node::make_shared(node_name_);
      sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
          topic_, rclcpp::SensorDataQoS(),
          [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { this->callback(msg); });
      executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
      executor_->add_node(node_);
    } catch (const std::exception& e) {
      std::cerr << "[OdomSubscriber] node/subscriber setup failed: " << e.what() << std::endl;
      return false;
    }

    running_.store(true);
    spin_thread_ = std::thread([this]() {
      try {
        executor_->spin();
      } catch (const std::exception& e) {
        std::cerr << "[OdomSubscriber] spin threw: " << e.what() << std::endl;
      }
    });

    std::cout << "[OdomSubscriber] subscribed to '" << topic_ << "' (node='"
              << node_name_ << "')" << std::endl;
    return true;
  }

  void stop() {
    if (!running_.load()) return;
    running_.store(false);
    try {
      if (executor_) executor_->cancel();
    } catch (...) {}
    if (spin_thread_.joinable()) spin_thread_.join();
    try {
      if (executor_ && node_) executor_->remove_node(node_);
    } catch (...) {}
    sub_.reset();
    executor_.reset();
    node_.reset();
  }

  bool has_data() const { return has_data_.load(); }

  /// Copy out the latest pose. Returns false if no message has been received yet.
  bool get_latest(std::array<double, 3>& pos_w, std::array<double, 4>& quat_w_wxyz) const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!has_data_.load()) return false;
    pos_w = latest_pos_w_;
    quat_w_wxyz = latest_quat_w_wxyz_;
    return true;
  }

 private:
  void callback(nav_msgs::msg::Odometry::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_pos_w_ = {
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        msg->pose.pose.position.z,
    };
    latest_quat_w_wxyz_ = {
        msg->pose.pose.orientation.w,
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
    };
    has_data_.store(true);
  }

  std::string topic_;
  std::string node_name_;

  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spin_thread_;
  std::atomic<bool> running_{false};

  mutable std::mutex data_mutex_;
  std::atomic<bool> has_data_{false};
  std::array<double, 3> latest_pos_w_{0.0, 0.0, 0.0};
  std::array<double, 4> latest_quat_w_wxyz_{1.0, 0.0, 0.0, 0.0};
};

#else  // !HAS_ROS2

#include <array>
#include <string>

// Stub when ROS2 is unavailable so call sites compile.
class OdomSubscriber {
 public:
  explicit OdomSubscriber(const std::string& = "/Odometry",
                          const std::string& = "g1_odom_subscriber") {}
  bool start() { return false; }
  void stop() {}
  bool has_data() const { return false; }
  bool get_latest(std::array<double, 3>&, std::array<double, 4>&) const { return false; }
};

#endif  // HAS_ROS2

#endif  // ODOM_SUBSCRIBER_HPP
