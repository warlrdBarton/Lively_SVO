#include "svo/ceres_backend_publisher.hpp"

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <vikit/output_helper.h>
#include <vikit/params_helper.h>

namespace
{
  template <typename T>
  void normalizeVector(const std::vector<T> &in, std::vector<float> *out)
  {
    auto res = std::minmax_element(in.begin(), in.end());
    const T min = *res.first;
    const T max = *res.second;
    const float dist = static_cast<float>(max - min);

    out->resize(in.size());
    for (size_t i = 0; i < in.size(); i++)
    {
      (*out)[i] = (in[i] - min) / dist;
    }
  }
}

namespace svo
{
  CeresBackendPublisher::CeresBackendPublisher(
      const ros::NodeHandle &nh_private,
      const std::shared_ptr<ceres_backend::Map> &map_ptr)
      : pnh_(nh_private), map_ptr_(map_ptr)
  {
    pub_imu_pose_ = pnh_.advertise<geometry_msgs::PoseWithCovarianceStamped>(
        "backend_pose_imu", 10);
    pub_imu_pose_viz_ =
        pnh_.advertise<geometry_msgs::PoseStamped>("backend_pose_imu_viz", 10);
    pub_points_ =
        pnh_.advertise<PointCloud>("backend_points", 10);
    pub_lines_ = pnh_.advertise<visualization_msgs::MarkerArray>("backend_lines", 10);

    pub_twist_ = pnh_.advertise<geometry_msgs::TwistStamped>("twist_imu", 10);
  }

  void CeresBackendPublisher::publish(ViNodeState &state,
                                      const int64_t timestamp,
                                      const int32_t seq)
  {
    publishImuPose(state, timestamp, seq);
    publishImuTwist(state, timestamp, seq);
    publishBackendLandmarks(timestamp);
    publishBackendSegmentLandmarks(timestamp);
  }

  // dgz change
  void CeresBackendPublisher::publishImuTwist(ViNodeState &state,
                                              const int64_t timestamp,
                                              const int32_t seq)
  {
    size_t n_twist_sub = pub_twist_.getNumSubscribers();
    if (n_twist_sub < 1)
      return;
    ros::Time time = ros::Time().fromNSec(timestamp);
    geometry_msgs::TwistStamped::Ptr msg_twist(new geometry_msgs::TwistStamped);
    Eigen::Vector3d W_v_B, W_g_B;
    W_v_B = state.get_W_v_B();
    state.set_W_g_B(get_last_g());
    W_g_B = state.get_W_g_B();
    msg_twist->header.seq = seq;
    msg_twist->header.stamp = time;
    msg_twist->header.frame_id = kWorldFrame;

    msg_twist->twist.linear.x = W_v_B.x();
    msg_twist->twist.linear.y = W_v_B.y();
    msg_twist->twist.linear.z = W_v_B.z();

    msg_twist->twist.angular.x = W_g_B.x();
    msg_twist->twist.angular.y = W_g_B.y();
    msg_twist->twist.angular.z = W_g_B.z();
    pub_twist_.publish(msg_twist);
    VLOG(100) << "Publish twist Pose";
  }

  void CeresBackendPublisher::publishImuPose(const ViNodeState &state,
                                             const int64_t timestamp,
                                             const int32_t seq)
  {
    // Trace state
    state_ = state;

    {
      std::lock_guard<std::mutex> lock(mutex_frame_id_);
      state_frame_id_ = BundleId(seq);
    }

    size_t n_pose_sub = pub_imu_pose_.getNumSubscribers();
    size_t n_pose_viz_sub = pub_imu_pose_viz_.getNumSubscribers();
    if (n_pose_sub == 0 && n_pose_viz_sub == 0)
    {
      return;
    }
    VLOG(100) << "Publish IMU Pose";

    Eigen::Quaterniond q = state.get_T_W_B().getRotation().toImplementation();
    Eigen::Vector3d p = state.get_T_W_B().getPosition();
    ros::Time time = ros::Time().fromNSec(timestamp);

    if (n_pose_sub > 0)
    {
      geometry_msgs::PoseWithCovarianceStampedPtr msg_pose(
          new geometry_msgs::PoseWithCovarianceStamped);
      msg_pose->header.seq = seq;
      msg_pose->header.stamp = time;
      msg_pose->header.frame_id = kWorldFrame;
      msg_pose->pose.pose.position.x = p[0];
      msg_pose->pose.pose.position.y = p[1];
      msg_pose->pose.pose.position.z = p[2];
      msg_pose->pose.pose.orientation.x = q.x();
      msg_pose->pose.pose.orientation.y = q.y();
      msg_pose->pose.pose.orientation.z = q.z();
      msg_pose->pose.pose.orientation.w = q.w();
      for (size_t i = 0; i < 36; ++i)
        msg_pose->pose.covariance[i] = 0;
      pub_imu_pose_.publish(msg_pose);
    }

    if (n_pose_viz_sub > 0)
    {
      geometry_msgs::PoseStampedPtr msg_pose(new geometry_msgs::PoseStamped);
      msg_pose->header.seq = seq;
      msg_pose->header.stamp = time;
      msg_pose->header.frame_id = kWorldFrame;
      msg_pose->pose.position.x = p[0];
      msg_pose->pose.position.y = p[1];
      msg_pose->pose.position.z = p[2];
      msg_pose->pose.orientation.x = q.x();
      msg_pose->pose.orientation.y = q.y();
      msg_pose->pose.orientation.z = q.z();
      msg_pose->pose.orientation.w = q.w();
      pub_imu_pose_viz_.publish(msg_pose);
    }
  }

  void CeresBackendPublisher::publishBackendLandmarks(
      const int64_t timestamp) const
  {
    if (pub_points_.getNumSubscribers() == 0)
    {
      return;
    }

    // get all landmarks
    const std::unordered_map<
        uint64_t, std::shared_ptr<ceres_backend::ParameterBlock>> &idmap =
        map_ptr_->idToParameterBlockMap();
    size_t n_pts = 0;
    std::vector<const double *> landmark_pointers;
    std::vector<uint64_t> point_ids;
    for (auto &it : idmap)
    {
      if (it.second->typeInfo() == "HomogeneousPointParameterBlock" &&
          !it.second->fixed())
      {
        n_pts++;
        landmark_pointers.push_back(it.second->parameters());
        point_ids.push_back(it.second->id());
      }
    }

    if (n_pts < 5)
    {
      return;
    }

    std::vector<float> intensities;
    normalizeVector(point_ids, &intensities);

    // point clound to publish
    PointCloud pc;
    ros::Time pub_time;
    pub_time.fromNSec(timestamp);
    pcl_conversions::toPCL(pub_time, pc.header.stamp);
    pc.header.frame_id = kWorldFrame;
    pc.reserve(n_pts);
    for (size_t i = 0; i < landmark_pointers.size(); i++)
    {
      const auto p = landmark_pointers[i];
      PointType pt;
      pt.intensity = intensities[i];
      pt.x = p[0];
      pt.y = p[1];
      pt.z = p[2];
      pc.push_back(pt);
    }
    pub_points_.publish(pc);//backend_points
  }

  void CeresBackendPublisher::publishBackendSegmentLandmarks(
      const int64_t timestamp) const
  {
    if (pub_lines_.getNumSubscribers() == 0)
    {
      // std::cout<<"cannot pub line"<<std::endl;
      return;
    }

    // get all landmarks
    const std::unordered_map<
        uint64_t, std::shared_ptr<ceres_backend::ParameterBlock>> &idmap =
        map_ptr_->idToParameterBlockMap();
    size_t n_line = 0;
    std::unordered_map<uint64_t, Eigen::Matrix<svo::FloatType, 6, 1>,
                       std::hash<uint64_t>, std::equal_to<uint64_t>,
                       Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Matrix<svo::FloatType, 6, 1>>>>
        landmark_lines;
    double *temp;
    for (auto &it : idmap)
    {
      if (it.first >> 56 == 4 &&
          !it.second->fixed())
      {

        if (landmark_lines.find(it.first/2) != landmark_lines.end())
        {
          temp = it.second->parameters();
          landmark_lines.at(it.first/2).tail<3>() << temp[0], temp[1], temp[2];
          n_line++;
        }
        else
        {
          temp = it.second->parameters();
          landmark_lines.emplace_hint(landmark_lines.end(), it.first/2, (Eigen::Matrix<double, 6, 1>() << temp[0], temp[1], temp[2], 0, 0, 0).finished());
        }
      }
    }
    // std::cout<<"has segment number"<<landmark_lines.size()<<" "<<idmap.size()<<" "<<n_line<<std::endl;

    if (n_line < 5)
    {
      return;
    }

    // point clound to publish
    visualization_msgs::MarkerArray marker_array;
    Eigen::Matrix<svo::FloatType, 6, 1> line_3d;
    // 遍历点云
    size_t i = 0;
    for (auto it = landmark_lines.begin(); it != landmark_lines.end(); ++it, ++i)
    {
      line_3d = it->second;

      // 计算点到平面的距离

      // 创建一个Marker
      visualization_msgs::Marker marker;
      marker.header.stamp = ros::Time::now(); // 假设你的点云是在base_link坐标系下的
      marker.header.frame_id = kWorldFrame;
      marker.id = i;
      // 使用当前时间作为时间戳
      marker.type = visualization_msgs::Marker::LINE_LIST;
      marker.scale.x = 0.08; // 设置线宽
      marker.color.r = 0.75;
      marker.color.g = 0.5;
      marker.color.b = 0.25;
      marker.color.a = 1.0;

      // 设置Marker的两个点
      geometry_msgs::Point p1, p2;
      p1.x = line_3d(0);
      p1.y = line_3d(1);
      p1.z = line_3d(2);
      p2.x = line_3d(3);
      p2.y = line_3d(4);
      p2.z = line_3d(5);
      // std::cout<<line_3d.transpose()<<std::endl;
      marker.points.push_back(p1);
      marker.points.push_back(p2);

      // 将Marker添加到MarkerArray中
      marker_array.markers.push_back(marker);
    }
    pub_lines_.publish(marker_array);
  }

} // namespace svo
