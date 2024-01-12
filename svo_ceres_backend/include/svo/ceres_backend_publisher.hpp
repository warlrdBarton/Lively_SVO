// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2017 Jonathan Huber <jonathan.huber at uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).

#pragma once

#include <boost/shared_ptr.hpp>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <svo/vio_common/backend_types.hpp>
#include <mutex>

#include "svo/ceres_backend/map.hpp"


//dgz change
namespace svo
{
class CeresBackendPublisher
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<CeresBackendPublisher> Ptr;
  using PointCloud = pcl::PointCloud<pcl::PointXYZI>;
  using PointType = pcl::PointXYZI;
  const std::string kWorldFrame = "world";

  CeresBackendPublisher(const ros::NodeHandle& nh_private,
                        const std::shared_ptr<ceres_backend::Map>& map_ptr);
  ~CeresBackendPublisher()
  {
  }

  Transformation getLastT_W_B() const
  {
    return state_.get_T_W_B();
  }

  void addFrame(const BundleId& bundle_id)
  {
    std::lock_guard<std::mutex> lock(mutex_frame_id_);
    last_added_frame_ = bundle_id;
  }
  
    void addFrame(const BundleId& bundle_id,const Eigen::Vector3d& last_w_g_B)
  {
    std::lock_guard<std::mutex> lock(mutex_frame_id_);
    last_added_frame_ = bundle_id;
    last_w_g_B_=last_w_g_B;
  }
  inline const Eigen::Vector3d& get_last_g() const 
  {
    return last_w_g_B_;
  }
  void publish(const ViNodeState& state, const int64_t timestamp,
               const int32_t seq);
  void publish(ViNodeState& state, const int64_t timestamp,
               const int32_t seq);
               
  void publishPropagationOdometry(const int64_t timestamp, const Transformation& T_WS,
                                  const Eigen::Vector3d& W_v_B, const Eigen::Vector3d& W_g_B);

private:
  ros::NodeHandle pnh_;

  mutable std::mutex mutex_frame_id_;

  std::shared_ptr<ceres_backend::Map> map_ptr_;  ///< The underlying svo::Map.

  // Transform used for tracing
  Eigen::Vector3d last_w_g_B_;
  ViNodeState state_;
  BundleId state_frame_id_ = -1;
  BundleId last_added_frame_ = -1;

  // publisher helpers
  ros::Publisher pub_imu_pose_;
  ros::Publisher pub_imu_pose_viz_;
  ros::Publisher pub_points_;
  ros::Publisher pub_lines_;
  ros::Publisher pub_twist_; 
  ros::Publisher pub_propagation_odometry_;


  // publisher functions
  void publishImuPose(const ViNodeState& state, const int64_t timestamp,
                      const int32_t seq);
  void publishBackendLandmarks(const int64_t timestamp) const;

  void publishBackendSegmentLandmarks(const int64_t timestamp) const;

  void publishImuTwist(ViNodeState& state, const int64_t timestamp,
                       const int32_t seq);
};

}  // namespace svo
