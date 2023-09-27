#pragma once

#include <utility> // std::pair
#include <boost/shared_ptr.hpp>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
// #include <gtsam/slam/generic_stereoFactor.h>
// forward declarationsSmartProjectionPoseFactor
// /home/sunteng/catkin_ws/src/gtsam-4.2.0/gtsam_unstable/slam/PoseBetweenFactor.h
// /home/sunteng/catkin_ws/src/gtsam-4.2.0/gtsam/sam/RangeFactor.h
// /home/sunteng/catkin_ws/src/rpg_svo_pro_open/svo_global_map/include/svo/gtsam/camera_bearing_extrinsics_factor.h
// /home/sunteng/catkin_ws/src/rpg_svo_pro_open/svo_global_map/include/svo/gtsam/camera_bearing_factor.h

#include "slam/PoseBetweenFactor.h"
#include "sam/RangeFactor.h"

#include "camera_bearing_extrinsics_factor.h"
#include "camera_bearing_factor.h"

namespace svo {

// Typedef of desired classes to reduce compile-time
using ProjectionFactor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>;
using SmartFactor = gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2>;
using SmartFactorPtr = boost::shared_ptr<SmartFactor>;
using RelativePoseFactor = gtsam::PoseBetweenFactor<gtsam::Pose3>;
using CamPointDistFactor = gtsam::RangeFactorWithTransform<gtsam::Pose3, gtsam::Point3, double>;
using CameraBearingFactor3D = gtsam::CameraBearingFactor<gtsam::Pose3, gtsam::Point3>;
using CameraBearingTbcFactor = gtsam::CameraBearingExtrinsicsFactor<gtsam::Pose3, gtsam::Point3>;
using PointMatchFactor = gtsam::BetweenFactor<gtsam::Point3>;

struct SmartFactorInfo
{
  boost::shared_ptr<SmartFactor> factor_ = nullptr;
  int slot_in_graph_ = -1;
  SmartFactorInfo() = delete;
  SmartFactorInfo(const boost::shared_ptr<SmartFactor>& factor,
                  const int slot_idx)
    :factor_(factor), slot_in_graph_(slot_idx)
  {

  }
};

using SmartFactorInfoMap = gtsam::FastMap<int, SmartFactorInfo> ;

} // namespace svo
