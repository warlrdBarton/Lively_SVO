/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *  Copyright (c) 2016, ETH Zurich, Wyss Zurich, Zurich Eye
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Jan 10, 2015
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Zurich Eye
 *********************************************************************************/

/**
 * @file implementation/Estimator.hpp
 * @brief Header implementation file for the Estimator class.
 * @author Stefan Leutenegger
 */

#pragma once

#include "svo/ceres_backend/estimator.hpp"

#include "svo/ceres_backend/reprojection_error.hpp"

/// \brief svo Main namespace of this package.
namespace svo {

// Add an observation to a landmark.
inline ceres::ResidualBlockId Estimator::addObservation(const FramePtr &frame,
                                                 const size_t keypoint_idx)
{
  const BackendId nframe_id = createNFrameId(frame->bundleId());
  DEBUG_CHECK_GE(frame->level_vec_(keypoint_idx), 0);
  const int cam_idx = frame->getNFrameIndex();
  // get Landmark ID.
  const BackendId landmark_backend_id = createLandmarkId(
        frame->track_id_vec_[keypoint_idx]);//get the observate landmark backend id ,the track_id_vec Storage that infomation
  DEBUG_CHECK(isLandmarkAdded(landmark_backend_id)) << "landmark not added";

  KeypointIdentifier kid(frame, keypoint_idx);
  // check for double observations
  DEBUG_CHECK(landmarks_map_.at(landmark_backend_id).observations.find(kid)
              == landmarks_map_.at(landmark_backend_id).observations.end())
      << "Trying to add the same landmark for the second time";

  // get the keypoint measurement
  size_t slot;
  bool success;
  std::tie(slot, success) = states_.findSlot(nframe_id);//find the frame slot 
  if (!success)
  {
    LOG(ERROR) << "Tried to add observation for frame that is either already "
               << "marginalized out or not yet added to the state. ID = "
               << nframe_id;
    return nullptr;
  }

  Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
  information *= 1.0 / static_cast<double>(1 << frame->level_vec_(keypoint_idx));

  // create error term
  DEBUG_CHECK(std::dynamic_pointer_cast<const Camera>(
                camera_rig_->getCameraShared(cam_idx)))
      << "Incorrect pointer cast requested. ";
  std::shared_ptr<ceres_backend::ReprojectionError > reprojection_error =
      std::make_shared<ceres_backend::ReprojectionError>(
        std::static_pointer_cast<const Camera>(
          camera_rig_->getCameraShared(cam_idx)),
        frame->px_vec_.col(keypoint_idx), information);//just new a resiual error ,but not link to the data

  if (isLandmarkFixed(landmark_backend_id.asInteger()))
  {
    reprojection_error->setPointConstant(true);
  }

  BackendId extrinsics_id = constant_extrinsics_ids_[cam_idx];
  if (estimate_temporal_extrinsics_)
  {
    extrinsics_id = changeIdType(nframe_id, IdType::Extrinsics, cam_idx);
  }
  ceres::ResidualBlockId ret_val = map_ptr_->addResidualBlock(
        reprojection_error,
        cauchy_loss_function_ptr_ ? cauchy_loss_function_ptr_.get() : nullptr,
        map_ptr_->parameterBlockPtr(nframe_id.asInteger()),
        map_ptr_->parameterBlockPtr(landmark_backend_id.asInteger()),
        map_ptr_->parameterBlockPtr(extrinsics_id.asInteger()));

  // remember
  landmarks_map_.at(landmark_backend_id).observations.insert(
        std::pair<KeypointIdentifier, uint64_t>(
          kid, reinterpret_cast<uint64_t>(ret_val)));

  return ret_val;
}

#ifdef SEGMENT_ENABLE//todo dgz

inline ceres::ResidualBlockId Estimator::addSegmentEndpointObservation(const FramePtr &frame,
                                                 const size_t segment_idx,const size_t endpoint_idx)
{
  const BackendId nframe_id = createNFrameId(frame->bundleId());
  DEBUG_CHECK_GE(frame->seg_level_vec_(segment_idx), 0);
  const int cam_idx = frame->getNFrameIndex();
  // get Landmark ID.
  const BackendId seg_landmark_backend_id= createSegmentEndpointLandmarkId(
        frame->seg_track_id_vec_[segment_idx],endpoint_idx);
  DEBUG_CHECK(isSegmentEndpointAdded(seg_landmark_backend_id)) << "segment "<< endpoint_idx<<" endpoint not added";

  SegmentIdentifier sid(frame, segment_idx);
  // check for double observations
  DEBUG_CHECK(seg_landmarks_map_.at(seg_landmark_backend_id).observations.find(sid)
              == seg_landmarks_map_.at(seg_landmark_backend_id).observations.end())
      << "Trying to add the same segment start endpoint for the second time";

  // get the keypoint measurement
  size_t slot;
  bool success;
  std::tie(slot, success) = states_.findSlot(nframe_id);//find the frame slot 
  if (!success)
  {
    LOG(ERROR) << "Tried to add segment observation for frame that is either already "
               << "marginalized out or not yet added to the state. ID = "
               << nframe_id;
    return nullptr;
  }

  Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
  information *= 1.0 / static_cast<double>(1 << frame->seg_level_vec_(segment_idx));

  // create error term
  DEBUG_CHECK(std::dynamic_pointer_cast<const Camera>(
                camera_rig_->getCameraShared(cam_idx)))
      << "Incorrect pointer cast requested. ";
  Eigen::Matrix<FloatType,2,1> measurement_;
  if(endpoint_idx==0)
  {
    measurement_=frame->seg_vec_.col(segment_idx).head<2>();
  }
  else{
    measurement_=frame->seg_vec_.col(segment_idx).tail<2>();

  }
  std::shared_ptr<ceres_backend::ReprojectionError > reprojection_error =
      std::make_shared<ceres_backend::ReprojectionError>(
        std::static_pointer_cast<const Camera>(
          camera_rig_->getCameraShared(cam_idx)),
        measurement_, information);//just new a resiual error ,but not link to the data


  if (isSegmentLandmarkFixed(seg_landmark_backend_id.asInteger()))
  {
    reprojection_error->setPointConstant(true);
  }

  BackendId extrinsics_id = constant_extrinsics_ids_[cam_idx];
  if (estimate_temporal_extrinsics_)
  {
    extrinsics_id = changeIdType(nframe_id, IdType::Extrinsics, cam_idx);
  }
  ceres::ResidualBlockId ret_val = map_ptr_->addResidualBlock(
        reprojection_error,
        cauchy_loss_function_ptr_ ? cauchy_loss_function_ptr_.get() : nullptr,
        map_ptr_->parameterBlockPtr(nframe_id.asInteger()),
        map_ptr_->parameterBlockPtr(seg_landmark_backend_id.asInteger()),
        map_ptr_->parameterBlockPtr(extrinsics_id.asInteger()));


  // remember
  seg_landmarks_map_.at(seg_landmark_backend_id).observations.insert(
        std::pair<SegmentIdentifier, uint64_t>(
          sid, reinterpret_cast<uint64_t>(ret_val)));//observation record the frane line info info and reprojection residual id

  return ret_val;
}

inline std::vector<ceres::ResidualBlockId> Estimator::addSegmentObservation(const FramePtr &frame,
                                                 const size_t segment_idx)
{
  ceres::ResidualBlockId start_endpoint=addSegmentEndpointObservation(frame,segment_idx,0);
  ceres::ResidualBlockId end_endpoint=addSegmentEndpointObservation(frame,segment_idx,1);
  return {start_endpoint,end_endpoint};
}
#endif

}  // namespace svo
