// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).

#pragma once

#include <svo/global.h>

#include <svo/line/precomp_custom.hpp>

namespace svo {

struct StereoTriangulationOptions
{
  size_t triangulate_n_features = 120;
  size_t triangulate_n_segment = 50;
  double mean_depth_inv = 1.0/3.0;
  double min_depth_inv = 1.0/1.0;
  double max_depth_inv = 1.0/50.0;
};

class StereoTriangulation
{
public:
  typedef std::shared_ptr<StereoTriangulation> Ptr;

  StereoTriangulationOptions options_;
  DetectorPtr feature_detector_;
  SegmentDetectorConstPtr segment_detector_;
  std::unique_ptr<svo::line_match> line_match;
  cv::Ptr<cv::line_descriptor::BinaryDescriptor> descriptor_;
  StereoTriangulation(
      const StereoTriangulationOptions& options,
      const DetectorPtr& feature_detector);
  StereoTriangulation(
      const StereoTriangulationOptions& options,
      const DetectorPtr& feature_detector,
      const  SegmentDetectorConstPtr& segment_detector
      );
  ~StereoTriangulation() = default;

  void compute(const FramePtr& frame0, const FramePtr& frame1);
  // void bresenhamLine(const Segments& line,std::vector<Keypoints>& linePoints);
};

} // namespace svo
