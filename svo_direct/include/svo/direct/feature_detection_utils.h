// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <memory>
#include <array>
#include <svo/common/types.h>
#include <svo/common/camera_fwd.h>
#include <svo/common/occupancy_grid_2d.h>
#include <svo/direct/feature_detection_types.h>
#ifdef CUDAFAST_ENABLE
#include <opencv2/cudafeatures2d.hpp>
#endif
#include <svo/direct/ELSED.h>
namespace svo {

class AbstractDetector;
class SegmentAbstractDetector;
using AbstractDetectorPtr = std::shared_ptr<AbstractDetector>;
using SegmentAbstractDetectorPtr = std::shared_ptr<SegmentAbstractDetector>;

namespace feature_detection_utils {

/// Factory returns a pointer to a detector that is specified with in the options.
AbstractDetectorPtr makeDetector(
    const DetectorOptions& options,
    const CameraPtr& cam);
SegmentAbstractDetectorPtr makeSegmentDetector(const SegmentDetectorOptions& options,
    const CameraPtr& cam);
void fillFeatures(const Corners& corners,
    const FeatureType& type,
    const cv::Mat& mask,
    const double& threshold,
    const size_t max_n_features,
    Keypoints& keypoints,
    Scores& scores,
    Levels& levels,
    Gradients& gradients,
    FeatureTypes& types,
    OccupandyGrid2D& grid);

void fastDetector(
    const ImgPyr& img_pyr,
    const int threshold,
    const int border,
    const size_t min_level,
    const size_t max_level,
    Corners& corners,
    OccupandyGrid2D& grid);

void shiTomasiDetector(
    const ImgPyr& img_pyr,
    const int threshold,
    const int border,
    const size_t min_level,
    const size_t max_level,
    Corners& corners,
    OccupandyGrid2D& grid,
    OccupandyGrid2D& closeness_check_grid);

#ifdef CUDAFAST_ENABLE
void cudaFastDetector(
    const ImgPyr& img_pyr,
    const int threshold,
    const int border,
    const size_t min_level,
    const size_t max_level,
    Corners& corners,
    OccupandyGrid2D& grid,
    cv::Ptr<cv::cuda::FastFeatureDetector> detector_ptr
    );
#endif

void edgeletDetector_V1(
    const ImgPyr& img_pyr,
    const int threshold,
    const int border,
    const int min_level,
    const int max_level,
    Corners& corners,
    OccupandyGrid2D& grid);

void edgeletDetector_V2(
    const ImgPyr& img_pyr,
    const int threshold,
    const int border,
    const int min_level,
    const int max_level,
    Corners& corners,
    OccupandyGrid2D& grid);

bool getCornerAngle(
    const ImgPyr& img_pyr,
    const Eigen::Ref<const Keypoint>& level_0,
    const size_t level,
    double* angle);

/// Get patch minimum eigenvalue.
bool getShiTomasiScore(
    const cv::Mat& img,
    const Eigen::Vector2i& px,
    double* score);

void setCornerAngles(
    const ImgPyr& img_pyr,
    Corners* corners);

void setCornerLevel(
    const ImgPyr& mag_pyr,
    Corners* corners);

void computeDerivMaxMagnitude(
    const cv::Mat& img_8u,
    cv::Mat& mag_8u);

void computeDerivHuangMumford(
    const cv::Mat& img_8u,
    cv::Mat& mag_32f,
    float alpha=10.0f,
    float q=0.98f);

void nonmax(
    const cv::Mat& img_32f,
    const float thresh,
    Corners* corners);

void displayGrid(
    const OccupandyGrid2D& old_grid,
    const Keypoints& keypoints,
    const int img_width,
    const int img_height);

void nonlinearDiffusion(
    const cv::Mat& img_8u,
    cv::Mat& img_8u_diffused,
    const double timestep = 0.25,   // absolute max is 0.5 for stability
    const double final_time = 2.5);

void detectGuillermoEdges(
    const cv::Mat& src_gray,
    cv::Mat& dest,
    int low_threshold = 20,
    int ratio = 3,
    int kernel_size = 3);

void detectCannyEdges(
    const cv::Mat& src_gray,
    cv::Mat& dest,
    int low_threshold = 20,
    int ratio = 3,
    int kernel_size = 3);

void detectSobelEdges(
    const cv::Mat& src_gray,
    cv::Mat& dest,
    int low_threshold = 30,
    int kernel_size = 2);

void drawFeatures(
    const Frame& frame,
    const size_t level,
    const bool only_matched_features,
    cv::Mat* img_rgb);

double getAngleAtPixelUsingHistogram(
    const cv::Mat& img,
    const Eigen::Vector2i& px,
    const size_t halfpatch_size);

void nonmax_3x3(
    const std::vector<Eigen::Vector2i>& corners,
    const std::vector<int>& scores,
    std::vector<int>& nonmax_corners);

void nonmax_3x3_double(const std::vector<Eigen::Vector2i> &corners, const std::vector<svo::Score> &scores,
                    std::vector<int> &nonmax_corners);
                    
void mergeGrids(const OccupandyGrid2D& grid1, OccupandyGrid2D* grid2);

// Compute an angle histogram.
namespace angle_hist {

constexpr size_t n_bins = 36;
using AngleHistogram = std::array<double, n_bins>;

void angleHistogram(
    const cv::Mat& img, int x, int y, int halfpatch_size, AngleHistogram& hist);

bool gradientAndMagnitudeAtPixel(
    const cv::Mat& img, int x, int y, double* mag, double* angle);

void smoothOrientationHistogram(
    AngleHistogram& hist);

double getDominantAngle(
    const AngleHistogram& hist);

} // angle_hist

void ElSEDdetect(
    const ImgPyr& img_pyr,
    const int border,
    const size_t min_level,
    const size_t max_level,
    ScoreSegments& segs,
    OccupandyGrid2D& grid,
    std::unique_ptr<upm::ELSED> &elsed_);

void fillSegment(
    const ScoreSegments& segs,//same level with corner
    const FeatureType& type,
    const cv::Mat& mask,
    const size_t max_n_features,
    Segments& keysegs,
    Scores& scores,
    Levels& levels,
    Gradients & grads,
    FeatureTypes& types,
    OccupandyGrid2D& grid,
    double threshold
);

    void LSDDetect(const ImgPyr &img_pyr,
        const int border,
        const size_t min_level,
        const size_t max_level,
        ScoreSegments &segs, // in this scope to add segment start mid end 3 point
        OccupandyGrid2D &grid,
        cv::Ptr<cv::LineSegmentDetector> detecter_);

}; // feature_detection_utils
} // namespace svo
