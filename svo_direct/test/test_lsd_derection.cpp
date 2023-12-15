// svo
#include <svo/test_utils/synthetic_dataset.h>
#include <svo/common/frame.h>
#include <svo/direct/matcher.h>
#include <svo/direct/feature_detection.h>
#include <svo/direct/feature_detection_utils.h>
#include <svo/line/precomp_custom.hpp>
// others
#include <opencv2/highgui/highgui.hpp> // imread
#include <opencv2/imgproc/imgproc.hpp>
#include <svo/common/types.h>

using namespace svo;
int main(int argc, char **argv)
{
  // Load dataset.
  std::string dataset_dir = "/home/sunteng/dataset/day_11_25_office_stereo_imu_sync_0_640_480";
  svo::test_utils::SyntheticDataset dataset(dataset_dir, 0, 0);

  svo::DetectorOptions options;
  svo::SegmentDetectorOptions segmenter_options;
  options.detector_type = svo::DetectorType::kFastGrad;
  segmenter_options.detector_type = svo::DetectorType::kLSDSegment;

  if (argc >= 4)
    options.threshold_primary = std::atof(argv[3]);

  svo::AbstractDetectorPtr detector =
      svo::feature_detection_utils::makeDetector(options, dataset.cam());

  svo::SegmentAbstractDetectorPtr segmenter =
      svo::feature_detection_utils::makeSegmentDetector(segmenter_options, dataset.cam());

  svo::FramePtr frame;
  while (dataset.getNextFrame(5u, frame, nullptr))
  {

    Keypoints new_px;
    Levels new_levels;
    Scores new_scores;
    Gradients new_grads;
    FeatureTypes new_types;
    const size_t max_n_features = detector->grid_.size();
    detector->detect(
        frame->img_pyr_, frame->getMask(), max_n_features, new_px,
        new_scores, new_levels, new_grads, new_types);

    // dgz todo
    Segments new_seg;
    Scores new_seg_score;
    FeatureTypes new_seg_types;
    Levels new_seg_levels;
      Gradients grad_vec;

    // uchar * frame->img_pyr_[0]=frame->img_pyr_[0].ptr();
    segmenter->detect(
        frame->img_pyr_, frame->getMask(), segmenter->options_.max_segment_num, new_seg,
        new_seg_score, new_seg_levels,grad_vec, new_seg_types);
    cv::Mat three_channel_mat;
    cv::cvtColor(frame->img_pyr_[0], three_channel_mat, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < new_types.size(); ++i)
    {
      const auto &px = new_px.col(i);
      const auto &g = new_grads.col(i);
      switch (new_types[i])
      {
      case FeatureType::kEdgelet:
        cv::line(three_channel_mat, cv::Point2f(px(0) + 3 * g(1), px(1) - 3 * g(0)),
                 cv::Point2f(px(0) - 3 * g(1), px(1) + 3 * g(0)),
                 cv::Scalar(255, 0, 255), 2);
        break;
      case FeatureType::kCorner:
        cv::rectangle(three_channel_mat, cv::Point2f(px(0) - 2, px(1) - 2),
                      cv::Point2f(px(0) + 2, px(1) + 2),
                      cv::Scalar(0, 255, 0), -1);
        break;
      case FeatureType::kMapPoint:
        cv::rectangle(three_channel_mat, cv::Point2f(px(0) - 2, px(1) - 2),
                      cv::Point2f(px(0) + 2, px(1) + 2),
                      cv::Scalar(255, 0, 0), -1);
        break;
      case FeatureType::kFixedLandmark:
        cv::rectangle(three_channel_mat, cv::Point2f(px(0) - 3, px(1) - 3),
                      cv::Point2f(px(0) + 3, px(1) + 3),
                      cv::Scalar(101, 236, 255), -1);
        break;
      case FeatureType::kEdgeletSeed:
      case FeatureType::kEdgeletSeedConverged:
        cv::line(three_channel_mat, cv::Point2f(px(0) + 3 * g(1), px(1) - 3 * g(0)),
                 cv::Point2f(px(0) - 3 * g(1), px(1) + 3 * g(0)),
                 cv::Scalar(0, 0, 255), 2);
        break;
      case FeatureType::kCornerSeed:
      case FeatureType::kCornerSeedConverged:
        cv::circle(three_channel_mat, cv::Point2f(px(0), px(1)),
                   5, cv::Scalar(0, 255, 0), 1);
        break;
      case FeatureType::kMapPointSeed:
      case FeatureType::kMapPointSeedConverged:
        cv::circle(three_channel_mat, cv::Point2f(px(0), px(1)),
                   5, cv::Scalar(255, 0, 0), 1);
        break;

      default:
        cv::circle(three_channel_mat, cv::Point2f(px(0), px(1)),
                   5, cv::Scalar(0, 0, 255), -1);
        break;
      }
    }
    // std::cout<<static_cast<size_t>(new_seg.cols());
    for (size_t i = 0; i < static_cast<size_t>(new_seg.cols()); i++)
    {
      // std::cout<<" "<<static_cast<int>(new_seg_types[i])<<std::endl;
      const auto &seg = new_seg.col(i);
      // if (frame.landmark_vec_[i] == nullptr && frame.seed_ref_vec_[i].keyframe == nullptr && only_matched_seg)
      //   continue;
      switch (new_seg_types[i])
      {
        // const auto &g = frame.grad_vec_.col(i);
      case FeatureType::kSegment:
        cv::line(three_channel_mat, cv::Point2f(seg(0), seg(1)),
                 cv::Point2f(seg(2), seg(3)),
                 cv::Scalar(63, 125, 50), 2);
        break;

      default:
        break;
      }
      // const auto &g = frame.grad_vec_.col(i);
    }

    cv::imshow("window_name", three_channel_mat);
    cv::waitKey(0);
  }

  return 0;
}
