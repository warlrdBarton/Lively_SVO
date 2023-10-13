// svo
#include <svo/test_utils/synthetic_dataset.h>
#include <svo/common/frame.h>
#include <svo/direct/matcher.h>
#include <svo/direct/feature_detection.h>
#include <svo/direct/feature_detection_utils.h>

// others
#include <opencv2/highgui/highgui.hpp> // imread
#include <opencv2/imgproc/imgproc.hpp>
#include <svo/common/types.h>


int main(int argc, char **argv)
{
  // Load dataset.
  std::string dataset_dir="/home/sunteng/world/data/dataset/realsense_d435i_office/day_11_25_office_stereo_imu_sync_0_640_480/";
  svo::test_utils::SyntheticDataset dataset(dataset_dir, 0, 0);

  svo::DetectorOptions options;
  svo::SegmentDetectorOptions segmenter_options;
  options.detector_type = svo::DetectorType::kFastGrad;
  segmenter_options.detector_type =svo::DetectorType::kELSEDSegment;

  if(argc >= 4)
    options.threshold_primary = std::atof(argv[3]);

  svo::AbstractDetectorPtr detector =
      svo::feature_detection_utils::makeDetector(options, dataset.cam_);
  
  svo::SegmentAbstractDetectorPtr segmenter =
      svo::feature_detection_utils::makeSegmentDetector(segmenter_options, dataset.cam_);

  
  svo::FramePtr frame;
  while(dataset.getNextFrame(5u, frame, nullptr))
  {

    Keypoints new_px;
    Levels new_levels;
    Scores new_scores;
    Gradients new_grads;
    FeatureTypes new_types;
    const size_t max_n_features = feature_detector_->grid_.size();
    feature_detector_->detect(
        frame->img_pyr_, frame->getMask(), max_n_features, new_px,
        new_scores, new_levels, new_grads, new_types);

    // linetodo
    Segments new_seg;
    Scores new_seg_score;
    FeatureTypes new_seg_types;
    Levels new_seg_levels;

    segment_detector_->detect(
        frame->img_pyr_, frame->getMask(), segment_detector_->options_.max_segment_num, new_seg,
        new_seg_score, new_seg_levels, new_seg_types);
    
    

    cv::imshow( "window_name", grad );
    cv::waitKey(0);
  }

  return 0;
}
