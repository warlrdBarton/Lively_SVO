#ifndef OPENCV_LINE_MATCH_HPP
#define OPENCV_LINE_MATCH_HPP
#include <future>
#include <thread>
#include <time.h>
#include <set>
#include <utility>
using namespace std;
#include <image_geometry/stereo_camera_model.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/fast_line_detector.hpp>

#include "descriptor_custom.hpp"
#include "auxiliar.h"
#include "stereoFeatures.h"
#include "matching.h"
#include <yaml-cpp/yaml.h>

using namespace cv;
using namespace line_descriptor;

#include <eigen3/Eigen/Core>
#include <svo/common/point.h>

using namespace Eigen;

#define GRID_ROWS 48
#define GRID_COLS 64

typedef Matrix<double, 6, 6> Matrix6d;
typedef Matrix<double, 6, 1> Vector6d;

namespace svo
{

    typedef std::pair<int, int> point_2d;
    typedef std::pair<point_2d, point_2d> line_2d;

    void detectLineFeatures(const cv::Mat &img, std::vector<KeyLine> &lines, Mat &ldesc, double min_line_length);
    void matchLineFeatures(BFMatcher *bfm, Mat ldesc_1, Mat ldesc_2, std::vector<std::vector<DMatch>> &lmatches_12);
    void filterLineSegmentDisparity(Vector2d spl, Vector2d epl, Vector2d spr, Vector2d epr, double &disp_s, double &disp_e);
    double lineSegmentOverlapStereo(double spl_obs, double epl_obs, double spl_proj, double epl_proj);
    void region2Mat(cv::Mat &P, const cv::Mat seq, int seq_rows, int seq_cols, int t = -1, int l = -1, int mat_rows = -1, int mat_cols = -1);

    void invert_pose(cv::Mat &out_r, cv::Mat &out_t, const cv::Mat &r, const cv::Mat &t);

    void cross(cv::Mat &out_r, cv::Mat &out_t, cv::Mat &dest);
    sensor_msgs::CameraInfo getCameraInfo(cv::Mat k, cv::Mat d, cv::Mat r, cv::Mat p, std::string model, int h, int w);
    void readSVOCameraParamFile(const std::string &path, std::vector<sensor_msgs::CameraInfo> &camera);

    class line_match
    {
    public:
        line_match()
        {
            std::string camera_param_path = conf.camera_model_param_yaml_path;
            if (camera_param_path.size() == 0)
            {
                camera_param_path = "/home/sunteng/catkin_ws/src/rpg_svo_pro_open/svo_ros/param/calib/realsense_stereo_best.yaml";
            }
            std::vector<sensor_msgs::CameraInfo> camera_infos_;
            readSVOCameraParamFile(camera_param_path, camera_infos_);

            model_.fromCameraInfo(camera_infos_[0], camera_infos_[1]);
        };
        Config conf;
        image_geometry::StereoCameraModel model_;
        void matchStereoLines(const vector<KeyLine> &lines_l, const vector<KeyLine> &lines_r, Mat &ldesc_l_, const Mat &ldesc_r_,  std::vector<int>& matches_12,std::vector<shared_ptr<svo::Line>>& lines_3d);
    };
}

#endif
