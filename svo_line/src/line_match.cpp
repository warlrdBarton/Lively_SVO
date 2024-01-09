
#include <svo/line/line_match.h>

#include <svo/line/auxiliar.h>

namespace svo
{
    void region2Mat(cv::Mat &P, const cv::Mat seq, int seq_rows, int seq_cols, int t , int l , int mat_rows , int mat_cols )
    {
        if (l != -1)
        {
            int idx = 0;
            int re = l + mat_cols - 1;
            for (int i = t * seq_cols + l; i <  seq_rows * seq_cols; i++)
            {
                if (i % seq_cols < l)
                {
                    i = (i - i % seq_cols) + l;
                }
                if (i % seq_cols > re)
                {
                    i = (i / seq_cols + 1) * seq_cols + l;
                }
                if (idx >= mat_cols * mat_rows)
                {
                    return;
                }
                P.at<double>(idx / mat_cols, idx % mat_cols) = seq.at<double>(i / seq_cols, i % seq_cols);
                idx++;
            }
        }
        else
        {
            for (int i = 0; i < seq_rows * seq_cols; i++)
            {
                P.at<double>(i / seq_cols, i % seq_cols) = seq.at<double>(i / seq_cols, i % seq_cols);
            }
        }
    }

    void invert_pose(cv::Mat &out_r, cv::Mat &out_t, const cv::Mat &r, const cv::Mat &t)
    {
        out_r = r.t();
        out_t = -t;
    }

    void cross(cv::Mat &out_r, cv::Mat &out_t, cv::Mat &dest)
    {
        cv::Mat hat_mat = (cv::Mat_<double>(3, 3) << 0, -1 * out_r.at<double>(2, 0), out_r.at<double>(1, 0), out_r.at<double>(2, 0), 0, -1 * out_r.at<double>(0, 0), -1 * out_r.at<double>(1, 0), out_r.at<double>(0, 0), 0);
        dest = hat_mat * out_t;
    }

    sensor_msgs::CameraInfo getCameraInfo(cv::Mat k, cv::Mat d, cv::Mat r, cv::Mat p, std::string model, int h, int w)
    {
        sensor_msgs::CameraInfo cam;
        std::vector<double> D{d.at<double>(0), d.at<double>(1), d.at<double>(2), d.at<double>(3), 0};
        boost::array<double, 9> K = {
            k.at<double>(0, 0),
            k.at<double>(0, 1),
            k.at<double>(0, 2),
            k.at<double>(1, 0),
            k.at<double>(1, 1),
            k.at<double>(1, 2),
            k.at<double>(2, 0),
            k.at<double>(2, 1),
            k.at<double>(2, 2),
        };

        // get rectified projection.
        boost::array<double, 12> P = {
            p.at<double>(0, 0), p.at<double>(0, 1), p.at<double>(0, 2), p.at<double>(0, 3),
            p.at<double>(1, 0), p.at<double>(1, 1), p.at<double>(1, 2), p.at<double>(1, 3),
            p.at<double>(2, 0), p.at<double>(2, 1), p.at<double>(2, 2), p.at<double>(2, 3)};
        boost::array<double, 9> R = {
            r.at<double>(0, 0),
            r.at<double>(0, 1),
            r.at<double>(0, 2),
            r.at<double>(1, 0),
            r.at<double>(1, 1),
            r.at<double>(1, 2),
            r.at<double>(2, 0),
            r.at<double>(2, 1),
            r.at<double>(2, 2),
        };

        cam.height = h;
        cam.width = w;
        cam.distortion_model = model;
        cam.D = D;
        cam.K = K;
        cam.P = P;
        cam.R = R;
        cam.binning_x = 1;
        cam.binning_y = 1;
        return cam;
    }
    void readSVOCameraParamFile(const std::string &path, std::vector<sensor_msgs::CameraInfo> &camera)
    {
        cv::Mat P0, P1, R0, R1, K0, K1, D0, D1;
        sensor_msgs::CameraInfo pcam1, pcam0;
        cv::Mat pose_w_cam0, pose_w_cam1;
        cv::Mat k_seq_0, k_seq_1;
        std::string model_dist;
        try
        {
            YAML::Node config = YAML::LoadFile(path);

            std::vector<double> T_B_C0_data = config["cameras"][0]["T_B_C"]["data"].as<std::vector<double>>();
            pose_w_cam0 = (cv::Mat_<double>(3, 4) << T_B_C0_data[0], T_B_C0_data[1], T_B_C0_data[2], T_B_C0_data[3],
                           T_B_C0_data[4], T_B_C0_data[5], T_B_C0_data[6], T_B_C0_data[7],
                           T_B_C0_data[8], T_B_C0_data[9], T_B_C0_data[10], T_B_C0_data[11]);

            std::vector<double> T_B_C1_data = config["cameras"][1]["T_B_C"]["data"].as<std::vector<double>>();
            pose_w_cam1 = (cv::Mat_<double>(3, 4) << T_B_C1_data[0], T_B_C1_data[1], T_B_C1_data[2], T_B_C1_data[3],
                           T_B_C1_data[4], T_B_C1_data[5], T_B_C1_data[6], T_B_C1_data[7],
                           T_B_C1_data[8], T_B_C1_data[9], T_B_C1_data[10], T_B_C1_data[11]);

            model_dist = config["cameras"][1]["camera"]["distortion"]["type"].as<std::string>();

            std::vector<double> intrinsics0 = config["cameras"][0]["camera"]["intrinsics"]["data"].as<std::vector<double>>();
            k_seq_0 = (cv::Mat_<double>(1, 4) << intrinsics0[0], intrinsics0[1], intrinsics0[2], intrinsics0[3]);

            // 读取相机0的畸变参数
            std::vector<double> distortion0 = config["cameras"][0]["camera"]["distortion"]["parameters"]["data"].as<std::vector<double>>();
            D0 = (cv::Mat_<double>(1, 4) << distortion0[0], distortion0[1], distortion0[2], distortion0[3]);

            // 读取相机1的内参
            std::vector<double> intrinsics1 = config["cameras"][1]["camera"]["intrinsics"]["data"].as<std::vector<double>>();
            k_seq_1 = (cv::Mat_<double>(1, 4) << intrinsics1[0], intrinsics1[1], intrinsics1[2], intrinsics1[3]);

            // 读取相机1的畸变参数
            std::vector<double> distortion1 = config["cameras"][1]["camera"]["distortion"]["parameters"]["data"].as<std::vector<double>>();
            D1 = (cv::Mat_<double>(1, 4) << distortion1[0], distortion1[1], distortion1[2], distortion1[3]);
            // 其他参数的读取...

            std::cout << "D1" << std::endl;
        }
        catch (YAML::Exception &e)
        {
            std::cerr << "YAML error: " << e.what() << std::endl;
        }
        catch (std::exception &e)
        {
            std::cerr << "stander error: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "unknown error" << std::endl;
        }
        cv::Mat k_cam0 = (cv::Mat_<double>(3, 3) << k_seq_0.at<double>(0, 0), 0, k_seq_0.at<double>(0, 2), 0, k_seq_0.at<double>(0, 1), k_seq_0.at<double>(0, 3), 0, 0, 1);
        cv::Mat k_cam1 = (cv::Mat_<double>(3, 3) << k_seq_1.at<double>(0, 0), 0, k_seq_1.at<double>(0, 2), 0, k_seq_1.at<double>(0, 1), k_seq_1.at<double>(0, 3), 0, 0, 1); // 两个相机的内参
        cv::Mat r_w_cam0(3, 3, CV_64FC1);
        cv::Mat t_w_cam0(3, 1, CV_64FC1);
        cv::Mat r_w_cam1(3, 3, CV_64FC1);
        cv::Mat t_w_cam1(3, 1, CV_64FC1);
        cv::Mat z_cam0(3, 1, CV_64FC1);
        cv::Mat z_cam1(3, 1, CV_64FC1);
        region2Mat(r_w_cam0, pose_w_cam0, 3, 4, 0, 0, 3, 3);
        region2Mat(t_w_cam0, pose_w_cam0, 3, 4, 0, 3, 3, 1);
        region2Mat(z_cam0, pose_w_cam0, 3, 4, 0, 2, 3, 1);
        region2Mat(r_w_cam1, pose_w_cam1, 3, 4, 0, 0, 3, 3);
        region2Mat(t_w_cam1, pose_w_cam1, 3, 4, 0, 3, 3, 1);
        region2Mat(z_cam1, pose_w_cam1, 3, 4, 0, 2, 3, 1);

        cv::Mat baseline = t_w_cam1 - t_w_cam0; // 得到基线的向量，求取一个垂直于基线的平面

        cv::Mat new_x = baseline / norm(baseline); // 得到新的x轴，长度为1
        cv::Mat middle = (z_cam0 + z_cam1) / 2;    // 取两个坐标轴的中间值，将该点投影到平面
        middle = middle / norm(middle);
        cv::Mat middle_projection_2_plane = middle.t() * new_x; // 点乘，两个值需要长度都为1
        // ROS_INFO_STREAM("CAMERA"<<'\n'<<z_cam0 <<'\n'<<z_cam0<<"\n"<<middle_projection_2_plane);
        cv::Mat new_z = middle - middle_projection_2_plane.at<double>(0, 0) * new_x; // 获得为归一化的新z轴
        new_z = new_z / norm(new_z);
        // ROS_INFO_STREAM("middle_projection_2_plane"<<middle_projection_2_plane);
        // ROS_INFO_STREAM("middle"<<middle);
        // ROS_INFO_STREAM("new_x"<<new_x);
        // ROS_INFO_STREAM("new_z"<<new_z);

        cv::Mat new_y;
        cross(new_z, new_x, new_y);
        new_y = new_y / norm(new_y); // 获得y轴
        // ROS_INFO_STREAM("norm"<<new_x.mul(new_y)<<new_x.mul(new_z)<<new_z.mul(new_y));
        std::vector<cv::Mat> vImgs;
        cv::Mat result;
        vImgs.push_back(new_x);
        vImgs.push_back(new_y);
        vImgs.push_back(new_z);
        hconcat(vImgs, result); // 存储的是新平面的世界坐标系下的位姿
        // ROS_INFO_STREAM("new pose"<<result);//得到一个新的位姿矩阵
        //  ROS_INFO_STREAM("norm(new_x.mul(new_z))"<<norm(new_x.mul(new_z)));
        //  ROS_INFO_STREAM("norm(new_x.mul(new_y))"<<norm(new_x.mul(new_y)));
        //  ROS_INFO_STREAM("norm(new_y.mul(new_z))"<<norm(new_y.mul(new_z)));
        R0 = result.t() * r_w_cam0; // 相机1到新平面的旋转
        R1 = result.t() * r_w_cam1;
        // ROS_INFO_STREAM("result"<<result<<" "<<result.t());
        cv::Mat diagm = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        // R0=diagm;
        // R1=diagm;}
        vImgs.clear();
        vImgs.push_back(diagm); // 新平面下的世界坐标位姿
        vImgs.push_back(cv::Mat::zeros(3, 1, CV_64FC1));
        hconcat(vImgs, P0);
        P0 = k_cam0 * P0;
        P0.copyTo(P1);
        P1.at<double>(0, 3) = -k_cam0.at<double>(0, 0) * norm(t_w_cam1 - t_w_cam0);
        // ROS_INFO_STREAM("R0"<<R0<<"\nR1"<<R1);
        // ROS_INFO_STREAM("P0"<<P0<<"\nP1"<<P1);

        // Convert OpenCV image to ROS message
        pcam0 = getCameraInfo(k_cam0, D0, R0, P0, model_dist, 480, 640);
        pcam1 = getCameraInfo(k_cam1, D1, R1, P1, model_dist, 480, 640);
        camera.push_back(pcam0);
        camera.push_back(pcam1);
    }
    void detectLineFeatures(const cv::Mat &img, std::vector<KeyLine> &lines, Mat &ldesc, double min_line_length)
    {
        // Detect line features
        lines.clear();
        Ptr<BinaryDescriptor> lbd = BinaryDescriptor::createBinaryDescriptor();
        if (Config::hasLines())
        {

            if (!Config::useFLDLines())
            {
                Ptr<line_descriptor::LSDDetectorC> lsd = line_descriptor::LSDDetectorC::createLSDDetectorC();
                // lsd parameters
                line_descriptor::LSDDetectorC::LSDOptions opts;
                opts.refine = Config::lsdRefine();
                opts.scale = Config::lsdScale();
                opts.sigma_scale = Config::lsdSigmaScale();
                opts.quant = Config::lsdQuant();
                opts.ang_th = Config::lsdAngTh();
                opts.log_eps = Config::lsdLogEps();
                opts.density_th = Config::lsdDensityTh();
                opts.n_bins = Config::lsdNBins();
                opts.min_length = min_line_length;
                lsd->detect(img, lines, Config::lsdScale(), 1, opts);
                // filter lines
                if (static_cast<int>(lines.size()) > Config::lsdNFeatures() && Config::lsdNFeatures() != 0)
                {
                    // sort lines by their response
                    sort(lines.begin(), lines.end(), sort_lines_by_response());
                    // sort( lines.begin(), lines.end(), sort_lines_by_length() );
                    lines.resize(Config::lsdNFeatures());
                    // reassign index
                    for (int i = 0; i < Config::lsdNFeatures(); i++)
                        lines[i].class_id = i;
                }
                lbd->compute(img, lines, ldesc);
            }
            else
            {
                cv::Mat fld_img, img_gray;
                vector<Vec4f> fld_lines;

                if (img.channels() != 1)
                {
                    cv::cvtColor(img, img_gray, cv::COLOR_RGB2GRAY);
                    img_gray.convertTo(fld_img, CV_8UC1);
                }
                else
                    img.convertTo(fld_img, CV_8UC1);

                Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector(min_line_length);
                fld->detect(fld_img, fld_lines);

                // filter lines
                if (static_cast<int>(fld_lines.size()) > Config::lsdNFeatures() && Config::lsdNFeatures() != 0)
                {
                    // sort lines by their response
                    sort(fld_lines.begin(), fld_lines.end(), sort_flines_by_length());
                    fld_lines.resize(Config::lsdNFeatures());
                }

                // loop over lines object transforming into a vector<KeyLine>
                lines.reserve(fld_lines.size());
                for (size_t i = 0; i < fld_lines.size(); i++)
                {
                    KeyLine kl;
                    double octaveScale = 1.f;
                    int octaveIdx = 0;

                    kl.startPointX = fld_lines[i][0] * octaveScale;
                    kl.startPointY = fld_lines[i][1] * octaveScale;
                    kl.endPointX = fld_lines[i][2] * octaveScale;
                    kl.endPointY = fld_lines[i][3] * octaveScale;

                    kl.sPointInOctaveX = fld_lines[i][0];
                    kl.sPointInOctaveY = fld_lines[i][1];
                    kl.ePointInOctaveX = fld_lines[i][2];
                    kl.ePointInOctaveY = fld_lines[i][3];

                    kl.lineLength = (float)sqrt(pow(fld_lines[i][0] - fld_lines[i][2], 2) + pow(fld_lines[i][1] - fld_lines[i][3], 2));

                    kl.angle = atan2((kl.endPointY - kl.startPointY), (kl.endPointX - kl.startPointX));
                    kl.class_id = i;
                    kl.octave = octaveIdx;
                    kl.size = (kl.endPointX - kl.startPointX) * (kl.endPointY - kl.startPointY);
                    kl.pt = Point2f((kl.endPointX + kl.startPointX) / 2, (kl.endPointY + kl.startPointY) / 2);

                    kl.response = kl.lineLength / max(fld_img.cols, fld_img.rows);
                    cv::LineIterator li(fld_img, Point2f(fld_lines[i][0], fld_lines[i][1]), Point2f(fld_lines[i][2], fld_lines[i][3]));
                    kl.numOfPixels = li.count;

                    lines.push_back(kl);
                }

                // compute lbd descriptor
                lbd->compute(fld_img, lines, ldesc);
            }
        }
    }

    void matchLineFeatures(BFMatcher *bfm, Mat ldesc_1, Mat ldesc_2, std::vector<std::vector<DMatch>> &lmatches_12)
    {
        bfm->knnMatch(ldesc_1, ldesc_2, lmatches_12, 2);
    }

    void filterLineSegmentDisparity(Vector2d spl, Vector2d epl, Vector2d spr, Vector2d epr, double &disp_s, double &disp_e)
    {
        disp_s = spl(0) - spr(0);
        disp_e = epl(0) - epr(0);
        // if they are too different, ignore them
        if (min(disp_s, disp_e) / max(disp_s, disp_e) < Config::lsMinDispRatio())
        {
            disp_s = -1.0;
            disp_e = -1.0;
        }
    }

    double lineSegmentOverlapStereo(double spl_obs, double epl_obs, double spl_proj, double epl_proj)
    {

        double overlap = 1.f;

        if (fabs(epl_obs - spl_obs) > Config::lineHorizTh()) // normal lines (verticals included)
        {
            double sln = min(spl_obs, epl_obs);
            double eln = max(spl_obs, epl_obs);
            double spn = min(spl_proj, epl_proj);
            double epn = max(spl_proj, epl_proj);

            double length = eln - spn;

            if ((epn < sln) || (spn > eln))
                overlap = 0.f;
            else
            {
                if ((epn > eln) && (spn < sln))
                    overlap = eln - sln;
                else
                    overlap = min(eln, epn) - max(sln, spn);
            }

            if (length > 0.01f)
                overlap = overlap / length;
            else
                overlap = 0.f;

            if (overlap > 1.f)
                overlap = 1.f;
        }

        return overlap;
    }

    void line_match::matchStereoLines(const vector<KeyLine> &lines_l, const vector<KeyLine> &lines_r, Mat &ldesc_l_, const Mat &ldesc_r_,  std::vector<int>& goal,std::vector<shared_ptr<svo::Line>>& lines_3d)
    {

        // Line segments stereo matching
        // --------------------------------------------------------------------------------------------------------------------
        size_t inv_width = GRID_COLS / static_cast<double>(640);
        size_t inv_height = GRID_ROWS / static_cast<double>(480);

        std::vector<std::shared_ptr<svo::LineFeature>> stereo_ls;
        if (!Config::hasLines() || lines_l.empty() || lines_r.empty())
            return;

        std::vector<line_2d> coords;
        coords.reserve(lines_l.size());
        for (const KeyLine &kl : lines_l)
            coords.push_back(std::make_pair(std::make_pair(kl.startPointX * inv_width, kl.startPointY * inv_height),
                                            std::make_pair(kl.endPointX * inv_width, kl.endPointY * inv_height)));

        // Fill in grid & directions
        std::list<std::pair<int, int>> line_coords; // the coordonate of all point in this line
        GridStructure grid(GRID_ROWS, GRID_COLS);

        std::vector<std::pair<double, double>> directions(lines_r.size()); // get the direction of each line in two frame

        for (size_t idx = 0; idx < lines_r.size(); ++idx)
        {
            const KeyLine &kl = lines_r[idx];

            std::pair<double, double> &v = directions[idx];
            v = std::make_pair((kl.endPointX - kl.startPointX) * inv_width, (kl.endPointY - kl.startPointY) * inv_height);
            normalize(v);

            getLineCoords(kl.startPointX * inv_width, kl.startPointY * inv_height, kl.endPointX * inv_width, kl.endPointY * inv_height, line_coords);
            for (const std::pair<int, int> &p : line_coords)
                grid.at(p.first, p.second).push_back(idx);
        } // get all the line

        GridWindow w;
        w.width = std::make_pair(Config::matchingSWs(), 0);
        w.height = std::make_pair(0, 0);

       std::vector<int> matches_12;
        matchGrid(coords, ldesc_l_, grid, ldesc_r_, directions, w, matches_12);
        //    match(ldesc_l, ldesc_r, Config::minRatio12P(), matches_12);
        // bucle around lmatches
        Mat ldesc_l_aux;
        int ls_idx = 0;
        for (int i1 = 0; i1 < static_cast<int>(matches_12.size()); ++i1)
        {
            const int i2 = matches_12[i1];
            if (i2 < 0)
                continue;

            // estimate the disparity of the endpoints
            Eigen::Vector3d sp_l;
            sp_l << lines_l[i1].startPointX, lines_l[i1].startPointY, 1.0;
            Eigen::Vector3d ep_l;
            ep_l << lines_l[i1].endPointX, lines_l[i1].endPointY, 1.0;
            Eigen::Vector3d le_l;
            le_l << sp_l.cross(ep_l);
            le_l = le_l / std::sqrt(le_l(0) * le_l(0) + le_l(1) * le_l(1));
            Eigen::Vector3d sp_r;
            sp_r << lines_r[i2].startPointX, lines_r[i2].startPointY, 1.0;
            Eigen::Vector3d ep_r;
            ep_r << lines_r[i2].endPointX, lines_r[i2].endPointY, 1.0;
            Eigen::Vector3d le_r;
            le_r << sp_r.cross(ep_r);

            double overlap = lineSegmentOverlapStereo(sp_l(1), ep_l(1), sp_r(1), ep_r(1));

            double disp_s, disp_e;
            sp_r << (sp_r(0) * (sp_l(1) - ep_r(1)) + ep_r(0) * (sp_r(1) - sp_l(1))) / (sp_r(1) - ep_r(1)), sp_l(1), 1.0;
            ep_r << (sp_r(0) * (ep_l(1) - ep_r(1)) + ep_r(0) * (sp_r(1) - ep_l(1))) / (sp_r(1) - ep_r(1)), ep_l(1), 1.0;
            filterLineSegmentDisparity(sp_l.head(2), ep_l.head(2), sp_r.head(2), ep_r.head(2), disp_s, disp_e);

            // check minimal disparity
            if (disp_s >= Config::minDisp() && disp_e >= Config::minDisp() && std::abs(sp_l(1) - ep_l(1)) > Config::lineHorizTh() && std::abs(sp_r(1) - ep_r(1)) > Config::lineHorizTh() && overlap > Config::stereoOverlapTh())
            {

                cv::Point2d left_uv_s(sp_l(0), sp_l(1));
                cv::Point2d left_uv_e(ep_l(0), ep_l(1));
                cv::Point3d sP_cv, eP_cv;
                model_.projectDisparityTo3d(left_uv_s, disp_s, sP_cv);
                model_.projectDisparityTo3d(left_uv_e, disp_e, eP_cv);

                    // sP_ = cam->backProjection(sp_l(0), sp_l(1), disp_s);
                Vector3d sP_(sP_cv.x, sP_cv.y, sP_cv.z);

                Vector3d eP_(eP_cv.x, eP_cv.y, eP_cv.z);
                // eP_ = cam->backProjection(ep_l(0), ep_l(1), disp_e);
                double angle_l = lines_l[i1].angle;

                ldesc_l_aux.push_back(ldesc_l_.row(i1));
                lines_3d.push_back(std::make_shared<svo::Line>( sP_, eP_));
                goal.push_back(i2);
                
            }
        }

        ldesc_l_aux.copyTo(ldesc_l_);
    }


    // void detectStereoLineSegments(double llength_th, const cv::Mat &img_l, const cv::Mat &img_r, const int frame_idx)
    // {

    //     // detect and estimate each descriptor for both the left and right image
    //     cv::Mat ldesc_l, ldesc_r;
    //     std::vector<KeyLine> lines_l, lines_r;
    //     if (0)
    //     {
    //         auto detect_l = async(launch::async, &detectLineFeatures,this, img_l, ref(lines_l), ref(ldesc_l), llength_th);
    //         auto detect_r = async(launch::async, &detectLineFeatures,this, img_r, ref(lines_r), ref(ldesc_r), llength_th);
    //         detect_l.wait();
    //         detect_r.wait();
    //     }
    //     else
    //     {
    //         detectLineFeatures(img_l, lines_l, ldesc_l, llength_th);
    //         detectLineFeatures(img_r, lines_r, ldesc_r, llength_th);
    //     }

    //     // perform the stereo matching
    //     matchStereoLines(lines_l, lines_r, ldesc_l, ldesc_r, (frame_idx == 0));
    // }
}
