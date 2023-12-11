
#include <svo/line/line_match.h>

void detectStereoLineSegments(double llength_th)
{

    // detect and estimate each descriptor for both the left and right image
    if(0)
    {
        auto detect_l = async(launch::async, &StereoFrame::detectLineFeatures, this, img_l, ref(lines_l), ref(ldesc_l), llength_th );
        auto detect_r = async(launch::async, &StereoFrame::detectLineFeatures, this, img_r, ref(lines_r), ref(ldesc_r), llength_th );
        detect_l.wait();
        detect_r.wait();
    }
    else
    {
        detectLineFeatures( img_l, lines_l, ldesc_l, llength_th );
        detectLineFeatures( img_r, lines_r, ldesc_r, llength_th );
    }

    // perform the stereo matching
    matchStereoLines(lines_l,  lines_r,  ldesc_l, ldesc_r, (frame_idx==0));

}

void detectLineFeatures( Mat img, vector<KeyLine> &lines, Mat &ldesc, double min_line_length )
{

    // Detect line features
    lines.clear();
    Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();
    if( Config::hasLines() )
    {

        if( !Config::useFLDLines() )
        {
            Ptr<line_descriptor::LSDDetectorC> lsd = line_descriptor::LSDDetectorC::createLSDDetectorC();
            // lsd parameters
            line_descriptor::LSDDetectorC::LSDOptions opts;
            opts.refine       = Config::lsdRefine();
            opts.scale        = Config::lsdScale();
            opts.sigma_scale  = Config::lsdSigmaScale();
            opts.quant        = Config::lsdQuant();
            opts.ang_th       = Config::lsdAngTh();
            opts.log_eps      = Config::lsdLogEps();
            opts.density_th   = Config::lsdDensityTh();
            opts.n_bins       = Config::lsdNBins();
            opts.min_length   = min_line_length;
            lsd->detect( img, lines, Config::lsdScale(), 1, opts);
            // filter lines
            if( lines.size()>Config::lsdNFeatures() && Config::lsdNFeatures()!=0  )
            {
                // sort lines by their response
                sort( lines.begin(), lines.end(), sort_lines_by_response() );
                //sort( lines.begin(), lines.end(), sort_lines_by_length() );
                lines.resize(Config::lsdNFeatures());
                // reassign index
                for( int i = 0; i < Config::lsdNFeatures(); i++  )
                    lines[i].class_id = i;
            }
            lbd->compute( img, lines, ldesc);
        }
        else
        {
            Mat fld_img, img_gray;
            vector<Vec4f> fld_lines;

            if( img.channels() != 1 )
            {
                cv::cvtColor( img, img_gray, CV_RGB2GRAY );
                img_gray.convertTo( fld_img, CV_8UC1 );
            }
            else
                img.convertTo( fld_img, CV_8UC1 );

            Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector(min_line_length);
            fld->detect( fld_img, fld_lines );

            // filter lines
            if( fld_lines.size()>Config::lsdNFeatures() && Config::lsdNFeatures()!=0  )
            {
                // sort lines by their response
                sort( fld_lines.begin(), fld_lines.end(), sort_flines_by_length() );
                fld_lines.resize(Config::lsdNFeatures());
            }

            // loop over lines object transforming into a vector<KeyLine>
            lines.reserve(fld_lines.size());
            for( int i = 0; i < fld_lines.size(); i++ )
            {
                KeyLine kl;
                double octaveScale = 1.f;
                int    octaveIdx   = 0;

                kl.startPointX     = fld_lines[i][0] * octaveScale;
                kl.startPointY     = fld_lines[i][1] * octaveScale;
                kl.endPointX       = fld_lines[i][2] * octaveScale;
                kl.endPointY       = fld_lines[i][3] * octaveScale;

                kl.sPointInOctaveX = fld_lines[i][0];
                kl.sPointInOctaveY = fld_lines[i][1];
                kl.ePointInOctaveX = fld_lines[i][2];
                kl.ePointInOctaveY = fld_lines[i][3];

                kl.lineLength = (float) sqrt( pow( fld_lines[i][0] - fld_lines[i][2], 2 ) + pow( fld_lines[i][1] - fld_lines[i][3], 2 ) );

                kl.angle    = atan2( ( kl.endPointY - kl.startPointY ), ( kl.endPointX - kl.startPointX ) );
                kl.class_id = i;
                kl.octave   = octaveIdx;
                kl.size     = ( kl.endPointX - kl.startPointX ) * ( kl.endPointY - kl.startPointY );
                kl.pt       = Point2f( ( kl.endPointX + kl.startPointX ) / 2, ( kl.endPointY + kl.startPointY ) / 2 );

                kl.response = kl.lineLength / max( fld_img.cols, fld_img.rows );
                cv::LineIterator li( fld_img, Point2f( fld_lines[i][0], fld_lines[i][1] ), Point2f( fld_lines[i][2], fld_lines[i][3] ) );
                kl.numOfPixels = li.count;

                lines.push_back( kl );

            }

            // compute lbd descriptor
            lbd->compute( fld_img, lines, ldesc);
        }

    }
}

void StereoFrame::matchStereoLines( vector<KeyLine> lines_l, vector<KeyLine> lines_r, Mat &ldesc_l_, Mat ldesc_r, bool initial )
{

    // Line segments stereo matching
    // --------------------------------------------------------------------------------------------------------------------
    stereo_ls.clear();
    if (!Config::hasLines() || lines_l.empty() || lines_r.empty())
        return;

    std::vector<line_2d> coords;
    coords.reserve(lines_l.size());
    for (const KeyLine &kl : lines_l)
        coords.push_back(std::make_pair(std::make_pair(kl.startPointX * inv_width, kl.startPointY * inv_height),
                                        std::make_pair(kl.endPointX * inv_width, kl.endPointY * inv_height)));

    //Fill in grid & directions
    list<pair<int, int>> line_coords;//the coordonate of all point in this line
    GridStructure grid(GRID_ROWS, GRID_COLS);

    std::vector<std::pair<double, double>> directions(lines_r.size());//get the direction of each line in two frame
    
    for (int idx = 0; idx < lines_r.size(); ++idx) {
        const KeyLine &kl = lines_r[idx];

        std::pair<double, double> &v = directions[idx];
        v = std::make_pair((kl.endPointX - kl.startPointX) * inv_width, (kl.endPointY - kl.startPointY) * inv_height);
        normalize(v);

        getLineCoords(kl.startPointX * inv_width, kl.startPointY * inv_height, kl.endPointX * inv_width, kl.endPointY * inv_height, line_coords);
        for (const std::pair<int, int> &p : line_coords)
            grid.at(p.first, p.second).push_back(idx);
    }//get all the line 

    GridWindow w;
    w.width = std::make_pair(Config::matchingSWs(), 0);
    w.height = std::make_pair(0, 0);

    std::vector<int> matches_12;
    matchGrid(coords, ldesc_l, grid, ldesc_r, directions, w, matches_12);
//    match(ldesc_l, ldesc_r, Config::minRatio12P(), matches_12);
    // bucle around lmatches
    Mat ldesc_l_aux;
    int ls_idx = 0;
    for (int i1 = 0; i1 < matches_12.size(); ++i1) {
        const int i2 = matches_12[i1];
        if (i2 < 0) continue;

        // estimate the disparity of the endpoints
        Vector3d sp_l; sp_l << lines_l[i1].startPointX, lines_l[i1].startPointY, 1.0;
        Vector3d ep_l; ep_l << lines_l[i1].endPointX,   lines_l[i1].endPointY,   1.0;
        Vector3d le_l; le_l << sp_l.cross(ep_l); le_l = le_l / std::sqrt( le_l(0)*le_l(0) + le_l(1)*le_l(1) );
        Vector3d sp_r; sp_r << lines_r[i2].startPointX, lines_r[i2].startPointY, 1.0;
        Vector3d ep_r; ep_r << lines_r[i2].endPointX,   lines_r[i2].endPointY,   1.0;
        Vector3d le_r; le_r << sp_r.cross(ep_r);

        double overlap = lineSegmentOverlapStereo( sp_l(1), ep_l(1), sp_r(1), ep_r(1) );

        double disp_s, disp_e;
        sp_r << ( sp_r(0)*( sp_l(1) - ep_r(1) ) + ep_r(0)*( sp_r(1) - sp_l(1) ) ) / ( sp_r(1)-ep_r(1) ) , sp_l(1) ,  1.0;
        ep_r << ( sp_r(0)*( ep_l(1) - ep_r(1) ) + ep_r(0)*( sp_r(1) - ep_l(1) ) ) / ( sp_r(1)-ep_r(1) ) , ep_l(1) ,  1.0;
        filterLineSegmentDisparity( sp_l.head(2), ep_l.head(2), sp_r.head(2), ep_r.head(2), disp_s, disp_e );

        // check minimal disparity
        if( disp_s >= Config::minDisp() && disp_e >= Config::minDisp()
            && std::abs( sp_l(1)-ep_l(1) ) > Config::lineHorizTh()
            && std::abs( sp_r(1)-ep_r(1) ) > Config::lineHorizTh()
            && overlap > Config::stereoOverlapTh() )
        {
            Vector3d sP_; sP_ = cam->backProjection( sp_l(0), sp_l(1), disp_s);
            Vector3d eP_; eP_ = cam->backProjection( ep_l(0), ep_l(1), disp_e);
            double angle_l = lines_l[i1].angle;
            if( initial )
            {
                ldesc_l_aux.push_back( ldesc_l_.row(i1) );
                stereo_ls.push_back( new LineFeature(Vector2d(sp_l(0),sp_l(1)),disp_s,sP_,
                                                     Vector2d(ep_l(0),ep_l(1)),disp_e,eP_,
                                                     le_l,angle_l,ls_idx,lines_l[i1].octave) );
                ls_idx++;
            }
            else
            {
                ldesc_l_aux.push_back( ldesc_l_.row(i1) );
                stereo_ls.push_back( new LineFeature(Vector2d(sp_l(0),sp_l(1)),disp_s,sP_,
                                                     Vector2d(ep_l(0),ep_l(1)),disp_e,eP_,
                                                     le_l,angle_l,-1,lines_l[i1].octave) );
            }
        }
    }

    ldesc_l_aux.copyTo(ldesc_l_);
}

void matchLineFeatures(BFMatcher* bfm, Mat ldesc_1, Mat ldesc_2, vector<vector<DMatch>> &lmatches_12  )
{
    bfm->knnMatch( ldesc_1, ldesc_2, lmatches_12, 2);
}

void filterLineSegmentDisparity( Vector2d spl, Vector2d epl, Vector2d spr, Vector2d epr, double &disp_s, double &disp_e )
{
    disp_s = spl(0) - spr(0);
    disp_e = epl(0) - epr(0);
    // if they are too different, ignore them
    if(  min( disp_s, disp_e ) / max( disp_s, disp_e ) < Config::lsMinDispRatio() )
    {
        disp_s = -1.0;
        disp_e = -1.0;
    }
}