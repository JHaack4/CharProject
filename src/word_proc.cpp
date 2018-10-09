#include <cassert>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "image_proc.h"

#include <chrono>

/* This code is all unused */

int cnt = 0;

bool checkIfConnectedComponentsWorks(cv::Mat& img, MyLine& word) {

    bool streetName = word.isStreetName_;
    std::string name = std::to_string(cnt++);
    //std::cout << "WORD CHECK: " << name << std::endl;

    make_dir(DEBUG_PATH, "word_stuff");

    cv::Mat branch_points (img.rows, img.cols, CV_8UC1, img.type());
    cv::Mat thinned;
    thinning(img, thinned, branch_points, true);


    cv::Mat bgrFull;
    package_bgr({img & ~thinned, thinned & ~branch_points, branch_points}, bgrFull);

    cv::Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(thinned, labels, stats, centroids, 8);

    std::vector<cv::Point> centroid_list;
    bool ccOk = true;

    for (int i = 1; i < centroids.rows; ++i) {
        cv::Point pt {(int)centroids.at<double>(i,0), (int)centroids.at<double>(i,1)};

        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area <= 5 && streetName) continue; // allow dots? (makes false positives)

        cv::circle(bgrFull, pt, 1, CV_RGB(255,150,0), 1);

        if (area < 0.5 * img.rows) {
                ccOk = false; // something is too small here...
                //pln("small fails");
        }
        if ((streetName && area > 4.5 * img.rows) || (!streetName && area > 2.2 * img.rows)) {
                ccOk = false; // too large
                //pln("large fails");
        }
        //std::cout << " area" << i << ": " << area;

        centroid_list.push_back(pt);
    }

    specific_imwrite(img, "word_stuff", name + "_img");
    specific_imwrite(bgrFull, "word_stuff", name + "_skel");

    if ((!streetName && centroid_list.size() < 2) || (streetName && centroid_list.size() < 2)) {
            ccOk = false; // not enough cc's
            //pln("count fails");
            return false;
    }
    //std::cout << "cnt: " << centroid_list.size();

    MyLine line = fitPointCloud(centroid_list);
    line.mid_ = cv::Point2f{img.cols/2.0f, img.rows/2.0f};

    if ((!streetName && line.height_ > 0.2 * img.rows) || (streetName && line.height_ > 0.5 * img.rows)) {
            ccOk = false; // too uncorrelated
            //pln("height fails");
    }
    if ((!streetName && (line.domi_.y > 0.2 || line.domi_.y < -0.2))
        || (streetName && (line.domi_.y > 0.13 || line.domi_.y < -0.13))) {
            ccOk = false; // too slanted
            //pln("horizontal fails");
    }
    //std::cout << " height: " << line.height_;
    //std::cout << " domi: " << line.domi_.y;

    std::vector<double> new_x;
    for (size_t i = 0; i < centroid_list.size(); ++i) {
        double w = line.pointToNormal(centroid_list[i]);
        new_x.push_back(w);
    }

    std::sort(new_x.begin(), new_x.end());

    for (size_t i = 1; i < new_x.size(); ++i) {
        double diff_x = new_x[i] - new_x[i-1];

        //std::cout << " diff" << i << ": " << diff_x;

        if ((streetName && diff_x < 12) || (!streetName && diff_x < 6)) {
            ccOk = false;
            //pln("spacing narrow fails");
        }
        if ((streetName && diff_x > 50) || (!streetName && diff_x > 26)) {
            ccOk = false;
            //pln("spacing wide fails");
        }
    }

    if (ccOk) {
        word.connectedComponentPass_ = true;

        //pln("passing");
        if (DRAW_DEBUG_IMAGES) {
            cv::Mat colorCC;
            color_connected_components(img, colorCC);
            specific_imwrite(colorCC, "word_stuff", name + "_vgood");
        }
    }
    //std::cout << std::endl;

    return ccOk;
}





