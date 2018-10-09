/**
 * Run img_proc.cpp
 */
#include <sys/types.h>
#include <sys/stat.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include "image_proc.h"
#include "opencv2/opencv.hpp"

int main(int, char* argv[]) {

    log("Running...", LogLevel::debug);

    // Change this if you want to do something else
    std::string path_to_main_directory = "C:\\Users\\Jordan Haack\\Desktop\\CharProject2";
    NAME = "00065u";

    DRAW_DEBUG_IMAGES = true;
    std::string o = "output";
    std::string maps = "maps";
    DEBUG_PATH = path_to_main_directory + PATH_SEP + o;
    std::string path_to_map = path_to_main_directory + PATH_SEP + maps + PATH_SEP + NAME + ".tif";
    make_dir(DEBUG_PATH, NAME);
    log("Loading image from: ", LogLevel::debug);
    log(path_to_map, LogLevel::debug);

    // load an image
    cv::Mat raw_img = cv::imread(path_to_map, CV_LOAD_IMAGE_GRAYSCALE);
    debug_imwrite(~raw_img, "a1 Echo");
    log("Image loaded", LogLevel::debug);

    // binarize image
    cv::Mat imBin;
    binarize(raw_img, imBin);
    debug_imwrite(imBin, "a2 Binary");
    log("Image Binarized", LogLevel::debug);

    // Thin image
    // First, erode to remove large blocks (these regions take too long to thin)
    int erosion_size = 20;
    cv::Mat erosion(imBin.rows, imBin.cols, imBin.type(), 0.0);
    cv::Mat element = getStructuringElement(cv::MORPH_RECT,
              cv::Size(2 * (erosion_size/2) + 1, 2 * (erosion_size/2) + 1),
              cv::Point((erosion_size/2), (erosion_size/2)) );
    cv::erode(imBin, erosion, element);
    cv::Mat imBoundary;
    imBoundary = imBin & (~erosion);
    debug_imwrite(imBoundary, "a3 Boundary");
    log("Boundary Computed", LogLevel::debug);

    // Apply iterative thinning algorithm
    cv::Mat imThin;
    cv::Mat imBranchPoints;
    thinning5(imBoundary, imThin, imBranchPoints, true, 20);

    cv::Mat bgrThin;
    package_bgr({(imBin & ~imThin) | imBranchPoints, imBin & ~imThin, imBin & ~imBranchPoints}, bgrThin);
    debug_imwrite(bgrThin, "a4 Thinned");
    log("Image Thinned", LogLevel::debug);

    return 0;

}

/**
 * @brief      Given an image and boundary points, runs image processing
 *
 * @param[in]  argc
 * @param[in]  argv
 */
/*int main2(int, char* argv[])
{
    std::string options_file = argv[1];
    std::ifstream t(options_file, std::ifstream::binary);
    std::stringstream buffer;
    buffer << t.rdbuf();
    t.close();
    std::string bufferstr = buffer.str();

    // dumb hack to make it work
    //bufferstr = argv[2];

    std::string path_to_map;
    std::vector<cv::Point> boundary;
    std::vector<Street> streets;

    read_args(bufferstr, path_to_map, boundary, streets);

    log("Processing " + NAME, LogLevel::debug);

    // Directories
    make_dir(DEBUG_PATH, NAME);
    make_dir(MIDWAY_PATH, NAME);
    make_dir(DEBUG_PATH, "!!cropped_digits");
    make_dir(DEBUG_PATH, "!!cropped_letters");
    make_dir(DEBUG_PATH, "!!cropped_words");

    // load the image
    cv::Mat img = cv::imread(path_to_map, CV_LOAD_IMAGE_GRAYSCALE);
    log("Image Loaded", LogLevel::debug);

    process_image(img, boundary, streets);

    dump_json();
    return 0;
}*/


