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

    std::cout << "Running..." << std::endl;

    cv::Mat img = cv::imread("maps" + std::to_string(PATH_SEP) + "00065.tif", CV_LOAD_IMAGE_GRAYSCALE);

    return 0;

}

/**
 * @brief      Given an image and boundary points, runs image processing
 *
 * @param[in]  argc
 * @param[in]  argv
 */
int main2(int, char* argv[])
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
}


