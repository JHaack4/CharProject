#include <cassert>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "image_proc.h"
#include <sys/types.h>
#include <sys/stat.h>

// Creates an BGR image of 3 layers
void package_bgr(const std::vector<cv::Mat>& layers, cv::Mat& output)
{
    if (!DRAW_DEBUG_IMAGES) return;

    assert(layers.size() <= 3);

    output = cv::Mat(layers[0].size(), CV_8UC3, 0.0);
    const int pairs[6] = {0,0, 1,1 , 2,2}; // mapping of channels from input to output
    cv::mixChannels(layers, output, pairs, layers.size());
}

// Creates a connected components image for the input
// background, rangeMin, and rangeMax specify colors of the output image.
void color_connected_components(const cv::Mat& input, cv::Mat& output,
                                cv::Vec3b background, cv::Vec3b rangeMin, cv::Vec3b rangeMax)
{
    if (!DRAW_DEBUG_IMAGES) return;

    cv::Mat cc_labels;
    size_t cc_num = cv::connectedComponents(input, cc_labels);

    std::vector<cv::Vec3b> random_colors;
    for (size_t i = 0; i < cc_num; ++i) {
        pixel_t b = rand() % (rangeMax[0] - rangeMin[0] + 1) + rangeMin[0];
        pixel_t g = rand() % (rangeMax[1] - rangeMin[1] + 1) + rangeMin[1];
        pixel_t r = rand() % (rangeMax[2] - rangeMin[2] + 1) + rangeMin[2];
        if (i == 0) {
            random_colors.push_back(background);
        }
        else {
            random_colors.push_back(cv::Vec3b(b,g,r));
        }
    }

    // create the output image
    output = cv::Mat(cc_labels.size(), CV_8UC3);
    for (int row = 0; row < output.rows; ++row) {
        for (int col = 0; col < output.cols; ++col) {
            size_t cc = cc_labels.at<uint>(row, col);
            output.at<cv::Vec3b>(row, col) = random_colors[cc];
        }
    }
}

void draw_pixel_chains(cv::Mat& imBin, std::vector< std::vector<cv::Point>>& pixelChains, cv::Mat& output) {

    if (!DRAW_DEBUG_IMAGES) return;

    output = cv::Mat::zeros(imBin.size(), CV_32SC1);

    for (size_t i = 0; i < pixelChains.size(); ++i) {
        for (size_t j = 0; j < pixelChains[i].size(); ++j) {
            output.at<int>(pixelChains[i][j]) = (int)(i+1);
        }
    }
}

std::vector<cv::Vec3b> random_colors2;

// Give a deterministic color to each integer.
cv::Vec3b id_color(int i) {

    if (random_colors2.size() < 1) {
        srand(31);
        std::vector<cv::Vec3b> random_colors;
        for (size_t ii = 0; ii < 1000; ++ii) {
            pixel_t b = rand() % (255 - 50 + 1) + 50;
            pixel_t g = rand() % (255 - 50 + 1) + 50;
            pixel_t r = rand() % (255 - 50 + 1) + 50;
            if (ii == 0) {
                random_colors2.push_back({50,50,50});
            }
            else {
                random_colors2.push_back(cv::Vec3b(b,g,r));
            }
        }
    }
    return random_colors2[i%1000];
}

// Creates a colored label image
// background, rangeMin, and rangeMax specify colors of the output image.
void color_labels(const cv::Mat& input, cv::Mat& output,
                  cv::Vec3b background, cv::Vec3b rangeMin, cv::Vec3b rangeMax)
{
    if (!DRAW_DEBUG_IMAGES) return;

    srand(31);
    std::vector<cv::Vec3b> random_colors;
    for (size_t i = 0; i < 1000; ++i) {
        pixel_t b = rand() % (rangeMax[0] - rangeMin[0] + 1) + rangeMin[2];
        pixel_t g = rand() % (rangeMax[1] - rangeMin[1] + 1) + rangeMin[1];
        pixel_t r = rand() % (rangeMax[2] - rangeMin[2] + 1) + rangeMin[0];
        if (i == 0) {
            random_colors.push_back(background);
        }
        else {
            random_colors.push_back(cv::Vec3b(b,g,r));
        }
    }

    // create the output image
    output = cv::Mat(input.size(), CV_8UC3);
    for (int row = 0; row < output.rows; ++row) {
        for (int col = 0; col < output.cols; ++col) {
            uint label = input.at<uint>(row, col);
            output.at<cv::Vec3b>(row, col) = random_colors[label % 1000];
        }
    }
}

// Writes an image to the debug folder, in a sub-folder for this map.
// This lets us debug one map at a time
void debug_imwrite(const cv::Mat& img, const std::string& title) {
    if (!DRAW_DEBUG_IMAGES) return;

    std::string img_path = DEBUG_PATH + PATH_SEP + NAME + PATH_SEP + title + ".png";
    if (!cv::imwrite(img_path, img)) {
        std::cout << "failed to write image" << std::endl;
    }
}

// Writes an image to the debug folder in a folder with the name type.
// this lets us debug one algorithm at a time.
void specific_imwrite(const cv::Mat& img, const std::string& type, const std::string& info) {
    if (!DRAW_DEBUG_IMAGES) return;

    std::string img_path = DEBUG_PATH + PATH_SEP + type + PATH_SEP + NAME + info + ".png";
    if (!cv::imwrite(img_path, img)) {
        std::cout << "failed to write image" << std::endl;
    }
}

// Writes an image to the midway folder. This means that our python code
// will read it. (I.e it is more useful than a debug image)
void midway_imwrite(const cv::Mat& img, const std::string& title) {
    std::string img_path = MIDWAY_PATH + PATH_SEP + NAME + PATH_SEP + title + ".png";
    if (!cv::imwrite(img_path, img)) {
        std::cout << "failed to write image" << std::endl;
    }
}

// Make a folder at some path.
void make_dir(const std::string& path, const std::string& name)
{
    std::string dir = path + PATH_SEP + name;
    #ifdef _WIN32
    mkdir(dir.c_str());
    #else
    mkdir(dir.c_str() ,  S_IRWXU | S_IRWXG | S_IRWXO);
    #endif
}
