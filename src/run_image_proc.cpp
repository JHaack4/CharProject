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



int main(int argc, char* argv[]) {

    log("Running...", LogLevel::debug);

    // Change this if you want to do something else
    std::string path_to_main_directory = "C:\\Users\\Jordan Haack\\Desktop\\CharProject2";
    //NAME = "00065u";
    NAME = "00017u-1951-Reel_24-V.1-LOC-36029-Buffalo-NY_w1093_c0_A";

    DRAW_DEBUG_IMAGES = true;
    std::string ots = "output";
    std::string datas = "data";
    std::string let = "letter";
    std::string rlet = "representative_letters";
    std::string avg = "average_";
    DEBUG_PATH = path_to_main_directory + PATH_SEP + ots;
    make_dir(DEBUG_PATH, "!RotatedChars!");

    if (argc > 1) {
        path_to_main_directory = argv[1];
    }
    if (argc > 2) {
        NAME = argv[2];
    }

    // mask to use for creating the spectrum
    //std::vector<float> mask = {1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1};
    std::vector<float> mask = {1,1,1,1,1,2,2,2,2,2,5,5,5,5,5,5,5,5,5,5,5,2,2,2,2,2,1,1,1,1,1};
    bool useAverages = true;




    std::vector<spectrum_t> represetative_spectrum;

    // generate the set of representatives
    for (int i = 0; i < 26; ++i) {
        char ch = (char)(i + 'A');
        std::string path_to_char = path_to_main_directory + PATH_SEP + datas + PATH_SEP + rlet + PATH_SEP + avg + ch + ".png";

        cv::Mat raw_img = cv::imread(path_to_char, CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat imBin;
        raw_img = raw_img < 220;
        binarize(raw_img, imBin);

        spectrum_t spec;
        char_to_spectrum(imBin, spec, mask, useAverages);
        represetative_spectrum.push_back(spec);
    }

    std::vector<int> anglesPos = {0,5,30};
    for (int angle: anglesPos)
    for (int i = 0; i < 26; ++i) {
        char ch = (char)(i + 'A');
        std::string path_to_char = path_to_main_directory + PATH_SEP + datas + PATH_SEP + rlet + PATH_SEP + avg + ch + ".png";

        cv::Mat raw_img = cv::imread(path_to_char, CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat imBin;
        raw_img = raw_img < 220;
        binarize(raw_img, imBin);
        cv::Mat imRot;
        //int angle = 0;
        cv_rotate(imBin, imRot, angle);
        std::string strch(1,ch);
        specific_imwrite(imRot, "!RotatedChars!", "_" +strch+std::to_string(angle));

        spectrum_t spec;
        char_to_spectrum(imRot, spec, mask, useAverages);

        if (DRAW_DEBUG_IMAGES) {
            int imgSize = 200;
            cv::Mat wheel_img_base = cv::Mat::zeros(cv::Size{2*imgSize-1,2*imgSize-1}, CV_8UC1);
            cv::Mat spec_img;
            package_bgr({wheel_img_base,wheel_img_base,wheel_img_base}, spec_img);
            cv::circle(spec_img, cv::Point{imgSize,imgSize}, imgSize-2, CV_RGB(255,255,255), 2);
            cv::circle(spec_img, cv::Point{imgSize,imgSize}, 1, CV_RGB(255,255,255), 2);
            for (int ii = 0; ii < 180; ii++) {
                float angles = ii / 180.0f * 3.14159f;
                cv::Point2f pt {cos(angles), sin(angles)};
                cv::line(spec_img, cv::Point{(int)(imgSize - imgSize*pt.x), (int)(imgSize - imgSize*pt.y)},
                    cv::Point{(int)(imgSize + imgSize*pt.x), (int)(imgSize + imgSize*pt.y)},
                    CV_RGB(std::min<int>(255, (int)(255*50*spec[ii])), 0, 0), 1);
            }
            specific_imwrite(spec_img, "!RotatedChars!", "_" +strch+std::to_string(angle)+"_s");
        }

        int bestRep;
        int bestAngle;
        find_best_representative_spectrum(represetative_spectrum, spec, bestRep, bestAngle);
        std::cout << (char)('A'+bestRep) << " angle=" << bestAngle << " best rep for " << ch << "/" << angle << std::endl;
    }




    // If you want to analyze the representative chars
    /*for (int i = 0; i < 26; ++i) {
        char ch = (char)(i + 'A');
        std::string avg2 = "average_";
        NAME = avg2 + ch;

        std::string path_to_map = path_to_main_directory + PATH_SEP + datas + PATH_SEP + rlet + PATH_SEP + NAME + ".png";
        look_at_char(path_to_map);
    }*/

//    // if you want to analyze a single char...
//    look_at_char(path_to_main_directory);


    if (1>0) return 0;

// ---------------------------------------------------------------------------------------

    DRAW_DEBUG_IMAGES = true;
    std::string o = "output";
    std::string maps = "maps";
    DEBUG_PATH = path_to_main_directory + PATH_SEP + o;
    std::string path_to_map = path_to_main_directory + PATH_SEP + maps + PATH_SEP + NAME + ".tif";
    make_dir(DEBUG_PATH, NAME);
    log("Loading image from: ", LogLevel::debug);
    log(path_to_map, LogLevel::debug);

    // what is the threshold for lengths of lines to ignore?
    const int LINE_IGNORE_LENGTH_THRESHOLD = 70; // in pixels

    // load an image
    cv::Mat raw_img = cv::imread(path_to_map, CV_LOAD_IMAGE_GRAYSCALE);
    debug_imwrite(~raw_img, "a1 Echo");
    log("Image loaded", LogLevel::debug);

    // binarize image
    cv::Mat imBin;
    binarize(raw_img, imBin);
    debug_imwrite(imBin, "a2 Binary");
    log("Image Binarized", LogLevel::debug);

    // First, erode to remove large blocks (these regions take too long to thin)
    // If large blocks of white pixels are in the image then
    // we will have to iterate a number of times on the order of their
    // radius. To prevent this, we delete the interiors
    int erosion_radius = 10; // radius of erosion
    cv::Mat erosion(imBin.rows, imBin.cols, imBin.type(), 0.0);
    cv::Mat element = getStructuringElement(cv::MORPH_RECT,
              cv::Size(2 * erosion_radius - 1, 2 * erosion_radius - 1),
              cv::Point(erosion_radius, erosion_radius) );
    cv::erode(imBin, erosion, element);
    cv::Mat imBoundary;
    imBoundary = imBin & (~erosion); // Subtract the opening of the image
    debug_imwrite(imBoundary, "a3 Boundary");
    log("Boundary Computed", LogLevel::debug);

    // Apply iterative thinning algorithm
    // (Here, we used modified Zhang-Suen thinning
    cv::Mat imThin;
    cv::Mat imBranchEndPoints;
    thinning5(imBoundary, imThin, imBranchEndPoints, true, 20);

    cv::Mat bgrThin;
    package_bgr({(imBin & ~imThin) | imBranchEndPoints, imBin & ~imThin, imBin & ~imBranchEndPoints}, bgrThin);
    debug_imwrite(bgrThin, "a4 Thinned");
    log("Image Thinned", LogLevel::debug);


    // Turn skeleton into pixel chains (might have to deal with adding branch points)
    std::vector< std::vector<cv::Point>> pixelChains = connectedComponentsSkel(imThin, imBranchEndPoints);
    if (DRAW_DEBUG_IMAGES) {
        cv::Mat imChainLabels;
        cv::Mat bgrChains;
        cv::Mat bgrBin;
        cv::Mat bgrChainsPretty;
        draw_pixel_chains(imBin, pixelChains, imChainLabels);
        color_labels(imChainLabels, bgrChains);
        package_bgr({imBin, imBin, imBin}, bgrBin);
        bgrChainsPretty = 0.2*bgrBin + 0.8*bgrChains;
        debug_imwrite(bgrChainsPretty, "a5 PixelChains");
    }
    log("Pixel Chains found", LogLevel::debug);
    log("Number of pixel chains: " + std::to_string(pixelChains.size()), LogLevel::debug);


    // Vectorize the thinned image
    // This splits the pixel chains into even more pixel chains.
    // It also gives us a graph, where each line segment is an
    // edge, and each branch/end/corner point is a vertex.
    std::vector< std::vector<cv::Point>> vecPixelChains;
    std::vector<MyLine> vecLines;
    std::vector<MyVertex> vecVerts;
    segment_lines(pixelChains, vecPixelChains, vecLines, vecVerts, 3, 7, 0);
    log("Lines Vectorized", LogLevel::debug);
    if (DRAW_DEBUG_IMAGES) {
        cv::Mat imChainLabels;
        cv::Mat bgrChains;
        cv::Mat bgrBin;
        cv::Mat bgrChainsPretty;
        draw_pixel_chains(imBin, vecPixelChains, imChainLabels);
        color_labels(imChainLabels, bgrChains);
        package_bgr({imBin, imBin, imBin}, bgrBin);
        bgrChainsPretty = 0.2*bgrBin + 0.8*bgrChains;
        debug_imwrite(bgrChainsPretty, "a6 VectorizedPixelChains2");

        for (size_t i = 0; i < vecLines.size(); ++i) {
            MyLine l = vecLines[i];
            std::pair<cv::Point, cv::Point> endpts = l.endpoints();
            if (endpts.first != endpts.second)
                cv::line(bgrThin, endpts.first, endpts.second, id_color(l.line_id),2);
        }
        debug_imwrite(bgrThin, "a6 VectorizedLines2");
        log("--Debug image", LogLevel::debug);
    }


    // Determine which pixel chain owns each pixel in original image
    std::vector< std::vector<cv::Point>> pixelChainOwners;
    pixel_chain_owners(imBin, vecPixelChains, pixelChainOwners, 20);
    log("Pixel Owners Determined", LogLevel::debug);
    if (DRAW_DEBUG_IMAGES) {
        cv::Mat labels;
        cv::Mat colorOwners;
        draw_pixel_chains(imBin, pixelChainOwners, labels);
        color_labels(labels, colorOwners);
        debug_imwrite(colorOwners, "a7 Pixel Chain Owners");
        log("--Debug image", LogLevel::debug);
    }


    // Detect long lines (aka line continuations aka poly-lines)
    // in the image. Separate them from the rest of the image,
    // which contains the symbols we care about.
    std::vector<LongLine> vecLongLines;
    generate_long_lines(vecLines, vecLongLines);
    combine_long_lines(vecLines, vecVerts, vecLongLines, imBin,
                        /*vert*/ 2, /*dist*/ 1, /*matching_algorithm*/ MATCH_VECLINES);
    log("Long line detection", LogLevel::debug);
    if (DRAW_DEBUG_IMAGES) {
        for (size_t i = 0; i < vecLines.size(); ++i) {
            MyLine l = vecLines[i];
            std::pair<cv::Point, cv::Point> endpts = l.endpoints();
            if (endpts.first != endpts.second)
                cv::line(bgrThin, endpts.first, endpts.second, id_color(l.long_line),2);
        }
        debug_imwrite(bgrThin, "a8 Long lines");
        log("--Debug image", LogLevel::debug);
    }

    // generate labels for pixels that are not part of a long line
    if (DRAW_DEBUG_IMAGES) {
        cv::Mat labels = cv::Mat::zeros(imBin.size(), CV_32SC1);
        cv::Mat bgrL;

        for (size_t i = 0; i < pixelChainOwners.size(); ++i) {
            LongLine& ll = vecLongLines[vecLines[i].long_line];
            if (ll.length_ > LINE_IGNORE_LENGTH_THRESHOLD) continue;
            for (size_t j = 0; j < pixelChainOwners[i].size(); ++j) {
                labels.at<int>(pixelChainOwners[i][j]) = (int)(i+1);
            }
        }
        color_labels(labels, bgrL);
        debug_imwrite(bgrL, "a8 markings ownership");
        log("--Debug image", LogLevel::debug);
    }

    // TODO: compute connected components of small components
    // and then draw/crop/output those

    // will need to write a matching algorithm (also, matches vectors at close
    // distance. needs to avoid all long lines

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


