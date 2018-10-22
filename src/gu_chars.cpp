/**
 * gu_chars.cpp
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
#include <math.h>
#include "image_proc.h"
#include "opencv2/opencv.hpp"

void find_best_representative_spectrum(std::vector<spectrum_t>& representatives, spectrum_t spectrum, int& outRep, int& outAngle) {

    float bestScore = 0;
    int bestRepIndex = 0;
    int bestAngle = 0;

    for (size_t j = 0; j < representatives.size(); j++) {
        spectrum_t repSpec = representatives[j];

        for (size_t i = 0; i < 180; i++) {
            float s = score_spectrum(spectrum, 0, repSpec, i);
            if (s > bestScore) {
                bestScore = s;
                bestRepIndex = j;
                bestAngle = i;
            }
        }
    }

    outRep = bestRepIndex;
    outAngle = bestAngle;

}

// compute the similarity between two spectrum using point-wise multiplication
float score_spectrum(std::vector<float>& spectrum1, int angle1, std::vector<float>& spectrum2, int angle2) {

    float score = 0;

    for (int i = 0; i < 180; i++) {
        //score += sqrt(spectrum1[(i+angle1)%180] * spectrum2[(i+angle2)%180]);
        score += 1 - std::abs(spectrum1[(i+angle1)%180] - spectrum2[(i+angle2)%180]);
    }

    return score;
}

// find the angle that aligns the two spectrum the most
float align_spectrum(std::vector<float>& spectrum1, std::vector<float>& spectrum2, int& outAngle) {

    float bestScore = 0;
    int bestAngle = 0;

    for (int i = 0; i < 180; i++) {
        float s = score_spectrum(spectrum1, 0, spectrum2, i);
        if (s > bestScore) {
            bestScore = s;
            bestAngle = i;
        }
    }

    outAngle = bestAngle;
    return bestScore;
}

// Take the wheel thing and output a spectrum. Uses a mask to deal with noise
void wheel_to_spectrum(std::vector<cv::Point2f>& wheel, std::vector<float>& spectrum, std::vector<float>& mask) {

    int wheelSize = 0;
    float maskSize = 0;

    for (cv::Point2f& p: wheel) {
        if (p != cv::Point2f{0,0})
            wheelSize++;
    }
    for (float f: mask) {
        maskSize += f;
    }
    if (maskSize == 0 || wheelSize == 0)
        return; // bad inputs

    for (size_t i = 0; i < wheel.size(); i++) {
        cv::Point2f& p = wheel[i];
        if (p == cv::Point2f{0,0})
            continue;

        float angleF = atan2(p.y, p.x) * 180.0f / 3.14159f;
        int angle = (int)(angleF + 0.5f) + 720; // ensure positive angle

        for (size_t j = 0; j < mask.size(); j++) {
            float f = mask[j];
            int index = angle + j - mask.size()/2;

            spectrum[index % 180] += f / maskSize / wheelSize;

        }
    }

}

// Take pixel chains and create a wheel
void pixel_chains_to_wheel(std::vector<std::vector<cv::Point>>& pixel_chains, std::vector<cv::Point2f>& wheel, float step_dist = 1.9f, bool avgOnly = false) {

    for (std::vector<cv::Point>& chain: pixel_chains) {
        wheel.push_back(cv::Point2f{0.0f,0.0f}); // separator between chains

        if (chain.size() < 1) continue;

        cv::Point2f avg = {(float)(chain[chain.size()-1].x - chain[0].x), (float)(chain[chain.size()-1].y - chain[0].y)};
        avg = avg / ((float)(cv::norm(avg)));

        cv::Point2f curSpot = chain[0];
        size_t target_idx = 1;

        while (target_idx < chain.size()) {
            cv::Point2f targetPoint = chain[target_idx];
            if (cv::norm(targetPoint - curSpot) < step_dist) {
                ++target_idx;
                continue;
            }

            cv::Point2f unit = targetPoint - curSpot;
            if (cv::norm(unit) == 0) {
                std::cout << "error, zero dist??" << std::endl;
                return;
            }
            unit = unit / ((float)cv::norm(unit));
            if (!avgOnly)
                wheel.push_back(unit);
            else
                wheel.push_back(avg);
            cv::Point2f dir = unit * step_dist;

            curSpot = curSpot + dir;
        }
    }

}

// take a binary image representing a character
// and return the spectrum of the char
void char_to_spectrum(cv::Mat& imBin, std::vector<float>& spectrum, std::vector<float>& mask, bool useAverages) {

    // Fill small holes in the image to avoid unnecessary artifacts in the thinned image.
    cv::Mat imHoles = imBin > 0;
    fill_holes(imHoles, 20);

    // Apply iterative thinning algorithm
    // Here, we used modified Zhang-Suen thinning
    // Why do we do two steps? - great question
    cv::Mat imThinRough;
    cv::Mat imThin;
    cv::Mat imBranchEndPoints;
    thinning5(imHoles, imThinRough, imBranchEndPoints, true, 20);
    thinning4(imThinRough, imThin, imBranchEndPoints, true, 5);

    // Turn skeleton into pixel chains
    std::vector< std::vector<cv::Point>> pixelChains = connectedComponentsSkel(imThin, imBranchEndPoints);

    // Vectorize the thinned image
    // This splits the pixel chains into even more pixel chains.
    // It also gives us a graph, where each line segment is an
    // edge, and each branch/end/corner point is a vertex.
    std::vector< std::vector<cv::Point>> vecPixelChains;
    std::vector<MyLine> vecLines;
    std::vector<MyVertex> vecVerts;
    segment_lines(pixelChains, vecPixelChains, vecLines, vecVerts, 1, 6, 0);

    // Find the wheel thing
    std::vector<cv::Point2f> wheel;
    if (useAverages) {
        std::vector<LongLine> vecLongLines;
        generate_long_lines(vecLines, vecLongLines);
        combine_long_lines(vecLines, vecVerts, vecLongLines, imBin,
                        /*vert*/ 2, /*dist*/ 1, /*matching_algorithm*/ MATCH_VECCHARS);

        for (size_t i = 0; i < vecLines.size(); ++i) {
            MyLine l = vecLines[i];
            LongLine ll = vecLongLines[l.long_line];
            cv::Point2f dir = ll.unitDirVector();

            for (int j = 0; j < l.width_ + 1; j++){
                wheel.push_back(dir);
            }
        }
    }

    else {
        pixel_chains_to_wheel(vecPixelChains, wheel, 1.9f, useAverages);
    }

    // create and draw spectrum from a wheel
    spectrum.clear();
    for (int i = 0; i < 180; ++i) spectrum.push_back(0.0f);

    wheel_to_spectrum(wheel, spectrum, mask);

}


// do some debugging stuff.
void look_at_char(std::string path_to_map) {

    bool DRAW_CHAR_IMAGES = true;

    if (DRAW_CHAR_IMAGES) make_dir(DEBUG_PATH, NAME);
    make_dir(DEBUG_PATH, "!Wheels!");
    log("Loading character from: ", LogLevel::debug);
    log(path_to_map, LogLevel::debug);

    // load an image
    cv::Mat raw_img = cv::imread(path_to_map, CV_LOAD_IMAGE_GRAYSCALE);
    if (path_to_map.find("representative") != std::string::npos) {
        raw_img = raw_img > 150; // deal with different formatting for average chars...
    }
    raw_img = raw_img == 0;
    if (DRAW_CHAR_IMAGES) debug_imwrite(~raw_img, "a1 Echo");

    // binarize image
    cv::Mat imBin;
    binarize(raw_img, imBin);
    if (path_to_map.find("representative") != std::string::npos) {
        cv::Mat imRot;
        cv_rotate(imBin, imRot, 30);
        //imBin = imRot;
    }
    if (DRAW_CHAR_IMAGES) debug_imwrite(imBin, "a2 Binary");

    // Fill small holes in the image to avoid unnecessary artifacts in the thinned image.
    cv::Mat imHoles = imBin > 0;
    fill_holes(imHoles, 20);
    if (DRAW_CHAR_IMAGES) debug_imwrite(imHoles, "a3 Holes Filled");

    // Apply iterative thinning algorithm
    // (Here, we used modified Zhang-Suen thinning
    cv::Mat imThin1;
    cv::Mat imThin;
    cv::Mat imBranchEndPoints;
    thinning5(imHoles, imThin1, imBranchEndPoints, true, 20);
    thinning4(imThin1, imThin, imBranchEndPoints, true, 5);

    cv::Mat bgrThin;
    package_bgr({(imBin & ~imThin) | imBranchEndPoints, imBin & ~imThin, imBin & ~imBranchEndPoints}, bgrThin);
    if (DRAW_CHAR_IMAGES) debug_imwrite(bgrThin, "a4 Thinned");
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
        if (DRAW_CHAR_IMAGES) debug_imwrite(bgrChainsPretty, "a5 PixelChains");
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
    segment_lines(pixelChains, vecPixelChains, vecLines, vecVerts, 2, 4, 0);
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
        if (DRAW_CHAR_IMAGES) debug_imwrite(bgrChainsPretty, "a6 VectorizedPixelChains2");
        specific_imwrite(bgrChainsPretty, "!Wheels!", "_1");

        package_bgr({imBin, imBin, imBin}, bgrBin);
        for (size_t i = 0; i < vecLines.size(); ++i) {
            MyLine l = vecLines[i];
            std::pair<cv::Point, cv::Point> endpts = l.endpoints();
            if (endpts.first != endpts.second)
                cv::line(bgrBin, endpts.first, endpts.second, id_color(l.line_id),2);
        }
        if (DRAW_CHAR_IMAGES) debug_imwrite(bgrBin, "a6 VectorizedLines2");
    }

    // Find the wheel thing
    std::vector<cv::Point2f> wheel;
    pixel_chains_to_wheel(vecPixelChains, wheel, 1.9f, false);
    std::vector<cv::Point2f> wheelLL;
    pixel_chains_to_wheel(vecPixelChains, wheelLL, 1.9f, true);

    // debug image for the wheel
    int imgSize = 200;
    cv::Mat wheel_img_base = cv::Mat::zeros(cv::Size{2*imgSize-1,2*imgSize-1}, CV_8UC1);
    cv::Mat wheel_img;
    package_bgr({wheel_img_base,wheel_img_base,wheel_img_base}, wheel_img);
    cv::circle(wheel_img, cv::Point{imgSize,imgSize}, imgSize-2, CV_RGB(255,255,255), 2);
    cv::circle(wheel_img, cv::Point{imgSize,imgSize}, 1, CV_RGB(255,255,255), 2);
    // draw the lines on the wheel
    int idColor = 0;
    for (size_t i = 0; i < wheel.size(); i++) {
        cv::Point2f pt = wheel[i];
        if (cv::norm(pt) == 0) {
            idColor++;
            continue;
        }
        cv::line(wheel_img, cv::Point{(int)(imgSize-(imgSize-3)*pt.x+0.5),(int)(imgSize-(imgSize-3)*pt.y+0.5)}, cv::Point{(int)(imgSize+(imgSize-3)*pt.x+0.5),(int)(imgSize+(imgSize-3)*pt.y+0.5)}, id_color(idColor), 1);
    }
    if (DRAW_CHAR_IMAGES) debug_imwrite(wheel_img, "b1 Wheel");
    specific_imwrite(wheel_img, "!Wheels!", "_3");


    std::vector<LongLine> vecLongLines;
    generate_long_lines(vecLines, vecLongLines);
    combine_long_lines(vecLines, vecVerts, vecLongLines, imBin,
                        /*vert*/ 2, /*dist*/ 1, /*matching_algorithm*/ MATCH_VECCHARS);
    log("Long line detection", LogLevel::debug);
    if (DRAW_DEBUG_IMAGES) {
        cv::Mat bgrThin2 = 0.2*bgrThin;
        for (size_t i = 0; i < vecLines.size(); ++i) {
            MyLine l = vecLines[i];
            std::pair<cv::Point, cv::Point> endpts = l.endpoints();
            if (endpts.first != endpts.second)
                cv::line(bgrThin2, endpts.first, endpts.second, id_color(l.long_line+1),2);
        }
        if (DRAW_CHAR_IMAGES) debug_imwrite(bgrThin2, "a8 Long lines");
        specific_imwrite(bgrThin2, "!Wheels!", "_2");
    }

    // debug image for the wheel
    int imgSizeLL = 200;
    cv::Mat wheel_img_baseLL = cv::Mat::zeros(cv::Size{2*imgSizeLL-1,2*imgSizeLL-1}, CV_8UC1);
    cv::Mat wheel_imgLL;
    package_bgr({wheel_img_baseLL,wheel_img_baseLL,wheel_img_baseLL}, wheel_imgLL);
    cv::circle(wheel_imgLL, cv::Point{imgSizeLL,imgSizeLL}, imgSizeLL-2, CV_RGB(255,255,255), 2);
    cv::circle(wheel_imgLL, cv::Point{imgSizeLL,imgSizeLL}, 1, CV_RGB(255,255,255), 2);
    // draw the lines on the wheel
    for (size_t i = 0; i < vecLongLines.size(); i++) {
        if (vecLongLines[i].ignore_me) continue;
        cv::Point2f pt = vecLongLines[i].unitDirVector();
        cv::line(wheel_imgLL, cv::Point{(int)(imgSize-(imgSize-3)*pt.x+0.5),(int)(imgSize-(imgSize-3)*pt.y+0.5)},
                 cv::Point{(int)(imgSize+(imgSize-3)*pt.x+0.5),(int)(imgSize+(imgSize-3)*pt.y+0.5)},
                 id_color(i+1), 1+(vecLongLines[i].length_/5));
    }
    if (DRAW_CHAR_IMAGES) debug_imwrite(wheel_imgLL, "b2 Wheel Avg");
    specific_imwrite(wheel_imgLL, "!Wheels!", "_4");


    // create and draw spectrum from a wheel
    std::vector<float> spectrum;
    for (int i = 0; i < 180; ++i) spectrum.push_back(0.0f);
    std::vector<float> mask = {1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1};
    wheel_to_spectrum(wheel, spectrum, mask);

    // create and draw spectrum from a wheel
    std::vector<float> spectrumLL;
    for (int i = 0; i < 180; ++i) spectrumLL.push_back(0.0f);
    std::vector<float> maskLL = {1,1,1,1,1,2,2,2,2,2,5,5,5,5,5,5,5,5,5,5,5,2,2,2,2,2,1,1,1,1,1};
    wheel_to_spectrum(wheelLL, spectrumLL, maskLL);


    // debug image for spectrum
    if (DRAW_DEBUG_IMAGES) {
        cv::Mat spec_img;
        package_bgr({wheel_img_base,wheel_img_base,wheel_img_base}, spec_img);
        cv::circle(spec_img, cv::Point{imgSize,imgSize}, imgSize-2, CV_RGB(255,255,255), 2);
        cv::circle(spec_img, cv::Point{imgSize,imgSize}, 1, CV_RGB(255,255,255), 2);
        for (int i = 0; i < 180; i++) {
            float angle = i / 180.0f * 3.14159f;
            cv::Point2f pt {cos(angle), sin(angle)};
            cv::line(spec_img, cv::Point{(int)(imgSize - imgSize*pt.x), (int)(imgSize - imgSize*pt.y)},
                 cv::Point{(int)(imgSize + imgSize*pt.x), (int)(imgSize + imgSize*pt.y)},
                 CV_RGB(std::min<int>(255, (int)(255*50*spectrum[i])), 0, 0), 1);
        }
        specific_imwrite(spec_img, "!Wheels!", "_5");
        if (DRAW_CHAR_IMAGES) debug_imwrite(spec_img, "b5 Spectrum");
    }
    // debug image for spectrum
    if (DRAW_DEBUG_IMAGES) {
        cv::Mat spec_img;
        package_bgr({wheel_img_base,wheel_img_base,wheel_img_base}, spec_img);
        cv::circle(spec_img, cv::Point{imgSize,imgSize}, imgSize-2, CV_RGB(255,255,255), 2);
        cv::circle(spec_img, cv::Point{imgSize,imgSize}, 1, CV_RGB(255,255,255), 2);
        for (int i = 0; i < 180; i++) {
            float angle = i / 180.0f * 3.14159f;
            cv::Point2f pt {cos(angle), sin(angle)};
            cv::line(spec_img, cv::Point{(int)(imgSize - imgSize*pt.x), (int)(imgSize - imgSize*pt.y)},
                 cv::Point{(int)(imgSize + imgSize*pt.x), (int)(imgSize + imgSize*pt.y)},
                 CV_RGB(0,std::min<int>(255, (int)(255*50*spectrumLL[i])), 0), 1);
        }
        specific_imwrite(spec_img, "!Wheels!", "_6");
        if (DRAW_CHAR_IMAGES) debug_imwrite(spec_img, "b5 SpectrumLL");
    }

    //float sum = 0;
    //for (int i = 0; i < 180; i++) {
    //    std::cout << i << " " << spectrum[i] << std::endl;
     //   sum += spectrum[i];
    //}
    //std::cout << "sum=" << sum << std::endl;

}



