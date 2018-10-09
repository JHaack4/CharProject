#include <iostream>
#include <ctime>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "image_proc.h"

// ------------- MASK, BINARIZE, and MARKINGS/LINES ------------------

// Get an image for the mask provided
void get_mask(cv::Mat& mask, const std::vector<cv::Point> & pts)
{
    std::vector<std::vector<cv::Point>> polys{pts};     // fillPoly takes in a list of polygons
    cv::fillPoly(mask, polys, 255);
}

// Binarize the image using an adaptive thresholding algorithm
void binarize(cv::Mat& img, cv::Mat& out)
{
    const size_t kernel_size = 101;
    const int offset = 40;
    const int ignore_thresh = 200;

    // Apply an initial threshold to ignore extremely light pixels
    cv::Mat thresh;
    cv::threshold(img, thresh, ignore_thresh, 255, img.type());

    // Apply adaptive threshold
    cv::adaptiveThreshold(img, out, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
        kernel_size, offset);

    out = out & ~thresh;
}

// Sort the black pixels in the image into lines and markings, based on their shapes
void remove_markings(cv::Mat& img, cv::Mat& lines, cv::Mat& mark,
                     std::vector<cv::Point>& marking_centroids, std::vector<int>& marking_sizes)
{

    // These parameters have been tested.
    const size_t AREA_MAX_LIMIT              = 3500;
    const size_t WIDTH_MAX_LIMIT             = 140;
    const size_t HEIGHT_MAX_LIMIT            = 105;
    const double DENSITY_MIN_LIMIT           = 0.35;
    const double ASPECT_RATIO_MAX_LIMIT      = 13.0;

    const double ASPECT_RATIO_MIN_DASH_LIMIT = 4.0;
    const double MIN_DASH_LENGTH = 10.0;
    const double MAX_DASH_HEIGHT = 8.0;
    const double MIN_DASH_VARIANCE = 2.4;

    // Apply connected components to the black pixels
    cv::Mat labels;
    std::vector<std::vector<cv::Point>> points_list = connectedComponentsWithIdx(img, labels);

    // Iterate over each component.
    for (size_t i = 1; i < points_list.size(); ++i) {
        std::vector<cv::Point> points = points_list[i];

        size_t area = points.size();
        if (area > AREA_MAX_LIMIT) {
            // If the component is really big, we know it is a line,
            // and we can ignore the fitting phase.
            for (cv::Point p: points)
                lines.at<pixel_t>(p) = 255;
            continue;
        } else if (area < 6) {
            continue; // ignore, too small
        }

        // Find the best fit line for the point cloud
        MyLine line = fitPointCloud(points);

        // Compute metrics.
        double density = (area * area * 1.0) / (line.varW_ * line.varH_);
        double aspectRatio = line.width_ * 1.0 / line.height_;

        // Determine if the connected component is a marking or a line,
        // based on various qualities.
        bool isMarking = line.width_ < WIDTH_MAX_LIMIT
                        && line.height_ < HEIGHT_MAX_LIMIT
                        && density > DENSITY_MIN_LIMIT
                        && aspectRatio < ASPECT_RATIO_MAX_LIMIT;

        // This algorithm can also be used to detect dashes.
        bool isDash    = isMarking
                        && aspectRatio > ASPECT_RATIO_MIN_DASH_LIMIT
                        && line.width_ > MIN_DASH_LENGTH
                        && (line.varH_ < MIN_DASH_VARIANCE || line.height_ < MAX_DASH_HEIGHT);

        // Fill in the output images
        if (isMarking) // marking
            for (cv::Point p: points)
                mark.at<pixel_t>(p) = 255;
        if (!isMarking || isDash) // line
            for (cv::Point p: points)
                lines.at<pixel_t>(p) = 255;

        // Record these for our curb line detector
        if (isMarking && area > 40) {
            marking_centroids.push_back(line.mid_);
            marking_sizes.push_back((int)area);
        }

    }

}

// Use flood filling to find connected components of the white pixels that
// are significantly large. These components tend to correspond with the streets.
void floodfill_streets(cv::Mat& img, cv::Mat& mask, cv::Mat& streets)
{

    const size_t AREA_CUTOFF             = 1500000 * STREET_SCALE_FACTOR * STREET_SCALE_FACTOR;
    const double LARGE_DENSITY_MAX_LIMIT = 0.8;
    const size_t WIDTH_MIN_LIMIT         = 1200   * STREET_SCALE_FACTOR;
    const size_t HEIGHT_MAX_LIMIT        = 600    * STREET_SCALE_FACTOR;
    const size_t HEIGHT_MIN_LIMIT        = 100    * STREET_SCALE_FACTOR;
    const double DENSITY_MIN_LIMIT       = 0.4;
    const double ASPECT_RATIO_MIN_LIMIT  = 10.0;
    const size_t SMALL_AREA_MIN_LIMIT    = 100000 * STREET_SCALE_FACTOR * STREET_SCALE_FACTOR;

    // Connected components of the whitespace
    cv::Mat labels;
    cv::Mat not_img = ~img;
    std::vector<std::vector<cv::Point>> points_list = connectedComponentsWithIdx(not_img, labels);

    // For each connected component
    for (size_t i = 1; i < points_list.size(); ++i) {
        std::vector<cv::Point> points = points_list[i];

        size_t area = points.size();
        if (area <= SMALL_AREA_MIN_LIMIT) continue; // too small

        // Find the best fit line for the point cloud
        MyLine line = fitPointCloud(points);

        // Compute metrics
        double densitySq = (area * area * 1.0) / (256 * line.varW_ * line.varH_);
        double aspectRatioSq = line.varW_ * 1.0 / line.varH_;

        // Determine if this connected component is a street,
        // based on these qualities.
        bool isStreet = (area > AREA_CUTOFF && densitySq < LARGE_DENSITY_MAX_LIMIT) ||
                        (area <= AREA_CUTOFF && densitySq > DENSITY_MIN_LIMIT
                         && line.width_ > WIDTH_MIN_LIMIT && line.varH_ < HEIGHT_MAX_LIMIT
                         && line.height_ > HEIGHT_MIN_LIMIT && aspectRatioSq > ASPECT_RATIO_MIN_LIMIT
                         && area > SMALL_AREA_MIN_LIMIT);

        // Fill in the image
        if (isStreet) // street
            for (cv::Point p: points)
                streets.at<pixel_t>(p) = 255;

    }

    // Blur slightly to cut across lines
    cv::blur(streets, streets, cv::Size(8,8));

    // Apply the mask
    bitwise_and(streets, mask, streets);

    // Erode to remove large blocks
    int erosion_size = 70;
    cv::Mat erosion(img.rows, img.cols, img.type(), 0.0);
    cv::Mat element = getStructuringElement(cv::MORPH_RECT,
              cv::Size(2 * (erosion_size/2) + 1, 2 * (erosion_size/2) + 1),
              cv::Point((erosion_size/2), (erosion_size/2)) );
    cv::erode(streets, erosion, element);
    streets = streets & (~erosion);

    // Close to remove small holes
    int close_size = 10;
    cv::Mat element2 = getStructuringElement(cv::MORPH_RECT,
              cv::Size(2 * (close_size /2) + 1, 2 * (close_size /2) + 1),
              cv::Point((close_size /2), (close_size /2)) );
    cv::morphologyEx(streets, streets, cv::MORPH_CLOSE, element2);

    // Threshold to make the image binary again
    cv::threshold(streets, streets, 1, 255, CV_8UC1);

}

// --------------- MAIN IMAGE PROC FUNCTION ------------
// This is where the magic happens
void process_image(cv::Mat& imRaw, const std::vector<cv::Point>& boundary, const std::vector<Street>& present_streets)
{
    // Here is a key for how variable names correspond to the sizes of the image.
    // imImg = large image
    // smImg = downscaled image
    // miniImg = further downscaled image
    // ownImg = tracks ownership of street pixels (also downscaled)

    const int R = imRaw.rows, C = imRaw.cols, T = imRaw.type();
    const int ownR = (int)(R*OWNERSHIP_SCALE_FACTOR), ownC = (int)(C*OWNERSHIP_SCALE_FACTOR);
    log("Image size: " + std::to_string(R) + ", " + std::to_string(C), LogLevel::debug);

    // Binarize image
    cv::Mat imBin;
    binarize(imRaw, imBin);
    log("Image binarized", LogLevel::debug);

    // Find the mask of the image
    cv::Mat mask(R, C, T, 0.0);
    get_mask(mask, boundary);
    log("Image masked", LogLevel::debug);

    // try running and timing Z-S thinning here ??

    //imBin = cv::imread("C:\\Users\\Jordan Haack\\Desktop\\RaD-HMC-2018\\testimg.png", CV_LOAD_IMAGE_GRAYSCALE);
    //imBin = imBin < 1;

    log("thin5 st", LogLevel::debug);
    cv::Mat imZSThin5;
    cv::Mat imZSBP5;
    thinning5(imBin, imZSThin5, imZSBP5);
    log("thin5 end", LogLevel::debug);

    cv::Mat bgrzs5;
    package_bgr({(imBin & ~imZSThin5) | imZSBP5, imBin & ~imZSThin5, imBin & ~imZSBP5}, bgrzs5);
    debug_imwrite(bgrzs5, "t1a_szraw5");

    log("thin4 st", LogLevel::debug);
    cv::Mat imZSThin4;
    cv::Mat imZSBP4;
    thinning4(imBin, imZSThin4, imZSBP4);
    log("thin4 end", LogLevel::debug);

    cv::Mat bgrzs4;
    package_bgr({(imBin & ~imZSThin4) | imZSBP4, imBin & ~imZSThin4, imBin & ~imZSBP4}, bgrzs4);
    debug_imwrite(bgrzs4, "t1a_szraw4");

    /*log("thin3 st", LogLevel::debug);
    cv::Mat imZSThin3;
    cv::Mat imZSBP3;
    thinning3(imBin, imZSThin3, imZSBP3);
    log("thin3 end", LogLevel::debug);

    cv::Mat bgrzs3;
    package_bgr({(imBin & ~imZSThin3) | imZSBP3, imBin & ~imZSThin3, imBin & ~imZSBP3}, bgrzs3);
    debug_imwrite(bgrzs3, "t1a_szraw3");

    log("thin2 st", LogLevel::debug);
    cv::Mat imZSThin2;
    cv::Mat imZSBP2;
    thinning2(imBin, imZSThin2, imZSBP2);
    log("thin2 end", LogLevel::debug);

    cv::Mat bgrzs2;
    package_bgr({(imBin & ~imZSThin2) | imZSBP2, imBin & ~imZSThin2, imBin & ~imZSBP2}, bgrzs2);
    debug_imwrite(bgrzs2, "t1a_szraw2");

    // old version
    log("thin1 st", LogLevel::debug);
    cv::Mat imZSThin1;
    cv::Mat imZSBP1;
    thinning(imBin, imZSThin1, imZSBP1);
    log("thin1 end", LogLevel::debug);

    cv::Mat bgrzs;
    package_bgr({(imBin & ~imZSThin1) | imZSBP1, imBin & ~imZSThin1, imBin & ~imZSBP1}, bgrzs);
    debug_imwrite(bgrzs, "t1a_szraw");*/


    if(1>0) return;

    // Sort lines and markings
    cv::Mat imLine(R, C, T, 0.0);
    cv::Mat imMark(R, C, T, 0.0);
    std::vector<cv::Point> marking_centroids;
    std::vector<int> marking_sizes;
    remove_markings(imBin, imLine, imMark, marking_centroids, marking_sizes);
    log("Lines and markings sorted", LogLevel::debug);
    log("There are " + std::to_string(marking_sizes.size()) + " markings.", LogLevel::debug);

    // Downscale the lines
    cv::Mat smLine, smMask;
    cv::resize(imLine, smLine, cv::Size(0,0), STREET_SCALE_FACTOR, STREET_SCALE_FACTOR, cv::INTER_AREA);
    cv::resize(mask, smMask, cv::Size(0,0), STREET_SCALE_FACTOR, STREET_SCALE_FACTOR, cv::INTER_AREA);
    const size_t smR = smLine.rows, smC = smLine.cols;
    cv::threshold(smLine, smLine, 1, 255, CV_8UC1);
    cv::threshold(smMask, smMask, 1, 255, CV_8UC1);

    // Erode and Close the lines, so that the skeleton pruning algorithm works better
    const int fill_area = 24; //tested
    const int erosion_diameter = 7; // must be odd, tested
    fill_holes(smLine, fill_area); // fill in any small holes in the lines, to prevent loops
    cv::Mat erosion_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size{erosion_diameter,erosion_diameter});
    cv::Mat smLineErode;
    cv::erode(smLine, smLineErode, erosion_kernel);
    smLine = smLine - smLineErode; // Open up any large areas, as we can't vectorize them
    log("Lines downsized", LogLevel::debug);
    log("Small size: " + std::to_string(smR) + ", " + std::to_string(smC), LogLevel::debug);

    // Apply skeleton pruning algorithm to the lines
    std::vector<std::vector<cv::Point>> line_chains = shrink(smLine, /*prune_thresh*/3.0, /*contour_approx*/18.0);
    log("Skeleton of small lines", LogLevel::debug);

    // Vectorize the small lines.
    std::vector<MyLine> smVecLines;
    std::vector<MyVertex> smVecVerts;
    segment_lines(line_chains, smVecLines, smVecVerts, /*ht*/4, /*len*/2, /*curl*/0);
    log("Vectorization of small lines", LogLevel::debug);

    // Apply long line detection algorithm to the small lines
    std::vector<LongLine> smVecLongLines;
    generate_long_lines(smVecLines, smVecLongLines);
    combine_long_lines(smVecLines, smVecVerts, smVecLongLines, smLine,
                        /*vert*/ 2, /*dist*/ 1, /*matching_algorithm*/ MATCH_VECLINES);
    log("Long line detection", LogLevel::debug);

    // Use flood fill to help curb line detector
    // This was the old street detection method
    cv::Mat smStreetFF(smR, smC, T, 0.0);
    floodfill_streets(smLine, smMask, smStreetFF);
    log("Flood fill streets located", LogLevel::debug);

    // Downscale the modern streets segments to match the size of other things
    std::vector<Street> smPresentStreets;
    for (Street s: present_streets) {
        struct Street ds;
        ds.feature_id = s.feature_id;
        ds.id = s.id;
        ds.segment = s.segment;
        ds.segment.p1 *= STREET_SCALE_FACTOR;
        ds.segment.p2 *= STREET_SCALE_FACTOR;
        smPresentStreets.push_back(ds);
    }

    // Use our curb line detection methods
    curbMatchModern(smVecLines, smVecLongLines, smPresentStreets);
    log("Curb Lines - Modern", LogLevel::debug);

    curbFloodFill(smVecLines, smVecLongLines, smStreetFF, smMask);
    log("Curb Lines - Flood Fill", LogLevel::debug);

    curbBranchPoints(smVecLines, smVecLongLines, smVecVerts);
    log("Curb Lines - Branch & End Points", LogLevel::debug);

    curbMatchMarkings(smVecLines, smVecLongLines, marking_centroids, marking_sizes, smLine);
    log("Curb Lines - Markings", LogLevel::debug);

    curbMatchPara(smVecLines, smVecLongLines);
    log("Curb Lines - Parallel", LogLevel::debug);

    curbAggregate(smVecLines, smVecLongLines);
    log("Curb Lines - Aggregate", LogLevel::debug);

    // Declare street ownership matrices
    cv::Mat ownOrientX(ownR, ownC, CV_32F, 0.0); // x orientation along street
    cv::Mat ownOrientY(ownR, ownC, CV_32F, 0.0); // y orientation along street
    cv::Mat ownCurbDistance(ownR, ownC, CV_32F, 0.0); // distance from curb to street pixel
    cv::Mat ownMidDistance(ownR, ownC, CV_32F, 0.0); // distance from center of street to street pixel
    cv::Mat streetOwnership(ownR, ownC, CV_32S, 0.0); // Which street owns each street pixel

    // Aggregate streets image, using all curb line results
    cv::Mat smStreetCurb(smR, smC, T, 0.0);
    curbToStreets(smVecLines, smVecLongLines, smStreetCurb, ownOrientX, ownOrientY, ownCurbDistance);
    log("Curb Lines to Streets", LogLevel::debug);

    // Consider the modern streets as well.
    // There is an option to use the OR of the modern streets and the streets
    // we detect using curb lines.
    cv::Mat smStreetModern(smR, smC, T, 0.0);
    cv::Mat smStreetAgg(smR, smC, T, 0.0);
    const int modernDrawThickness = 20 * STREET_SCALE_FACTOR / RESOLUTION; // represents width of modern street
    if (SEAM_CARVE_MODERN_STREETS) {
        for (size_t i = 0; i < present_streets.size(); ++i) {
            cv::line(smStreetModern, (present_streets[i]).segment.p1 * STREET_SCALE_FACTOR, present_streets[i].segment.p2 * STREET_SCALE_FACTOR, CV_RGB(255,255,255), modernDrawThickness);
        }
        smStreetAgg = smStreetModern | smStreetCurb;
    }
    else {
        smStreetAgg = smStreetCurb;
    }
    log("Modern streets considered", LogLevel::debug);

    // Downscale streets even further
    cv::Mat miniStreet;
    cv::resize(smStreetAgg, miniStreet, cv::Size(0,0), SHRINK_SCALE_FACTOR, SHRINK_SCALE_FACTOR, cv::INTER_AREA);
    cv::threshold(miniStreet, miniStreet, 1, 255, CV_8UC1);
    fill_holes(miniStreet, 5);
    log("Streets made mini", LogLevel::debug);

    // Apply skeleton pruning to the streets
    std::vector<std::vector<cv::Point>> chains = shrink(miniStreet, 4.0, 25.0);
    log("Street shrink found", LogLevel::debug);

    // Vectorize the streets
    std::vector<MyLine> miniStreetLines;
    std::vector<MyVertex> miniStreetVerts;
    segment_lines(chains, miniStreetLines, miniStreetVerts, 4, 7, 3);

    // Fill in info about the streets
    int street_segment_id = 0;
    for (MyLine& l: miniStreetLines) {
        l.streetId_ = street_segment_id++; // Assign an ID to each street segment.
        l.correctDomi();

        // Check if the street is modern only, or if it was
        // detected using curb lines
        cv::Point smMid = l.mid_ / SHRINK_SCALE_FACTOR;
        if (pixelInBounds(smMid, smStreetCurb)) {
            pixel_t curbSt = smStreetCurb.at<pixel_t>(smMid);
            if (curbSt > 0) l.streetFromCurb_ = true;
            pixel_t modSt = smStreetModern.at<pixel_t>(smMid);
            if (modSt > 0) l.streetFromModern_ = true;
        }
    }
    log("Streets vectorized", LogLevel::debug);

    // Rescale the streets so that they are at the size of the original image
    std::vector<MyLine> streetLines; // street lines that are fully sized
    std::vector<MyVertex> streetVerts;
    copyGraph(miniStreetLines, miniStreetVerts, streetLines, streetVerts);
    scaleGraph(streetLines, streetVerts, 1 / (STREET_SCALE_FACTOR * SHRINK_SCALE_FACTOR));
    log("Streets upscaled", LogLevel::debug);

    // match the historical and present streets
    std::unordered_map<uint, std::vector<int>> h2p_matches = match_streets(streetLines, present_streets);
    std::vector<LongLine> extendedHistoricalStreetLongLines;
    generate_long_lines(streetLines, extendedHistoricalStreetLongLines);
    // extend the historical street segments into historical streets
    combine_long_lines(streetLines, streetVerts, extendedHistoricalStreetLongLines, imRaw,
                        /*vert*/ 2, /*dist*/ 1, /*matching_algorithm*/ MATCH_EXTENDED_STREETS);
    for (LongLine& ll: extendedHistoricalStreetLongLines) {
        ll.forceConsistentOrderings(streetLines, streetVerts);
    }
    log("Streets matched/extended", LogLevel::debug);

    // Apply seam carving algorithm
    cv::Mat imStMidd(R, C, T, 0.0);
    cv::Mat imStEdge(R, C, T, 0.0);
    cv::Mat imStSeam(R, C, T, 0.0);

    for (size_t i = 0; i < streetLines.size(); ++i) {
        MyLine& street = streetLines[i]; // Iterate over each street segment

        // Check if we should actually seam carve this segment.
        // We ignore short and misaligned segments
        cv::Point2f domiAlign {searchForOwnerF(street.mid_, ownOrientX), searchForOwnerF(street.mid_, ownOrientY)};
        float dotP = domiAlign.x * street.domi_.x + domiAlign.y * street.domi_.y;
        if ((street.width_ < 600 && std::abs(dotP) < 0.8) || street.width_ < 100) {
            street.probablyNotStreet_ = true;
            continue;
        }

        // Seam carve the street, once on each side.
        street.width_ += SEAM_CARVE_EXTENSION; // extend slightly to also get the intersection
        seam_carve_street(street, imLine, smStreetAgg, streetLines,
                          imStMidd, imStEdge, imStSeam, streetOwnership, ownMidDistance);
        street.norm_ *= -1; // flip to other side
        seam_carve_street(street, imLine, smStreetAgg, streetLines,
                          imStMidd, imStEdge, imStSeam, streetOwnership, ownMidDistance);
        street.norm_ *= -1;
        street.width_ -= SEAM_CARVE_EXTENSION;
    }

    // Create images for the edge and middle of the street based on seam carving
    cv::blur(imStMidd, imStMidd, cv::Size(9,9));
    cv::blur(imStEdge, imStEdge, cv::Size(9,9));
    cv::threshold(imStMidd, imStMidd, 1, 255, cv::THRESH_BINARY);
    cv::threshold(imStEdge, imStEdge, 1, 255, cv::THRESH_BINARY);
    log("Seam carving done", LogLevel::debug);

    // Find the markings that are in the street. Note,
    // we ignore lines in the middle of the street, but keep them
    // in the edges, because they may have been house numbers
    // that were recovered using seam carving.
    cv::Mat imMarkInStreet = (imMark & (imStEdge | imStMidd))
                        | (imLine & imStEdge & ~imStSeam);
    log("Markings in the street found", LogLevel::debug);

    // Detect the words in the image
    std::vector<MyLine> words;
    std::vector<std::vector<cv::Point>> wordPixels;
    form_words_all(imMarkInStreet, ownMidDistance, ownCurbDistance, streetOwnership, ownOrientX, ownOrientY, streetLines, extendedHistoricalStreetLongLines, words, wordPixels);
    log("Words found", LogLevel::debug);

    // Create texture pack matrix image
    cv::Mat texturePack (2048, 2048, CV_8U, 0.0); // Size could be adjusted
    int texturePackCount = 0, texMaxHeight = 128-1, texLastR = 0, texLastC = 0;
    std::string textureFileName = "texturepack";

    // Iterate over each detected word
    for (size_t i = 0; i < words.size(); ++i) {
        MyLine& word = words[i];

        // Get the word in a matrix
        cv::Mat rot_word = my_rotate(imMarkInStreet, word);

        std::string fname = "ZZZrot_word_w" + std::to_string(word.wordId_) + "_s"
                            + std::to_string(word.streetId_);

        if (word.isHouseNumber_) fname = fname + "_hn";
        if (word.isStreetName_) fname = fname + "_st";
        if (word.connectedComponentPass_) fname = fname + "_cc";
        if (word.probablyNotWord_) fname = fname + "_nw";

        // rescale to be 32 tall
        //cv::resize(rotWord, rotWord, cv::Size((int)(rotWord.cols*32.0/rotWord.rows), 32), 0, 0, cv::INTER_LINEAR);
        //cv::threshold(rotWord, rotWord, 128, 255, CV_8UC1);

        // If connected components passes, crop out individual chars as well
        if (word.connectedComponentPass_) {

            // Use connected components
            cv::Mat charLabels;
            std::vector<std::vector<cv::Point>> char_points_list =
                connectedComponentsWithIdx(rot_word, charLabels);

            // sort the connected components by their x-position
            // start at begin+1 because the first element is an empty vector (the background component)
            std::sort(char_points_list.begin() + 1, char_points_list.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) -> bool
                {
                    return averageXPos(a) < averageXPos(b);
                });

            // Iterate over each char in the word
            for (size_t q = 1; q < char_points_list.size(); q++) {

                // Convert pixels to image
                cv::Mat charImg = idxToImg(char_points_list[q], 2);

                // Write the char to the texture pack, and if the option is
                // set, write the char to file as well
                if (word.isStreetName_ && char_points_list[q].size() > 80) {
                    if (OUTPUT_CROPPED_IMAGES)
                        specific_imwrite(charImg, "!!cropped_letters", fname + std::to_string(q));
                    texturepack(texturePack, charImg, word, texLastR, texLastC,
                        texMaxHeight, textureFileName, texturePackCount);
                }
                else if (!word.isStreetName_ && char_points_list[q].size() > 35) {
                    if (OUTPUT_CROPPED_IMAGES)
                        specific_imwrite(charImg, "!!cropped_digits", fname + std::to_string(q));
                    texturepack(texturePack, charImg, word, texLastR, texLastC,
                        texMaxHeight, textureFileName, texturePackCount);
                }
            }
        }

        // Output word to special folder
        // Not needed, since we're outputting to the texture pack.
        if (OUTPUT_CROPPED_IMAGES) {
            specific_imwrite(rot_word, "!!cropped_words", fname + "_nFit");
        }

        texturepack(texturePack, rot_word, word, texLastR, texLastC,
                    texMaxHeight, textureFileName, texturePackCount);

        // Option to deskew house numbers (currently unused)
        // if (word.isHouseNumber_) {
        //     cv::Mat deskewWord = my_deskew(rot_word, 0.50);
        //     if (OUTPUT_CROPPED_CHARS)
        //         midway_imwrite(deskewWord, fname + "_nSkew");
        //     texturepack(texturePack, deskewWord, ignoreWord, texLastR, texLastC,
        //             texMaxHeight, textureFileName, texturePackCount);
        // }
    }

    // Output final texture pack
    midway_imwrite(texturePack, textureFileName + std::to_string(texturePackCount));
    log("Words output", LogLevel::debug);

    // Write JSON stuff
    write_streets(streetLines);
    write_words(words);
    write_matches(h2p_matches);
    write_extensions(extendedHistoricalStreetLongLines);
    log("JSON outputted", LogLevel::debug);


    // Do all of the debug draws at once so timing is more accurate
    // Keep them all in separate if statements for memory
    if (DRAW_DEBUG_IMAGES) {

        debug_imwrite(imRaw, "a1raw");
        debug_imwrite(imBin, "a2binary");
        debug_imwrite(mask, "a2mask");

        cv::Mat bgr;
        package_bgr({(~mask * 0.2) | imLine, (~mask * 0.2) | imLine, (~mask * 0.2) | imMark}, bgr);
        debug_imwrite(bgr, "a3a_lines_markings");
        //make_dir(DEBUG_PATH, "!lines_markings");
        //specific_imwrite(bgrLM, "!lines_markings");

        cv::Mat bgr2;
        package_bgr({0*imRaw, imBin, 255-imRaw}, bgr2);
        debug_imwrite(bgr2, "a2binary_compare");
        //make_dir(DEBUG_PATH, "!binarize");
        //specific_imwrite(bgr2, "!binarize");

    }

    if (DRAW_DEBUG_IMAGES) {
        //make_dir(DEBUG_PATH, "!streets");
        cv::Mat bgr;
        package_bgr({(~smMask * 0.2) | smLine, (~smMask * 0.2) | smLine * 0.5, smStreetModern}, bgr);

        for (size_t i = 0; i < present_streets.size(); ++i) {
            cv::line(bgr, (present_streets[i]).segment.p1 * STREET_SCALE_FACTOR, present_streets[i].segment.p2 * STREET_SCALE_FACTOR, CV_RGB(0,255,0), 2);
        }
        debug_imwrite(bgr, "a2present_streets");
        //specific_imwrite(bgr, "!streets", "_ps");
    }

    if (DRAW_DEBUG_IMAGES) {
        cv::Mat colorSt;
        cv::Mat notSt(smR, smC, T, 0.0);
        cv::bitwise_not(smLine, notSt);
        color_connected_components(notSt, colorSt, {255,255,0}, {0,0,40}, {40,130,255});
        debug_imwrite(colorSt, "a3f_floodfill_streets");
        colorSt.release();
        notSt.release();
    }

    if (DRAW_DEBUG_IMAGES) {
        cv::Mat miniChains = cv::Mat(smLine.size(), smLine.type(), 0.0);
        cv::Mat miniBranchPoints = cv::Mat(smLine.size(), smLine.type(), 0.0);
        draw_chains(line_chains, miniChains, miniBranchPoints);

        cv::Mat bgr;
        package_bgr({smLine & ~miniBranchPoints, miniChains & ~miniBranchPoints, miniBranchPoints}, bgr);
        cv::Mat bgrLongLines = cv::Mat(smLine.size(), bgr.type(), 0.0);
        debug_imwrite(bgr, "a3c_skel_lines");
        //make_dir(DEBUG_PATH, "!skel_lines");
        //specific_imwrite(bgr, "!skel_lines", "_skl_s");

        for (size_t i = 0; i < smVecLines.size(); ++i) {
            MyLine l = smVecLines[i];
            std::pair<cv::Point, cv::Point> endpts = l.endpoints();
            cv::line(bgr, endpts.first, endpts.second, id_color(l.line_id),2);
            cv::line(bgrLongLines, endpts.first, endpts.second, id_color(l.long_line),2);
        }
        debug_imwrite(smLine, "a3b_smlines");
        debug_imwrite(bgr, "a3d_skel_lines_vec");
        debug_imwrite(bgrLongLines, "a3e_long_lines_vec");
        //specific_imwrite(bgr, "!skel_lines", "_skl_vec");
        //specific_imwrite(bgrLongLines, "!skel_lines", "_skl_ll");
    }

    bool curbLineDebugImages = true;
    bool spamtext = false;
    if (DRAW_DEBUG_IMAGES && curbLineDebugImages) {
        // curb lines
        cv::Mat bgr;
        package_bgr({imLine * 0.4, imLine * 0.4, imMark * 0.3}, bgr);

        { //Markings
            cv::Mat bgr1 = bgr.clone();
            for (size_t i = 0; i < marking_centroids.size(); ++i) {
                cv::circle(bgr1, marking_centroids[i], 1+sqrt(marking_sizes[i]/3.14), CV_RGB(255,0,0));
            }
            for (size_t i = 0; i < smVecLines.size(); ++i) {
                MyLine line = smVecLines[i];
                std::pair<cv::Point, cv::Point> endpts = line.endpoints();
                LongLine& ll = smVecLongLines[line.long_line];
                if (ll.ignore_me) continue;
                cv::Point norm = ll.normVector();
                cv::Point mid = line.mid_;
                norm *= (15.0/cv::norm(norm)); // rescale to full size image for readability
                mid /= STREET_SCALE_FACTOR;
                endpts.first /= STREET_SCALE_FACTOR;
                endpts.second /= STREET_SCALE_FACTOR;
                // positive
                int colP = 25 + 100*(ll.curbLineMarkingScoreP);
                cv::line(bgr1, endpts.first + norm, endpts.second + norm, CV_RGB(0,colP,colP),20);
                // negative
                int colN = 25 + 100*(ll.curbLineMarkingScoreN);
                cv::line(bgr1, endpts.first - norm, endpts.second - norm, CV_RGB(0,colN,colN),20);
                // helpful text
                if (spamtext&&colP!=25) cv::putText(bgr1, std::to_string(ll.curbLineMarkingScoreP).substr(0,5), mid + norm, 0, 1.0, CV_RGB(255,255,155),2);
                if (spamtext&&colN!=25) cv::putText(bgr1, std::to_string(ll.curbLineMarkingScoreN).substr(0,5), mid - norm, 0, 1.0, CV_RGB(255,205,205),2);
            }
            //specific_imwrite(bgr1, "!streets", "_z1_markings");
            debug_imwrite(bgr1, "a4z1_markings");
        }

        { //Modern
            cv::Mat bgr1 = bgr.clone();
            for (size_t i = 0; i < present_streets.size(); ++i) {
                cv::line(bgr1, (present_streets[i]).segment.p1, present_streets[i].segment.p2, CV_RGB(0,80,0), 30);
            }
            for (size_t i = 0; i < smVecLines.size(); ++i) {
                MyLine line = smVecLines[i];
                std::pair<cv::Point, cv::Point> endpts = line.endpoints();
                LongLine& ll = smVecLongLines[line.long_line];
                if (ll.ignore_me) continue;
                cv::Point norm = ll.normVector();
                cv::Point mid = line.mid_;
                norm *= (15.0/cv::norm(norm)); // rescale to full size image for readability
                mid /= STREET_SCALE_FACTOR;
                endpts.first /= STREET_SCALE_FACTOR;
                endpts.second /= STREET_SCALE_FACTOR;
                // positive
                int colP = 25 + 100*(ll.curbLineModernScoreP);
                cv::line(bgr1, endpts.first + norm, endpts.second + norm, CV_RGB(colP,0,colP),20);
                // negative
                int colN = 25 + 100*(ll.curbLineModernScoreN);
                cv::line(bgr1, endpts.first - norm, endpts.second - norm, CV_RGB(colN,0,colN),20);
                // helpful text
                if (spamtext&&colP!=25) cv::putText(bgr1, std::to_string(ll.curbLineModernScoreP).substr(0,5), mid + norm, 0, 1.0, CV_RGB(255,255,155),2);
                if (spamtext&&colN!=25) cv::putText(bgr1, std::to_string(ll.curbLineModernScoreN).substr(0,5), mid - norm, 0, 1.0, CV_RGB(205,255,205),2);
            }
            //specific_imwrite(bgr1, "!streets", "_z2_modern");
            debug_imwrite(bgr1, "a4z2_modern");
        }

        { // Flood fill
            cv::Mat bgr1;
            cv::Mat imStreetFF;
            cv::Size fullSize (bgr.cols, bgr.rows);
            cv::resize(smStreetFF, imStreetFF, fullSize);
            package_bgr({imLine * 0.4 | imStreetFF, imLine * 0.4, imMark * 0.3}, bgr1);
            for (size_t i = 0; i < smVecLines.size(); ++i) {
                MyLine line = smVecLines[i];
                std::pair<cv::Point, cv::Point> endpts = line.endpoints();
                LongLine& ll = smVecLongLines[line.long_line];
                if (ll.ignore_me) continue;
                cv::Point norm = ll.normVector();
                cv::Point mid = line.mid_;
                norm *= (15.0/cv::norm(norm)); // rescale to full size image for readability
                mid /= STREET_SCALE_FACTOR;
                endpts.first /= STREET_SCALE_FACTOR;
                endpts.second /= STREET_SCALE_FACTOR;
                // positive
                int colP = 25 + 100*(ll.curbLineFloodFillScoreP);
                cv::line(bgr1, endpts.first + norm, endpts.second + norm, CV_RGB(colP,colP,0),20);
                // negative
                int colN = 25 + 100*(ll.curbLineFloodFillScoreN);
                cv::line(bgr1, endpts.first - norm, endpts.second - norm, CV_RGB(colN,colN,0),20);
                // helpful text
                if (spamtext&&colP!=25) cv::putText(bgr1, std::to_string(ll.curbLineFloodFillScoreP).substr(0,5), mid + norm, 0, 1.0, CV_RGB(255,155,255),2);
                if (spamtext&&colN!=25) cv::putText(bgr1, std::to_string(ll.curbLineFloodFillScoreN).substr(0,5), mid - norm, 0, 1.0, CV_RGB(205,205,255),2);
            }
            //specific_imwrite(bgr1, "!streets", "_z3_floodfill");
            debug_imwrite(bgr1, "a4z3_floodfill");
        }

        { // Parallel
            cv::Mat bgr1 = bgr.clone();
            for (size_t i = 0; i < smVecLines.size(); ++i) {
                MyLine line = smVecLines[i];
                std::pair<cv::Point, cv::Point> endpts = line.endpoints();
                LongLine& ll = smVecLongLines[line.long_line];
                if (ll.ignore_me) continue;
                cv::Point norm = ll.normVector();
                cv::Point mid = line.mid_;
                norm *= (15.0/cv::norm(norm)); // rescale to full size image for readability
                mid /= STREET_SCALE_FACTOR;
                endpts.first /= STREET_SCALE_FACTOR;
                endpts.second /= STREET_SCALE_FACTOR;
                // positive
                int colP = 25 + 100*(ll.curbLineParaMatchScoreP);
                cv::line(bgr1, endpts.first + norm, endpts.second + norm, CV_RGB(colP,0,0),20);
                // negative
                int colN = 25 + 100*(ll.curbLineParaMatchScoreN);
                cv::line(bgr1, endpts.first - norm, endpts.second - norm, CV_RGB(colN,0,0),20);
                // helpful text
                //if (spamtext) cv::putText(bgr1, std::to_string(ll.id_).substr(0,5), mid, 0, 1.0, CV_RGB(255,255,255),2);
                if (spamtext&&colP!=25) cv::putText(bgr1, std::to_string(ll.curbLineParaMatchScoreP).substr(0,5), mid + norm, 0, 1.0, CV_RGB(155,255,255),2);
                if (spamtext&&colN!=25) cv::putText(bgr1, std::to_string(ll.curbLineParaMatchScoreN).substr(0,5), mid - norm, 0, 1.0, CV_RGB(205,205,255),2);
            }
            //specific_imwrite(bgr1, "!streets", "_z4_parallel");
            debug_imwrite(bgr1, "a4z4_parallel");
        }

        { // Branch Point
            cv::Mat bgr1 = bgr.clone();
            for (size_t i = 0; i < smVecLines.size(); ++i) {
                MyLine line = smVecLines[i];
                std::pair<cv::Point, cv::Point> endpts = line.endpoints();
                LongLine& ll = smVecLongLines[line.long_line];
                if (ll.ignore_me) continue;
                cv::Point norm = ll.normVector();
                cv::Point mid = line.mid_;
                norm *= (15.0/cv::norm(norm)); // rescale to full size image for readability
                mid /= STREET_SCALE_FACTOR;
                endpts.first /= STREET_SCALE_FACTOR;
                endpts.second /= STREET_SCALE_FACTOR;
                // positive
                int colP = 25 + 100*(ll.curbLineBranchPointScoreP);
                cv::line(bgr1, endpts.first + norm, endpts.second + norm, CV_RGB(0,colP,0),20);
                // negative
                int colN = 25 + 100*(ll.curbLineBranchPointScoreN);
                cv::line(bgr1, endpts.first - norm, endpts.second - norm, CV_RGB(0,colN,0),20);
                // helpful text
                if (spamtext&&colP!=25) cv::putText(bgr1, std::to_string(ll.curbLineBranchPointScoreP).substr(0,5), mid + norm, 0, 1.0, CV_RGB(255,155,255),2);
                if (spamtext&&colN!=25) cv::putText(bgr1, std::to_string(ll.curbLineBranchPointScoreN).substr(0,5), mid - norm, 0, 1.0, CV_RGB(255,205,205),2);
            }
            //specific_imwrite(bgr1, "!streets", "_z5_branchpt");
            debug_imwrite(bgr1, "a4z5_branchpt");
        }

        { // End Point
            cv::Mat bgr1 = bgr.clone();
            for (size_t i = 0; i < smVecLines.size(); ++i) {
                MyLine line = smVecLines[i];
                std::pair<cv::Point, cv::Point> endpts = line.endpoints();
                LongLine& ll = smVecLongLines[line.long_line];
                if (ll.ignore_me) continue;
                cv::Point norm = ll.normVector();
                cv::Point mid = line.mid_;
                norm *= (15.0/cv::norm(norm)); // rescale to full size image for readability
                mid /= STREET_SCALE_FACTOR;
                endpts.first /= STREET_SCALE_FACTOR;
                endpts.second /= STREET_SCALE_FACTOR;
                // positive
                int colP = 25 + 100*(ll.curbLineEndPointScoreP);
                cv::line(bgr1, endpts.first + norm, endpts.second + norm, CV_RGB(0,colP,colP),20);
                // negative
                int colN = 25 + 100*(ll.curbLineEndPointScoreN);
                cv::line(bgr1, endpts.first - norm, endpts.second - norm, CV_RGB(0,colN,colN),20);
                // helpful text
                if (spamtext&&colP!=25) cv::putText(bgr1, std::to_string(ll.curbLineEndPointScoreP).substr(0,5), mid + norm, 0, 1.0, CV_RGB(255,155,255),2);
                if (spamtext&&colN!=25) cv::putText(bgr1, std::to_string(ll.curbLineEndPointScoreN).substr(0,5), mid - norm, 0, 1.0, CV_RGB(255,205,205),2);
            }
            //specific_imwrite(bgr1, "!streets", "_z6_endpt");
            debug_imwrite(bgr1, "a4z6_endpt");
        }
    }

    if (DRAW_DEBUG_IMAGES)
    {
        cv::Mat bgr;
        package_bgr({imLine * 0.4, imLine * 0.4, imMark * 0.3}, bgr);

        { // Aggregate
            cv::Mat bgr1;
            cv::Mat imStreetAgg;
            cv::Size fullSize (bgr.cols, bgr.rows);
            cv::resize(smStreetCurb, imStreetAgg, fullSize);
            package_bgr({imLine * 0.4 | imStreetAgg, imLine * 0.4, imMark * 0.3}, bgr1);
            for (size_t i = 0; i < present_streets.size(); ++i) {
                cv::line(bgr1, (present_streets[i]).segment.p1, present_streets[i].segment.p2, CV_RGB(0,80,0), 30);
            }
            for (size_t i = 0; i < smVecLines.size(); ++i) {
                MyLine line = smVecLines[i];
                std::pair<cv::Point, cv::Point> endpts = line.endpoints();
                LongLine& ll = smVecLongLines[line.long_line];
                if (ll.ignore_me) continue;
                cv::Point norm = ll.normVector();
                cv::Point mid = line.mid_;
                norm *= (15.0/cv::norm(norm)); // rescale to full size image for readability
                mid /= STREET_SCALE_FACTOR;
                endpts.first /= STREET_SCALE_FACTOR;
                endpts.second /= STREET_SCALE_FACTOR;
                // positive
                int colP = 25 + 100*(ll.curbLineScoreP);
                cv::line(bgr1, endpts.first + norm, endpts.second + norm, CV_RGB(colP,0,colP),20);
                // negative
                int colN = 25 + 100*(ll.curbLineScoreN);
                cv::line(bgr1, endpts.first - norm, endpts.second - norm, CV_RGB(colN,0,colN),20);
                // helpful text
                if (spamtext&&colP!=25) cv::putText(bgr1, std::to_string(ll.curbLineScoreP).substr(0,5), mid + norm, 0, 1.0, CV_RGB(255,155,255),2);
                if (spamtext&&colN!=25) cv::putText(bgr1, std::to_string(ll.curbLineScoreN).substr(0,5), mid - norm, 0, 1.0, CV_RGB(255,205,205),2);
            }
            //specific_imwrite(bgr1, "!streets", "_z7_aggregate");
            debug_imwrite(bgr1, "a4a_curb_lines");
        }
    }

    if (DRAW_DEBUG_IMAGES) {
        cv::Mat bgr;
        cv::Mat b = (~smMask * 0.2) | smLine | smStreetAgg;
        cv::Mat g = (~smMask * 0.2) | smLine | smStreetCurb * 0.5;
        cv::Mat r = (~smMask * 0.2) | smStreetModern * 0.5;
        package_bgr({b, g, r}, bgr);
        debug_imwrite(bgr, "a5small_streets_agg");
    }

    if (DRAW_DEBUG_IMAGES) {
        cv::Mat miniChains = cv::Mat(miniStreet.size(), miniStreet.type(), 0.0);
        cv::Mat miniBranchPoints = cv::Mat(miniStreet.size(), miniStreet.type(), 0.0);
        draw_chains(chains, miniChains, miniBranchPoints);

        cv::Mat bgr;
        package_bgr({miniStreet & ~miniBranchPoints, miniChains & ~miniBranchPoints, miniBranchPoints}, bgr);
        debug_imwrite(bgr, "a6a_shrink_streets");
        for (size_t i = 0; i < miniStreetLines.size(); ++i) {
            MyLine street = miniStreetLines[i];
            std::pair<cv::Point, cv::Point> endpts = street.endpoints();
            cv::Point pm {(int) street.mid_.x, (int) street.mid_.y};
            cv::line(bgr, endpts.first, pm, CV_RGB(255,0,255),2); //light=left
            cv::line(bgr, pm, endpts.second, CV_RGB(225,0,255),2); //dark = right
        }
        //make_dir(DEBUG_PATH, "!vectorize_st");
        debug_imwrite(bgr, "a6b_vectorized_streets");
        //specific_imwrite(bgr, "!vectorize_st", "_vst");
    }

    if (DRAW_DEBUG_IMAGES) {
        bool infoColors = false;
        cv::Mat bgr;
        package_bgr({(~mask * 0.2) | imLine, (~mask * 0.2) | imLine, (~mask * 0.2) | imMark}, bgr);
        for (size_t i = 0; i < streetLines.size(); ++i) {
            MyLine street = streetLines[i];
            std::pair<cv::Point, cv::Point> endpts = street.endpoints();
            cv::Point pm {(int) street.mid_.x, (int) street.mid_.y};
            int g = infoColors ? (street.probablyNotStreet_ ? 255:0) : 0;
            int b = infoColors ? (street.streetFromCurb_ ? 225:0) : 220;
            int r = infoColors ? (street.streetFromModern_ ? 225:0) : 220;
            cv::line(bgr, endpts.first, pm, CV_RGB(r,g,b),30);
            cv::line(bgr, pm, endpts.second, CV_RGB(r+25,g,b+25),30);
        }
        debug_imwrite(bgr, "a6c_vectorized_streets_full");
        //specific_imwrite(bgr, "!vectorize_st", "_full");
    }

    if (DRAW_DEBUG_IMAGES) {
        cv::Mat bgr;
        package_bgr({imLine, imLine, imMark}, bgr);
        cv::Mat bgrLongLines;
        package_bgr({imLine, imLine, imMark}, bgrLongLines);

        for (size_t i = 0; i < streetLines.size(); ++i) {
            MyLine street = streetLines[i];
            std::pair<cv::Point, cv::Point> endpts = street.endpoints();
            cv::Point pm {(int) street.mid_.x, (int) street.mid_.y};
            cv::line(bgr, endpts.first, endpts.second, id_color(street.line_id),50);
            cv::Point emid = (endpts.first+endpts.second)/2;
            cv::line(bgrLongLines, endpts.first, emid, id_color(street.long_line),40);
            cv::line(bgrLongLines, emid, endpts.second, id_color(street.long_line),50);
        }
        debug_imwrite(bgr, "a6y_vectorized_street_segments");
        debug_imwrite(bgrLongLines, "a6z_extended_historical_streets");
    }

    if (DRAW_DEBUG_IMAGES) {
        cv::Mat bgrStOwn;
        color_labels(streetOwnership, bgrStOwn);
        debug_imwrite(bgrStOwn, "a8b_streetOwnership");

        cv::Mat bgrOrient(R, C, CV_8U, 0.0);
        for (int y = 0; y < R; y += 15) {
            for (int x = y%15; x < C; x += 15) {
                cv::Point p{x,y};
                cv::Point smQ = point_scale(p, OWNERSHIP_SCALE_FACTOR);
                if (!pixelInBounds(smQ, bgrOrient)) continue;
                cv::Point q{(int)(10*ownOrientX.at<float>(smQ)), (int)(10*ownOrientY.at<float>(smQ))};
                if (pixelInBounds(p, bgrOrient) && pixelInBounds(p+q, bgrOrient))
                    cv::line(bgrOrient, p-q, p+q, CV_RGB(0,0,255),1);
            }
        }
        debug_imwrite(bgrOrient, "a8c_orient");
        debug_imwrite(ownMidDistance, "a8d_mid_Dist");
        debug_imwrite(ownCurbDistance, "a8e_curb_Dist");
        debug_imwrite(ownCurbDistance, "a8e_curb_Dist");
    }

    if (DRAW_DEBUG_IMAGES) {
        cv::Mat bgrBB;
        cv::Mat bm = (imLine & ~imMarkInStreet);
        cv::Mat gm = (imLine & ~imMarkInStreet);
        cv::Mat rm = imMarkInStreet | imMark * 0.4;
        package_bgr({bm, gm, rm}, bgrBB);
        for (size_t j = 0; j < words.size(); ++j) {
            MyLine word = words[j];
            std::vector<cv::Point> e = word.boundingBox();
            int r = word.probablyNotWord_ ? 100 : 255;
            int g = word.isHouseNumber_ ? 155 : (word.isStreetName_ ? 255 : 0);
            int b = 20;
            bool simpleColors = false;
            cv::line(bgrBB, e[0], e[1], CV_RGB(r,g,b), 3);
            cv::line(bgrBB, e[2], e[3], CV_RGB(r,g,b), 3);
            if (simpleColors) {
                cv::line(bgrBB, e[2], e[1], CV_RGB(r,g,b), 3);
                cv::line(bgrBB, e[0], e[3], CV_RGB(r,g,b), 3);
            }
            else {
                int ccp = 255*word.connectedComponentPass_;
                cv::line(bgrBB, e[2], e[1], CV_RGB(255,ccp,ccp), 3);
                cv::line(bgrBB, e[0], e[3], CV_RGB(255,0,150), 3);
            }

        }
        debug_imwrite(bgrBB, "a9d_bounding_boxes");
        //specific_imwrite(bgrBB, "!word_finding", "_boxes");
        bgrBB.release();
        bm.release();
        gm.release();
        rm.release();

        log("Debug images drawn", LogLevel::debug);
    }

    if (DRAW_DEBUG_IMAGES) {
        // Seam carving last, it can give out of memory errors
        cv::Mat bgrSeamCarve;
        cv::Mat x = (imStSeam & ~imStMidd);
        cv::Mat b = imStMidd * 0.3;
        b = b | (imStEdge & ~imStMidd) * 0.6;
        b = b | x;
        cv::Mat g = (imLine & ~imMarkInStreet);
        g = g | x;
        cv::Mat r = imMarkInStreet | x;
        package_bgr({b, g, r}, bgrSeamCarve);
        debug_imwrite(bgrSeamCarve, "a8a_carved_streets");
        //make_dir(DEBUG_PATH, "!seam_carving");
        //specific_imwrite(bgrSeamCarve, "!seam_carving");
    }



}

