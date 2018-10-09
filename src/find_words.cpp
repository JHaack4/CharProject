#include <cassert>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "image_proc.h"

// This function is used to assign penalties. If the value x lies
// outside of the interval (m,M), there is a penalty of s
// times how far outside of the interval x is.
double clamp_score(double m, double M, double s, double x) {
    if (x > M) return s*(x-M);
    if (x < m) return s*(m-x);
    return 0;
}

// Used to sort by x-coordinate
bool lineXSort(const std::pair<MyLine, int>& a, const std::pair<MyLine, int>& b)
{
	return a.second < b.second;
}

// Gets the orientation of the closest curb line to a given point
cv::Point2f getOrientation(cv::Point in, cv::Mat& orientX, cv::Mat& orientY) {
    float x = searchForOwnerF(in, orientX);
    float y = searchForOwnerF(in, orientY);
    return cv::Point2f{x,y};
}

// ------------- FINDING WORDS ------------------

void form_words_all(cv::Mat& imMarkSt, cv::Mat& midDist, cv::Mat& curbDist,
                    cv::Mat& streetOwnership, cv::Mat& orientX, cv::Mat& orientY,
                     std::vector<MyLine>& streets, std::vector<LongLine>& extendedStreets, std::vector<MyLine>& words,
                     std::vector<std::vector<cv::Point>>& /*wordPixels*/) {

    // Run connected components on all of the markings in the street
    cv::Mat char_labels;
    std::vector<std::vector<cv::Point>> points_list = connectedComponentsWithIdx(imMarkSt, char_labels);

    // We will represent each char as a MyLine, and then group the characters
    // into words, which will be represented as LongLines. This representation
    // allows us to reuse all of our matching code.
    std::vector<MyLine> char_lines;      // MyLine for each char
    std::vector<int> char_kept_index;    // Pixel List indexes for the kept chars
    size_t num_chars = 0;                // number of chars kept

    // ---------------- Character shape analysis ---------------------

    // Parameters for character selection. Limits in pixel size
    // for a character that will be considered.
    const int CHAR_WAY_TOO_SMALL = 15;
    const int CHAR_WAY_TOO_TALL  = 100;
    const int CHAR_WAY_TOO_WIDE  = 120;
    const double CHAR_ASPECT_RATIO_LIMIT = 7.0;

    for (size_t i = 1; i < points_list.size(); ++i) {

        // IMPORTANT. Which characters do we keep?

        if (points_list[i].size() < CHAR_WAY_TOO_SMALL) continue; // Character too small in area

        // Determine an orientation to view this char from
        cv::Point2f charDomi {1,0};
        cv::Point repPt = points_list[i][points_list[i].size()/2];
        int stOwn = searchForOwner(repPt, streetOwnership);
        if (stOwn < 0 || stOwn >= (int)streets.size()) {
            cv::Point2f orientDomi = getOrientation(repPt, orientX, orientY);
            if (cv::norm(orientDomi) > 0.5) {
                charDomi = orientDomi; // Use orientation matrix
            } else {
                continue; // Can't determine orientation
            }
        }
        else { // Use street
            charDomi = streets[stOwn].domi_;
        }

        // Each character is a point cloud of connected pixels.
        // Fit a my line to that point. The orientation used for the
        // character is equal to the orientation of the street.
        MyLine charLine = fitPointCloud(points_list[i], charDomi, /*UseBoxCenter*/true);

        // Run more shape tests to throw out bad characters
        if (charLine.width_ > CHAR_WAY_TOO_WIDE || charLine.height_ > CHAR_WAY_TOO_TALL) continue;
        if (charLine.width_ > charLine.height_ * CHAR_ASPECT_RATIO_LIMIT) continue;

        // Make a note of how far the char is from the center of the street,
        // and how far it is from the edge.
        charLine.charDistFromCurb_ = searchForOwnerF(charLine.mid_, curbDist);
        charLine.charDistFromMiddle_ = searchForOwnerF(charLine.mid_, midDist);

        // If we make it to this point, the char is good. Add it to our lists.
        charLine.line_id = char_lines.size();
        char_lines.push_back(charLine);
        char_kept_index.push_back(i);
        num_chars++;
    }

    log("Character connected components done", LogLevel::debug);

    // ---------------- Character grouping into words ---------------------

    std::vector<LongLine> charLongLines;
    std::vector<MyVertex> charVerts;

    // Apply a matching algorithm (implemented in vectorize.cpp)
    // to match the characters into words.
    generate_long_lines(char_lines, charLongLines);
    generate_dummy_vertex(char_lines, charVerts);
    combine_long_lines(char_lines, charVerts, charLongLines, imMarkSt,
                        /*vert*/ 0, /*dist*/ 1, /*matching_algorithm*/ MATCH_CHARS);
    log("Character adjacency analyzed", LogLevel::debug);

    // ---------------- Determining an orientation for each word ---------------------
    // Now that we have decided which characters are words, we will
    // determine an orientation for each word. We then convert each word
    // to a MyLine object. Then, we will try to guess if the word is a
    // house number or street name, and whether connected components works
    // or not.

    int wordCount = 0;

    // Iterate over each group of chars that was matched
    for (size_t i = 0; i < charLongLines.size(); ++i) {
        LongLine& charGroup = charLongLines[i];
        if (charGroup.ignore_me) continue;

        int numCharsInWord = charGroup.size_;

        if (numCharsInWord == 1) {
            // This word only has 1 connected component. Deal with it separately

            // Get the line for this char
            MyLine& firstChar = char_lines[charGroup.line_ids[0]];

            // Determine the street that this word lies in
            cv::Point repPt = firstChar.mid_;
            bool middleOfStreet = firstChar.charDistFromCurb_ > firstChar.charDistFromMiddle_;
            int stOwn = searchForOwner(repPt, streetOwnership);
            if (stOwn == -1) {
                continue; // should never happen...
            }
            MyLine& street = streets[stOwn];

            // Create a word for this char
            MyLine wordS = firstChar;

            // Determine if this single char is likely to be a word or not
            int pixelArea = wordS.pixelCnt_;

            bool midPass = middleOfStreet && wordS.width_ < 200 && wordS.width_ > 30
                                        && wordS.height_ < 85 && wordS.height_ > 20
                                        && pixelArea > 150;

            bool edgePass = !middleOfStreet && wordS.width_ < 200 && wordS.width_ > 30
                                        && wordS.height_ < 45 && wordS.height_ > 15
                                        && pixelArea > 95;

            if (midPass) {
                // A single char in the middle is unlikely to be a street name
                // wordS.probablyNotWord_ = true;
                wordS.isStreetName_ = true;
            }
            else if (edgePass) {
                wordS.isHouseNumber_ = true;
            }
            else {
                wordS.probablyNotWord_ = true;
            }

            // Fix orientation relative to the street
            if (street.domi_.ddot(wordS.domi_) < 0) {
                wordS.flipDomi();
            }

            // Add to our word list
            wordS.wordId_ = wordCount++;
            wordS.streetId_ = stOwn;
            wordS.wordRelativeToStreet_ = street.line_transform(wordS.mid_);

            // Calculate location relative to extended historical street
            LongLine& extendedStreet = extendedStreets[street.long_line];
            Segment extendedStreetSegment = extendedStreet.segment();
            MyLine extendedStreetLine{extendedStreetSegment.p1, extendedStreetSegment.p2, true, false};
            wordS.wordRelativeToExtendedStreet_ = extendedStreetLine.line_transform(wordS.mid_);

            words.push_back(wordS);
            continue;
        }

        // Otherwise, there are at least 2 chars in the word

        // List of the indexes of the chars in this word
        std::vector<int>& this_word = charGroup.line_ids;

        // Find all of the points in this word to form a point cloud.
        std::vector<cv::Point> this_word_points;
        for (int j: this_word) {
            for (cv::Point pt: points_list[char_kept_index[j]])
                this_word_points.push_back(pt);
        }

        // Fit a line to the centers of each char
        // This helps measure the word's alignment
        std::vector<cv::Point> wordCenters;
        double maxCharHeight = 0;
        int wordArea = 0;
        for (int j: this_word) {
            wordCenters.push_back(char_lines[j].mid_); // uses center of bounding box
            maxCharHeight = std::max<double>(maxCharHeight, char_lines[j].height_);
            wordArea += char_lines[j].pixelCnt_;
        }

        // Fit a line to the word centers
        MyLine wordCent = fitPointCloud(wordCenters);
        // Measure how well the centers fall into a straight line
        // The smaller the centerHeight, the better aligned the chars are
        double centerHeight = wordCent.height_;

        // Determine the street that the word lies in
        cv::Point repPt = wordCent.mid_; // Center for the word
        bool middleOfStreet = searchForOwnerF(repPt, curbDist) > searchForOwnerF(repPt, midDist);
        int stOwn = searchForOwner(repPt, streetOwnership);
        if (stOwn == -1) {
            continue; // Should never happen...
        }
        MyLine& street = streets[stOwn];

        // Get the orientation of the street,
        // and the orientation of the nearest curb line.
        cv::Point2f streetDomi = street.domi_;
        cv::Point2f orientDomi = getOrientation(repPt, orientX, orientY);

        // flip orientation vector so that it matches the street's orientation
        // this ensures that all house numbers along the same street will either
        // all be upside down or all right side up
        float dotP = streetDomi.x*orientDomi.x + streetDomi.y*orientDomi.y;
        if (dotP < 0) {
            orientDomi *= -1; // Flip on misalignment
        } else if (std::abs<float>(orientDomi.x) + std::abs<float>(orientDomi.y) < 0.2) {
            // bad orientation all together, use the street orientation instead
            orientDomi = streetDomi;
        }

        // Create a MyLine to store the word
        MyLine word;
        if (middleOfStreet && numCharsInWord > 2) {
            // This word is a street name candidate.
            // Use the orientation of the fitted line to the point cloud.
            word = fitPointCloud(this_word_points);
            if (streetDomi.ddot(word.domi_) < 0) {
                word.flipDomi();
            }
        } else {
            // This word is a house number candidate. Use the orientation
            // of the closest curb line if it is trustworthy, or use the
            // orientation of the street otherwise.
            word = fitPointCloud(this_word_points, orientDomi, /*UseBoxCenter*/true);
            if (numCharsInWord > 2 && word.height_ > word.width_) {
                // This fit appears to be perpendicular to the street, instead of parallel.
                // Flip the word, to handle house numbers perpendicular to the curb line.
                // Then, fit it again.
                cv::Point2f newDomi {-orientDomi.y, orientDomi.x};
                orientDomi = newDomi;
                word = fitPointCloud(this_word_points, orientDomi, /*UseBoxCenter*/true);
            }
        }

        // Extend the size to avoid clipping
        word.width_ += 2;
        word.height_ += 2;

        // ----------- Is the word a street name, house number, or neither ? ------------
        // ----------- Does connected components work for this word? --------------------

        // Low score = better. Scores fall in the range 0-100.
        double heightConsistencyScore = 0;
        double spacingStreetNameScore = 0;
        double spacingHouseNumScore   = 0;
        double areaStreetNameScore    = 0;
        double areaHouseNumScore      = 0;
        double connCompScore          = 0;

        // NOTE: Nearly all of the following parameters are untested.
        // They would be difficult to test accurately.

        // Analyze how consistent the heights are.
        for (int j: this_word) {
            double tHeight = char_lines[j].height_;
            double htRatio = tHeight/maxCharHeight; // Ratio of the height of this char to the largest
            // Apply a penalty if the heights deviate from each other a lot.
            if (htRatio > 0.4)
                heightConsistencyScore += 100.0/(numCharsInWord-1) * clamp_score(0.75, 1.0, 2, htRatio);
            else connCompScore += 100; // Automatically fail connected components
        }
        connCompScore += clamp_score(0.0, 4.7, 10, centerHeight);

        // Find the spacings between each adjacent char
        std::vector<std::pair<MyLine, int>> lineXList;
        for (int j: this_word) {
            cv::Point tCenter = char_lines[j].mid_;
            cv::Point tLine = word.line_transform(tCenter); // Transform to word's coords
            if (char_lines[j].height_ > 0.5 * maxCharHeight)
                lineXList.push_back({char_lines[j], tLine.x});
        }
        // sort the characters based on their X coordinate in the word.
        // So, the chars will be sorted left to right in the word.
        std::sort(lineXList.begin(), lineXList.end(), lineXSort);

        // Analyze the consistency in the spacings.
        // We expect a particular spacing to width ratio for each
        // street name and house number.
        for (size_t j = 0; j < lineXList.size() - 1; j++) {
            std::pair<MyLine, int> a = lineXList[j];
            std::pair<MyLine, int> b = lineXList[j+1];

            // Find the distance between two adjacent chars,
            // relative to their widths
            double dist = std::abs<double>(b.second - a.second);
            double sumWid = (a.first.width_ + b.first.width_)/2;

            // Apply a penalty if the spacing falls outside of the expected range
            spacingStreetNameScore += 100.0/(lineXList.size() - 1) * clamp_score(0.87, 2.3, 2, dist/sumWid);
            spacingHouseNumScore   += 100.0/(lineXList.size() - 1) * clamp_score(0.37, 1.7, 2, dist/sumWid);
        }

        // Finally, look at areas and area/height ratios.
        // Are the sizes what we expect for a street name or house number?
        for (int j: this_word) {
            MyLine tChar = char_lines[j];

            // Look at the area to height ratio for each char
            double area = char_lines[j].pixelCnt_;
            double areaHtRatio = area/tChar.height_;

            // Apply a penalty if the ratio is not what we expect for a house number
            areaHouseNumScore += 100.0/(numCharsInWord) * clamp_score(2.5, 12.0, 0.35, areaHtRatio);

            // Ignore dashes (they get penalized elsewhere)
            if (tChar.width_ * 2.0 < tChar.height_) {
                continue; //dash
            }

            // Apply a penalty if the ratio is not what we expect for a street name
            // Due to the variable sizes of street names, use different thresholds
            if (maxCharHeight > 57)
                areaStreetNameScore += 100.0/(numCharsInWord) * clamp_score(10.0, 36.0, 0.35, areaHtRatio);
            else if (maxCharHeight > 45)
                areaStreetNameScore += 100.0/(numCharsInWord) * clamp_score(8.0, 30.0, 0.35, areaHtRatio);
            else
                areaStreetNameScore += 100.0/(numCharsInWord) * clamp_score(7.0, 21.0, 0.35, areaHtRatio);

        }

        // Consider location in the street
        if (middleOfStreet) spacingHouseNumScore += 20;
        if (!middleOfStreet) spacingStreetNameScore += 30;

        // Consider overall number of pixels in the word
        // Penalize small areas.
        if (wordArea < 250) spacingStreetNameScore += 40;
        if (wordArea < 95) spacingHouseNumScore += 40;

        // Aggregate all of the metrics to get a final prediction.
        bool isStreetName = spacingStreetNameScore < spacingHouseNumScore
                            && spacingStreetNameScore < 35
                            && heightConsistencyScore < 35;

        bool isHouseNumber = spacingHouseNumScore < spacingStreetNameScore
                            && spacingHouseNumScore < 35
                            && heightConsistencyScore < 45;

        // Too many tests fail, so guess not a word
        bool probablyNotWord = heightConsistencyScore > 50
                    || (isStreetName && spacingStreetNameScore > 35)
                    || (isHouseNumber && spacingHouseNumScore > 35);

        // If everything is perfect, we can say connected components works
        bool connectedComponentsWorks = heightConsistencyScore < 1
                        && ((isStreetName && spacingStreetNameScore < 1)
                                    || (isHouseNumber && spacingHouseNumScore < 1))
                        && connCompScore < 1
                        && ((isStreetName && areaStreetNameScore < 1)
                                    || (isHouseNumber && areaHouseNumScore < 1));

        // Update word values, so that
        // we can write the info to the JSON file
        if (isStreetName) {
            word.isStreetName_ = true;
        }
        else if (isHouseNumber) {
            word.isHouseNumber_ = true;
        }
        if (connectedComponentsWorks) {
            word.connectedComponentPass_ = true;
        }
        if (probablyNotWord || word.height_ < 15 || word.width_ < 25) {
            word.probablyNotWord_ = true;
        }

        word.heightConsistencyScore = heightConsistencyScore;
        word.spacingStreetNameScore = spacingStreetNameScore;
        word.spacingHouseNumScore = spacingHouseNumScore;
        word.areaStreetNameScore = areaStreetNameScore;
        word.areaHouseNumScore = areaHouseNumScore;
        word.connCompScore = connCompScore;

        // Remember which street segment we are in
        word.streetId_ = stOwn;
        // Calculate location relative to the street
        word.wordRelativeToStreet_ = street.line_transform(word.mid_);

        // Calculate location relative to extended historical street
        LongLine& extendedStreet = extendedStreets[street.long_line];
        Segment extendedStreetSegment = extendedStreet.segment();
        MyLine extendedStreetLine{extendedStreetSegment.p1, extendedStreetSegment.p2, true, false};
        word.wordRelativeToExtendedStreet_ = extendedStreetLine.line_transform(word.mid_);

        // Add word to our list
        word.wordId_ = wordCount++;
        words.push_back(word);

        //wordPixels.push_back(this_word_points);
    }

    log("Individual words analyzed", LogLevel::debug);

    if (DRAW_DEBUG_IMAGES) {
        cv::Mat colorSt;
        color_connected_components(imMarkSt, colorSt, {0,0,0}, {50,50,50}, {130,130,255});
        for (size_t i = 0; i < char_lines.size(); ++i) {
            std::vector<cv::Point> bpt = char_lines[i].boundingBox();
            cv::line(colorSt, bpt[0], bpt[1], CV_RGB(0,255,255), 1);
            cv::line(colorSt, bpt[1], bpt[2], CV_RGB(0,205,255), 1);
            cv::line(colorSt, bpt[2], bpt[3], CV_RGB(0,255,255), 1);
            cv::line(colorSt, bpt[3], bpt[0], CV_RGB(0,205,255), 1);

            for (int j: charLongLines[char_lines[i].long_line].line_ids) {
                cv::line(colorSt, char_lines[i].mid_, char_lines[j].mid_, CV_RGB(0,255,30), 1);
            }
            cv::circle(colorSt, char_lines[i].mid_, 1, CV_RGB(0,255,255), 2);
        }
        debug_imwrite(colorSt, "a9cc_chars");
        //make_dir(DEBUG_PATH, "!word_finding");
        //specific_imwrite(colorSt, "!word_finding", "_chars");
        colorSt.release();
        log("Debug imwrite", LogLevel::debug);
    }

}


