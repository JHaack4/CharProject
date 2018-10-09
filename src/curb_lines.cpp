#include <cassert>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "image_proc.h"

// This file contains the code that takes as input the vectorized line segments,
// converted into long lines (which are essentially groups of vectorized line segments
// that together form a long and approximately straight lines). The collective output
// is a determination of whether each long line in the map is a curb line. A curb line
// is the line between the edge of the street and any other feature in the map, such as
// a tax parcel.

// Each long line is checked on both sides for curb line status. The
// "positive" side is the side that corresponds with the street's normal
// vector. The "negative" side is the other side. If the long line has a
// high curbLineP score, this means that the line is probably a curb line,
// and there is a street on the positive side of the line.

// This is the minimum length in meters for a long line to
// be considered a curb line.
float LONG_LINE_LENGTH_THRESH = -9999999; // value not known at compile time, so assigned below

// This function assigns a score to each long line, based on how well they match
// with the modern day street segments
void curbMatchModern(std::vector<MyLine>& lines, std::vector<LongLine>& lls, std::vector<Street>& presentStreets) {
    LONG_LINE_LENGTH_THRESH = 35/RESOLUTION * STREET_SCALE_FACTOR;

    // Designate a maximum distance (in meters) and angle (in degrees) for the matching
    const float DIST_TO_MODERN_THRESH = 15/RESOLUTION * STREET_SCALE_FACTOR; // in meters, tested
    const float ANGLE_TO_MODERN_THRESH = 10; // in degrees, tested

    // Iterate over each line segment
    for (MyLine& l: lines) {
        if (l.long_line == -1) continue; // ignore invalid lines
        LongLine& ll = lls[l.long_line]; // find the long line that the line is in
        if (ll.length_ < LONG_LINE_LENGTH_THRESH) continue; // ignore if the long line is too short

        // We will check for a match with modern streets on both sides
        bool posMatch = false, negMatch = false;

        // Check for a match against all present streets
        // No need for binning, as this is fast enough, due to the small amount of
        // modern street segments
        for (Street s: presentStreets) {

            // Compute angle, distance, and alignment between
            // the present street segment and the line
            Segment lseg = l.segment();
            double angle = segments_angle(s.segment, lseg);
            double dist = segments_distance(s.segment, lseg);
            double wproj = worst_point_segment_horizontal_proj(s.segment, lseg);
            // Note, wproj checks that the projection of each segment onto the
            // other aligns properly.

            // Test if the match is good enough.
            if (angle < ANGLE_TO_MODERN_THRESH && dist < DIST_TO_MODERN_THRESH && wproj < 0.1) {

                // We need to determine which side is being matched, the positive
                // one or the negative one. To do this, we use the normOriented function
                // on the long line, which checks if the gap between the things we
                // are matching is appropriately oriented with the normal vector of the long line
                cv::Point2f smid = segment_midpoint(s.segment);
                cv::Point2f gap = smid - l.mid_;

                if (ll.normOriented(gap)) {
                    posMatch = true;
                } else {
                    negMatch = true;
                }


            }

        }

        // Assign matching bonuses proportional to length
        if (posMatch) {
            ll.curbLineModernScoreP += l.width_ / ll.length_;
        }
        if (negMatch) {
            ll.curbLineModernScoreN += l.width_ / ll.length_;
        }

    }

}


// This function assigns curb line scores based on how well the long line
// corresponds to the large flood fill components. Streets tend to lie in a
// large flood fill component, so curb lines tend to have a large flood fill component on
// one side but not the other.
void curbFloodFill(std::vector<MyLine>& lines, std::vector<LongLine>& lls, cv::Mat& imFloodFill, cv::Mat& imMask) {

    const int shortDist = 50 * STREET_SCALE_FACTOR; //tested
    const int longDist = 150 * STREET_SCALE_FACTOR; //tested

    for (MyLine& l: lines) {
        if (l.long_line == -1) continue;
        LongLine& ll = lls[l.long_line];
        if (ll.length_ < LONG_LINE_LENGTH_THRESH) continue;

        // Determine points to probe at. The probe is perpendicular to the line,
        // starting at the mid point of the line.
        cv::Point testPL = l.absolute_tranform(cv::Point2f{l.width_/2, longDist});
        cv::Point testPS = l.absolute_tranform(cv::Point2f{l.width_/2, shortDist});
        cv::Point testNS = l.absolute_tranform(cv::Point2f{l.width_/2, -shortDist});
        cv::Point testNL = l.absolute_tranform(cv::Point2f{l.width_/2, -longDist});

        // Check the flood fill image. We also check the mask, to avoid penalizing
        // curb lines that are near the edges of the map.
        bool fPL = pixelInBounds(testPL, imFloodFill) && imFloodFill.at<pixel_t>(testPL)  > 0;
        bool fPS = pixelInBounds(testPS, imFloodFill) && imFloodFill.at<pixel_t>(testPS)  > 0;
        bool fNS = pixelInBounds(testNS, imFloodFill) && imFloodFill.at<pixel_t>(testNS)  > 0;
        bool fNL = pixelInBounds(testNL, imFloodFill) && imFloodFill.at<pixel_t>(testNL)  > 0;
        bool mPL = pixelInBounds(testPL, imMask) && imMask.at<pixel_t>(testPL)  == 0;
        bool mPS = pixelInBounds(testPS, imMask) && imMask.at<pixel_t>(testPS)  == 0;
        bool mNS = pixelInBounds(testNS, imMask) && imMask.at<pixel_t>(testNS)  == 0;
        bool mNL = pixelInBounds(testNL, imMask) && imMask.at<pixel_t>(testNL)  == 0;

        // Get a weighted score. Note, having a flood fill on both sides of the
        // line incurs a slight penalty.
        float Pscore = 0.5*fPL + 0.5*fPS + 0.4*mPL + 0.4*mPS - 0.5*fNL + 0.3*mNL;
        float Nscore = 0.5*fNL + 0.5*fNS + 0.4*mNL + 0.4*mNS - 0.5*fPL + 0.3*mPL;

        // Check the orientation relative to the long line.
        // Assign scores proportional to length.
        if (ll.normOriented(l.norm_)) {
            ll.curbLineFloodFillScoreP += Pscore * l.width_ / ll.length_;
            ll.curbLineFloodFillScoreN += Nscore * l.width_ / ll.length_;
        } else {
            ll.curbLineFloodFillScoreN += Pscore * l.width_ / ll.length_;
            ll.curbLineFloodFillScoreP += Nscore * l.width_ / ll.length_;
        }

    }



}

// This checks the branch points and end point of the long line, and assigns
// a curb line score. Curb lines tend to have a lot of branch points due to
// tax parcels. These branch points should all be directed the same way, away
// from the street. The end points should be hard, nearly right angle corners,
// because intersecting streets are generally drawn with clear corner points.
void curbBranchPoints(std::vector<MyLine>& lines, std::vector<LongLine>& lls, std::vector<MyVertex>& verts) {

    // We assign a penalty based on the length of the line coming from the
    // branch point. This determines the max value of that penalty.
    const float MAX_LENGTH_PENALTY = 60/RESOLUTION * STREET_SCALE_FACTOR; // in meters

    // These parameters given score bonuses to lines that have
    // many branch points going in the same direction, and penalties
    // for branch points in the other direction.
    const float BRANCH_BONUS = 0.08; //tested
    const float LENGTH_BONUS = 0.005; //tested
    const float PENALTY = 0.4; //tested

    // iterate over each valid long line
    for (LongLine& ll: lls) {
        if (ll.ignore_me || ll.length_ < LONG_LINE_LENGTH_THRESH) continue;

        // track the number of branch points found for each side of the line
        int posCnt = 0;
        int negCnt = 0;
        // track the length of line segments on each side of the line
        float posCumLength = 0;
        float negCumLength = 0;
        // Track if we have seen the endpoints of the long line yet
        bool endptChecked1 = 0;
        bool endptChecked2 = 0;

        // iterate over each line segment in the long line
        for (int line_id: ll.line_ids) {
            MyLine& l = lines[line_id];

            // Look at the first vertex.
            MyVertex& v1 = verts[l.vert1_];
            // Iterate over each line adjacent to that vertex
            for (int adj_id: v1.adj_) {
                // Ignore the line if it is already part of the same long line
                MyLine& adj_line = lines[adj_id];
                if (adj_line.long_line == l.long_line) continue;
                // Create a vector that goes from the vertex to the other endpoint
                // of the other line segment at the branch point
                cv::Point endptVec = adj_line.endpointVector(v1.loc_);
                // Look at the length of that line
                float len = lls[adj_line.long_line].length_;
                if (len > MAX_LENGTH_PENALTY) len = MAX_LENGTH_PENALTY;
                // Determine which side of the street that this line is on
                if (ll.normOriented(endptVec)) {
                    posCnt++;
                    posCumLength += len;
                } else {
                    negCnt++;
                    negCumLength += len;
                }
            }

            // Duplicated code for the other end point.
            MyVertex& v2 = verts[l.vert2_];
            for (int adj_id: v2.adj_) {
                MyLine& adj_line = lines[adj_id];
                if (adj_line.long_line == l.long_line) continue;
                cv::Point endptVec = adj_line.endpointVector(v2.loc_);
                float len = lls[adj_line.long_line].length_;
                if (len > MAX_LENGTH_PENALTY) len = MAX_LENGTH_PENALTY;
                if (ll.normOriented(endptVec)) {
                    posCnt++;
                    posCumLength += len;
                } else {
                    negCnt++;
                    negCumLength += len;
                }
            }

            // Check if the first vertex is one of the end points of the long line.
            // Track whether or not we have checked both endpoints of the long line,
            // to make sure we don't run this code twice (should never happen)
            if ((v1.loc_ == ll.endpt1 && !endptChecked1) || (v1.loc_ == ll.endpt2 && !endptChecked2)) {
                MyVertex& v = v1;

                if (v1.loc_ == ll.endpt1) {
                    endptChecked1 = true;
                } else endptChecked2 = true;

                // If the endpoint ends at a vertex of degree 1 (an endpoint)
                if (v.adj_.size() == 1) {
                    // Apply a slight penalty for not having an end point vertex
                    ll.curbLineEndPointScoreN -= PENALTY/2;
                    ll.curbLineEndPointScoreP -= PENALTY/2;
                }

                // Otherwise, we consider each line adjacent to the end point vertex
                for (int adj_id: v.adj_) {
                    // Ignore line segments that are part of the same long line.
                    MyLine& adj_line = lines[adj_id];
                    if (adj_line.long_line == l.long_line) continue;
                    // Get a vector from the vertex to the other end point of the line
                    cv::Point endptVec = adj_line.endpointVector(v.loc_);
                    // Check the length
                    float len = lls[adj_line.long_line].length_;
                    if (len > MAX_LENGTH_PENALTY) len = MAX_LENGTH_PENALTY;
                    // Check which side of the curb line the end point is on
                    if (ll.normOriented(endptVec)) {
                        // Apply a bonus if the corner is directed away from the side,
                        // and a penalty the other way. Generally, the corner faces the
                        // opposite direction of the street.
                        ll.curbLineEndPointScoreN += 0.5 * len / MAX_LENGTH_PENALTY;
                        ll.curbLineEndPointScoreP -= (0.5+PENALTY) * len / MAX_LENGTH_PENALTY;
                    } else {
                        ll.curbLineEndPointScoreP += 0.5 * len / MAX_LENGTH_PENALTY;
                        ll.curbLineEndPointScoreN -= (0.5+PENALTY) * len / MAX_LENGTH_PENALTY;
                    }
                }
            }

            // Duplicated code for the other endpoint
            if ((v2.loc_ == ll.endpt1 && !endptChecked1) || (v2.loc_ == ll.endpt2 && !endptChecked2)) {
                MyVertex& v = v2;

                if (v1.loc_ == ll.endpt1) {
                    endptChecked1 = true;
                } else endptChecked2 = true;

                if (v.adj_.size() == 1) {
                    ll.curbLineEndPointScoreN -= PENALTY/2;
                    ll.curbLineEndPointScoreP -= PENALTY/2;
                }

                for (int adj_id: v.adj_) {
                    MyLine& adj_line = lines[adj_id];
                    if (adj_line.long_line == l.long_line) continue;
                    cv::Point endptVec = adj_line.endpointVector(v.loc_);
                    float len = lls[adj_line.long_line].length_;
                    if (len > MAX_LENGTH_PENALTY) len = MAX_LENGTH_PENALTY;

                    if (ll.normOriented(endptVec)) {
                        ll.curbLineEndPointScoreN += 0.5 * len / MAX_LENGTH_PENALTY;
                        ll.curbLineEndPointScoreP -= (0.5+PENALTY) * len / MAX_LENGTH_PENALTY;
                    } else {
                        ll.curbLineEndPointScoreP += 0.5 * len / MAX_LENGTH_PENALTY;
                        ll.curbLineEndPointScoreN -= (0.5+PENALTY) * len / MAX_LENGTH_PENALTY;
                    }
                }
            }

        }

        // Aggregate all of the information gathered for this long line.

        // Difference in length of the two sides
        float diff = posCumLength - negCumLength;

        // If there are no branch points on either side,
        // assign a slight penalty. Usually, the curb
        // lines have some branch points.
        if (posCnt == 0 && negCnt == 0) {
            ll.curbLineBranchPointScoreN = -PENALTY;
            ll.curbLineBranchPointScoreP = -PENALTY;
            continue;
        }

        // Assign score for the negative side of the curb line
        if (negCnt <= 1 || negCumLength < MAX_LENGTH_PENALTY) {
            // In this case, most of the branch points are on the positive side
            // which means we should give the negative side a bonus
            ll.curbLineBranchPointScoreN = posCnt * BRANCH_BONUS + diff * LENGTH_BONUS;
            if (ll.curbLineBranchPointScoreN > 1) ll.curbLineBranchPointScoreN = 1; // Cap score at 1
            ll.curbLineBranchPointScoreN -= negCnt * BRANCH_BONUS/3 + negCumLength * LENGTH_BONUS/3;
        } else {
            // This case is less ideal. We will assign a smaller bonus
            // based on the ration of the branch points on either side.
            // We also have a penalty for extra branch points on the negative side.
            float posRatio = posCnt * 1.0f / negCnt;
            ll.curbLineBranchPointScoreN = (posRatio-1) * BRANCH_BONUS + diff * LENGTH_BONUS;
            if (ll.curbLineBranchPointScoreN > 1-PENALTY) ll.curbLineBranchPointScoreN = 1-PENALTY; // Cap score
            ll.curbLineBranchPointScoreN -= negCumLength * LENGTH_BONUS/3;
        }

        // Duplicated code for the positive side
        if (posCnt <= 1 || posCumLength < MAX_LENGTH_PENALTY) { // Really good case
            ll.curbLineBranchPointScoreP = negCnt * BRANCH_BONUS - diff * LENGTH_BONUS;
            if (ll.curbLineBranchPointScoreP > 1) ll.curbLineBranchPointScoreP = 1;
            ll.curbLineBranchPointScoreP -= posCnt * BRANCH_BONUS/3 + posCumLength * LENGTH_BONUS/3;
        } else { // not so good case
            float negRatio = negCnt * 1.0f / posCnt;
            ll.curbLineBranchPointScoreP = (negRatio-1) * BRANCH_BONUS - diff * LENGTH_BONUS;
            if (ll.curbLineBranchPointScoreP > 1-PENALTY) ll.curbLineBranchPointScoreP = 1-PENALTY;
            ll.curbLineBranchPointScoreP -= posCumLength * LENGTH_BONUS/3;
        }

        // Floor scores at -1
        if (ll.curbLineBranchPointScoreN < -1) ll.curbLineBranchPointScoreN = -1;
        if (ll.curbLineBranchPointScoreP < -1) ll.curbLineBranchPointScoreP = -1;

    }



}

// Assign a curb line score based on the markings. Curb lines usually have
// markings (house numbers) near them.
void curbMatchMarkings(std::vector<MyLine>& lines, std::vector<LongLine>& lls,
                       std::vector<cv::Point>& markingCentroids, std::vector<int>& markingSizes,
                       cv::Mat& smLines) {

    // Search distances (in pixels)
    const float CLOSE_DIST = 40 * STREET_SCALE_FACTOR; //tested
    const float FAR_DIST = 150 * STREET_SCALE_FACTOR; //tested

    // Scoring weights based on distance from the curb to the marking, and size of the marking
    const int MAX_CHAR_COUNT = 4; // max number of chars per line that count to the score, tested
    const float CLOSE_WEIGHT = 0.5; // tested
    const float FAR_WEIGHT = 0.1; // tested
    const float SIZE_WEIGHT = 0.0001; // tested

    // Since there can be thousands of markings, we can't use
    // a simple for loop. Instead, we place the markings into
    // bins, and then only search the bins that are close to the
    // curb.
    int binSize = 20;
    int xBins = smLines.cols / binSize + 2;
    int yBins = smLines.rows / binSize + 2;

    // Store the bins in a matrix of vectors, referring to the
    // markings by their integer index.
    std::vector<std::vector<std::vector<int>>> bins(yBins, std::vector<std::vector<int>>(xBins, std::vector<int>(0)));

    // Place the markings into the appropriate bins
    for (size_t i = 0; i < markingCentroids.size(); ++i) {
        cv::Point v = markingCentroids[i];
        v *= STREET_SCALE_FACTOR; // Convert the marking to downscaled coords
        int xb = v.x / binSize;
        int yb = v.y / binSize;
        if (xb >= 0 && yb >= 0 && xb < xBins && yb < yBins)
            (bins[yb][xb]).push_back((int)i);
    }

    // Remember which bins have been checked already for a given line
    // Note, this matrix method is much faster than using a hash set
    // or other data structure.
    std::vector<std::vector<int>> binsChecked (yBins, std::vector<int>(xBins, 0));

    // Iterate over each valid line
    for (MyLine& l: lines) {
        if (l.long_line == -1) continue;
        LongLine& ll = lls[l.long_line];
        if (ll.length_ < LONG_LINE_LENGTH_THRESH) continue;

        // Track the number of markings we see
        // at each distance, along with their size
        int closeCntP = 0;
        int closeCntN = 0;
        int farCntP = 0;
        int farCntN = 0;
        int sizeP = 0;
        int sizeN = 0;

        // Iterate across the line on either side.
        for (int lx = 0; lx < l.width_; lx += binSize/2) {
            for (int ly = -FAR_DIST; ly <= FAR_DIST; ly += binSize/2) {

                // Transform the line coords to absolute coords on the map
                cv::Point linec{lx, ly};
                cv::Point absc = l.absolute_tranform(linec);

                // Determine the bin that the marking is in
                int xb = absc.x / binSize;
                int yb = absc.y / binSize;

                // Ignore bins we have already checked
                // Use the line ID plus one to track which bins are checked.
                if (yb < 0 || xb < 0 || yb >= yBins || xb >= xBins) continue;
                if (binsChecked[yb][xb] == l.line_id + 1) {
                    continue;
                }
                binsChecked[yb][xb] = l.line_id + 1;

                // Now, check every marking in this bin
                for (int& mark_id: bins[yb][xb]) {

                    // Find the coords of the marking on the downscaled image
                    cv::Point2f mark = markingCentroids[mark_id];
                    mark *= STREET_SCALE_FACTOR;

                    // Project the marking onto the line
                    cv::Point2f gap = mark - l.mid_;
                    cv::Point proj = l.line_transform(mark);

                    // Check if the marking is too far away from the line.
                    if (std::abs<int>(proj.y) > FAR_DIST || std::abs<int>(proj.x) > l.width_/2)
                        continue;

                    // Determine which side of the line the marking is on
                    if (ll.normOriented(gap)) {
                        // Update tracking variables for this line
                        if (std::abs<int>(proj.y) <= CLOSE_DIST) closeCntP++;
                        else if (std::abs<int>(proj.y) <= FAR_DIST) farCntP++;
                        sizeP += markingSizes[mark_id];
                    }
                    else {
                        if (std::abs<int>(proj.y) <= CLOSE_DIST) closeCntN++;
                        else if (std::abs<int>(proj.y) <= FAR_DIST) farCntN++;
                        sizeN += markingSizes[mark_id];
                    }

                }

            }
        }

        // Limit sizes to max value to avoid giving too high of a score
        if (closeCntP > MAX_CHAR_COUNT) closeCntP = MAX_CHAR_COUNT;
        if (closeCntN > MAX_CHAR_COUNT) closeCntN = MAX_CHAR_COUNT;
        if (farCntP   > MAX_CHAR_COUNT) farCntP   = MAX_CHAR_COUNT;
        if (farCntN   > MAX_CHAR_COUNT) farCntN   = MAX_CHAR_COUNT;

        // Assign a marking score, proportional to length of the line.
        double thisScoreP = (closeCntP * CLOSE_WEIGHT + sizeP * SIZE_WEIGHT + farCntP * FAR_WEIGHT) / MAX_CHAR_COUNT;
        double thisScoreN = (closeCntN * CLOSE_WEIGHT + sizeN * SIZE_WEIGHT + farCntN * FAR_WEIGHT) / MAX_CHAR_COUNT;
        ll.curbLineMarkingScoreP += thisScoreP * l.width_ / ll.length_ ;
        ll.curbLineMarkingScoreN += thisScoreN * l.width_ / ll.length_ ;

    }

}

// The minimum and maximum distances for a parallel match (in meters), and
// the maximum angle in degrees.  Value of MIN/MAX_DIST_PARA not known at compile time
// since they depend on the resolution, so assigned below
float MIN_DIST_PARA = -99999999999;
float MAX_DIST_PARA = -99999999999;
const float MAX_ANGLE_PARA = 20; // degrees

// Assign a curb line score for each long line, testing if the long
// line has any other curb lines that are parallel to it. This captures the
// idea that each street should have two curb lines that are parallel to each other.
void curbMatchPara(std::vector<MyLine>& lines, std::vector<LongLine>& lls) {

    // Parameters tested, in meters
    MIN_DIST_PARA = 5/RESOLUTION * STREET_SCALE_FACTOR;
    MAX_DIST_PARA = 35/RESOLUTION * STREET_SCALE_FACTOR;

    // Tolerance for allowing a long line to match parallel
    const float SKIP_MATCH_THRESH = 0.3;    //tested

    // Iterate over each pair of valid long lines.
    for (int i = 0; i < (int)lls.size(); ++i) {
        LongLine& ll1 = lls[i];
        if (ll1.ignore_me) continue;
        if (ll1.length_ < LONG_LINE_LENGTH_THRESH) continue;

        // Check for bad curb lines. We only want to do parallel matching of
        // long lines that have a decent chance of being a curb line.
        if (ll1.curbLineBranchPointScoreN + ll1.curbLineEndPointScoreN + ll1.curbLineModernScoreN <= SKIP_MATCH_THRESH &&
            ll1.curbLineBranchPointScoreP + ll1.curbLineEndPointScoreP + ll1.curbLineModernScoreP <= SKIP_MATCH_THRESH) continue;

        for (int j = i+1; j < (int)lls.size(); ++j) {
            LongLine& ll2 = lls[j];
            if (ll2.ignore_me) continue;
            if (ll2.length_ < LONG_LINE_LENGTH_THRESH) continue;

            // Check if the second line is also a decent candidate for a curb line.
            if (ll2.curbLineBranchPointScoreN + ll2.curbLineEndPointScoreN + ll2.curbLineModernScoreN
                    <= SKIP_MATCH_THRESH &&
                ll2.curbLineBranchPointScoreP + ll2.curbLineEndPointScoreP + ll2.curbLineModernScoreP
                    <= SKIP_MATCH_THRESH)
                        continue;

            cv::Point mid1 = segment_midpoint(ll1.segment());
            cv::Point mid2 = segment_midpoint(ll2.segment());

            // Ensure that the two chosen lines face the same direction. In other words,
            // the two curb lines we are trying to match should have a street in between them
            if (ll1.normOriented(mid2 - mid1)) {
                if (ll1.curbLineBranchPointScoreP + ll1.curbLineEndPointScoreP + ll1.curbLineModernScoreP
                    <= SKIP_MATCH_THRESH) continue;
            }
            else {
                if (ll1.curbLineBranchPointScoreN + ll1.curbLineEndPointScoreN + ll1.curbLineModernScoreN
                    <= SKIP_MATCH_THRESH) continue;
            }
            if (ll2.normOriented(mid1 - mid2)) {
                if (ll2.curbLineBranchPointScoreP + ll2.curbLineEndPointScoreP + ll2.curbLineModernScoreP
                    <= SKIP_MATCH_THRESH) continue;
            }
            else {
                if (ll2.curbLineBranchPointScoreN + ll2.curbLineEndPointScoreN + ll2.curbLineModernScoreN
                    <= SKIP_MATCH_THRESH) continue;
            }

            // We want to track what portion of line segments in each long candidate curb line
            // match with a segment from the other long line.
            std::vector<int> matches1 (ll1.line_ids.size(), 0);
            std::vector<int> matches2 (ll2.line_ids.size(), 0);

            // Pairwise compare all line segments in the two potential curb lines
            for (int ii = 0; ii < (int)ll1.line_ids.size(); ++ii) {
                for (int jj = 0; jj < (int)ll2.line_ids.size(); ++jj) {

                    MyLine& l1 = lines[ll1.line_ids[ii]];
                    MyLine& l2 = lines[ll2.line_ids[jj]];
                    Segment s1 = l1.segment();
                    Segment s2 = l2.segment();

                    // Compute angle and distance between the segments
                    float angle = segments_angle(s1, s2);
                    float dist = segments_distance(s1, s2);
                    float wproj = worst_point_segment_horizontal_proj(s1, s2)
                                    + worst_point_segment_horizontal_proj(s2, s1);

                    // Check if there is a good match based on angle and distance
                    if (wproj < 0.5 && angle < MAX_ANGLE_PARA && dist > MIN_DIST_PARA && dist < MAX_DIST_PARA) {
                        // Good match. Record this
                        matches1[ii] = 1;
                        matches2[jj] = 1;
                    }
                }
            }

            float thisMatchScore1 = 0.0;
            float thisMatchScore2 = 0.0;

            // Assign a parallel match score proportional to the length of each matching segment
            for (int ii = 0; ii < (int)(ll1.line_ids.size()); ++ii) {
                thisMatchScore1 += matches1[ii] * (lines[ll1.line_ids[ii]].width_) / ll1.length_;
            }
            for (int ii = 0; ii < (int)(ll2.line_ids.size()); ++ii) {
                thisMatchScore2 += matches2[ii] * (lines[ll2.line_ids[ii]].width_) / ll2.length_;
            }

            // Recompute the orientations between the curb lines. Cap the scores at 1.
            if (ll1.normOriented(mid2 - mid1)) {
                ll1.curbLineParaMatchScoreP += thisMatchScore1;
                if (ll1.curbLineParaMatchScoreP > 1) ll1.curbLineParaMatchScoreP = 1;
            } else {
                ll1.curbLineParaMatchScoreN += thisMatchScore1;
                if (ll1.curbLineParaMatchScoreN > 1) ll1.curbLineParaMatchScoreN = 1;
            }
            if (ll2.normOriented(mid1 - mid2)) {
                ll2.curbLineParaMatchScoreP += thisMatchScore2;
                if (ll2.curbLineParaMatchScoreP > 1) ll2.curbLineParaMatchScoreP = 1;
            } else {
                ll2.curbLineParaMatchScoreN += thisMatchScore2;
                if (ll2.curbLineParaMatchScoreN > 1) ll2.curbLineParaMatchScoreN = 1;
            }

        }

    }

}

// Aggregate the curb line scores for the six features along with the length of the long line.
// to get a final determination if the long line is a curb line.
void curbAggregate(std::vector<MyLine>& /*lines*/, std::vector<LongLine>& lls) {

    // Weights for each of the 7 considerations, all tested
    // Note: these parameters are all pretty volatile
    const float MODERN_WEIGHT = 0.4;
    const float BRANCH_WEIGHT = 0.3;
    const float END_WEIGHT = 0.1;
    const float MARKING_WEIGHT = 0.2;
    const float PARA_WEIGHT = 0.1;
    const float FLOODFILL_WEIGHT = 0.1;
    const float LENGTH_WEIGHT = 0.0001;

    const float FINAL_THRESH = 0.4;

    // Iterate over each valid long line.
    for (int i = 0; i < (int)lls.size(); ++i) {
        LongLine& ll = lls[i];
        if (ll.ignore_me) continue;
        if (ll.length_ < LONG_LINE_LENGTH_THRESH) continue;

        // Use a linear combination of the factors.
        // Other ways to aggregate could be considered in the future
        ll.curbLineScoreP =   MODERN_WEIGHT * ll.curbLineModernScoreP
                            + BRANCH_WEIGHT * ll.curbLineBranchPointScoreP
                            + END_WEIGHT    * ll.curbLineEndPointScoreP
                            + FLOODFILL_WEIGHT * ll.curbLineFloodFillScoreP
                            + MARKING_WEIGHT * ll.curbLineMarkingScoreP
                            + PARA_WEIGHT * ll.curbLineParaMatchScoreP
                            + ll.length_ / (ll.gap_jump_count+1) * LENGTH_WEIGHT;

        ll.curbLineScoreN =   MODERN_WEIGHT * ll.curbLineModernScoreN
                            + BRANCH_WEIGHT * ll.curbLineBranchPointScoreN
                            + END_WEIGHT    * ll.curbLineEndPointScoreN
                            + FLOODFILL_WEIGHT * ll.curbLineFloodFillScoreN
                            + MARKING_WEIGHT * ll.curbLineMarkingScoreN
                            + PARA_WEIGHT * ll.curbLineParaMatchScoreN
                            + ll.length_ / (ll.gap_jump_count+1) * LENGTH_WEIGHT;

        ll.curbLineP = ll.curbLineScoreP > FINAL_THRESH;
        ll.curbLineN = ll.curbLineScoreN > FINAL_THRESH;

    }


}

// Given a list of the curb lines, we need to determine which pixels fall into the streets.
// We store this info in imStreet. We also determine the orientation of the closest curb line for
// each pixel as well as the distance to the curb.
void curbToStreets(std::vector<MyLine>& lines, std::vector<LongLine>& lls, cv::Mat& imStreet, cv::Mat& imOrientX, cv::Mat& imOrientY, cv::Mat& curbDist) {

    // Parameters for filling. Fill_start is the space between the curb and where the street
    // starts. Width extension is how much to expand the street widthwise (this is
    // used to cover intersections). No_match_dist is how large to make the street if
    // we don't find a parallel curb line. Match_fill proportion is the proportion of the
    // gap between the streets to fill if there is actually a match.
    const int gap = 2;
    const int width_extension = 25/RESOLUTION * STREET_SCALE_FACTOR; // in meters
    const int no_match_dist = 15/RESOLUTION * STREET_SCALE_FACTOR; // in meters
    const float match_fill_proportion = 0.8;
    const int fill_start = 40 * STREET_SCALE_FACTOR; // meters
    const int final_blur = 50 * STREET_SCALE_FACTOR; // meters

    // For each valid line segment
    for (MyLine& l: lines) {
        if (l.long_line == -1) continue;
        LongLine& ll = lls[l.long_line];
        if (ll.length_ < LONG_LINE_LENGTH_THRESH) continue;

        // Check if the line is part of a curb line in the positive direction
        if (ll.curbLineP) {

            // Determine the direction to fill
            int dir = ll.normOriented(l.norm_) ? 1 : -1;

            // We will check for a matching parallel curb line.
            float distSum = 0;
            int distCnt = 0;

            // Iterate over each other curb line
            for (LongLine& ll2: lls) {
                if (ll2.ignore_me || ll2.id_ == ll.id_) continue;
                if (!ll2.curbLineN && !ll2.curbLineP) continue;

                // Iterate over the line segments in the other curb line
                for (int line_id2: ll2.line_ids) {
                    MyLine& l2 = lines[line_id2];

                    // Make sure that the orientations match,
                    // i.e. that there is a street between the two curb lines
                    if (!ll.normOriented(l2.mid_ - l.mid_)) continue;
                    if (ll2.normOriented(l2.mid_ - l.mid_)) {
                        if (!ll2.curbLineN) continue;
                    } else {
                        if (!ll2.curbLineP) continue;
                    }

                    Segment s1 = l.segment();
                    Segment s2 = l2.segment();

                    // Apply same matching metrics
                    float angle = segments_angle(s1, s2);
                    float tdist = segments_distance(s1, s2);
                    float tproj = worst_point_segment_horizontal_proj(s1, s2)
                                    + worst_point_segment_horizontal_proj(s2, s1);

                    if (tproj < 0.5 && angle < MAX_ANGLE_PARA && tdist > MIN_DIST_PARA && tdist < MAX_DIST_PARA) {
                        // If there is a match, record the distance
                        distSum += tdist;
                        distCnt++;
                    }
                }
            }

            // The distance in pixels of how wide this street will be
            // for this curb line.
            int dist = 0;

            if (distCnt > 0) {
                // If we found a matching curb line, then use the average distance to its matches
                // times the proportion above
                dist = distSum / distCnt * match_fill_proportion;
            } else {
                // If there is no match, determine how far to fill
                dist = ll.curbLineN ? no_match_dist/2 : no_match_dist; // Case of no match
            }

            // Fill in the values in our images
            // Iterate over the curb line in the direction specified
            // using the distance specified.
            for (int lx = -width_extension; lx <= l.width_ + width_extension; lx += gap) {
                for (int ly = 0; ly <= dist; ly += gap) {

                    // Transform the line coords into absolute coords on the image
                    cv::Point linec{lx, ly*dir};
                    cv::Point absc = l.absolute_tranform(linec);
                    // Also, convert to ownership coords
                    cv::Point ownP = point_scale(absc, OWNERSHIP_SCALE_FACTOR / STREET_SCALE_FACTOR);

                    if (pixelInBounds(absc, imStreet) && ly >= fill_start)
                        imStreet.at<pixel_t>(absc) = 255; // fill in the street

                    if (pixelInBounds(ownP, imOrientX)) {
                        // Previous curb distance
                        float oldCurbDist = curbDist.at<float>(ownP);
                        // New curb distance
                        float newCurbDist = (ly + 2*std::max<float>({0.0f, lx-l.width_, 0.0f-lx}))
                                                / STREET_SCALE_FACTOR + 0.01f;
                        if (newCurbDist < oldCurbDist || oldCurbDist == 0) {
                            // Fill in the curb distance and curb orientation matrices,
                            // but don't overwrite a previously written value unless we are
                            // closer to the edge
                            curbDist.at<float>(ownP) = newCurbDist;
                            imOrientX.at<float>(ownP) = l.domi_.x;
                            imOrientY.at<float>(ownP) = l.domi_.y;
                        }
                    }
                }
            }

        }

        // Duplicated code for the negative case
        if (ll.curbLineN) {

            // Determine the direction to fill
            int dir = ll.normOriented(l.norm_) ? -1 : 1;

            // We will check for a matching parallel curb line.
            float distSum = 0;
            int distCnt = 0;

            // Iterate over each other curb line
            for (LongLine& ll2: lls) {
                if (ll2.ignore_me || ll2.id_ == ll.id_) continue;
                if (!ll2.curbLineN && !ll2.curbLineP) continue;

                // Iterate over the line segments in the other curb line
                for (int line_id2: ll2.line_ids) {
                    MyLine& l2 = lines[line_id2];

                    // Make sure that the orientations match,
                    // i.e. that there is a street between the two curb lines
                    if (ll.normOriented(l2.mid_ - l.mid_)) continue;
                    if (ll2.normOriented(l2.mid_ - l.mid_)) {
                        if (!ll2.curbLineN) continue;
                    } else {
                        if (!ll2.curbLineP) continue;
                    }

                    Segment s1 = l.segment();
                    Segment s2 = l2.segment();

                    // Apply same matching metrics
                    float angle = segments_angle(s1, s2);
                    float tdist = segments_distance(s1, s2);
                    float tproj = worst_point_segment_horizontal_proj(s1, s2)
                                    + worst_point_segment_horizontal_proj(s2, s1);

                    if (tproj < 0.5 && angle < MAX_ANGLE_PARA && tdist > MIN_DIST_PARA && tdist < MAX_DIST_PARA) {
                        // If there is a match, record the distance
                        distSum += tdist;
                        distCnt++;
                    }
                }
            }

            // The distance in pixels of how wide this street will be
            // for this curb line.
            int dist = 0;

            if (distCnt > 0) {
                // If we found a matching curb line, then use the average distance to its matches
                // times the proportion above
                dist = distSum / distCnt * match_fill_proportion;
            } else {
                // If there is no match, determine how far to fill
                dist = ll.curbLineP ? no_match_dist/2 : no_match_dist; // Case of no match
            }

            // Fill in the values in our images
            // Iterate over the curb line in the direction specified
            // using the distance specified.
            for (int lx = -width_extension; lx <= l.width_ + width_extension; lx += gap) {
                for (int ly = 0; ly <= dist; ly += gap) {

                    // Transform the line coords into absolute coords on the image
                    cv::Point linec{lx, ly*dir};
                    cv::Point absc = l.absolute_tranform(linec);
                    // Also, convert to ownership coords
                    cv::Point ownP = point_scale(absc, OWNERSHIP_SCALE_FACTOR / STREET_SCALE_FACTOR);

                    if (pixelInBounds(absc, imStreet) && ly >= fill_start)
                        imStreet.at<pixel_t>(absc) = 255; // fill in the street

                    if (pixelInBounds(ownP, imOrientX)) {
                        // Previous curb distance
                        float oldCurbDist = curbDist.at<float>(ownP);
                        // New curb distance
                        float newCurbDist = (ly + 2*std::max<float>({0.0f, lx-l.width_, 0.0f-lx}))
                                                / STREET_SCALE_FACTOR + 0.01f;
                        if (newCurbDist < oldCurbDist || oldCurbDist == 0) {
                            // Fill in the curb distance and curb orientation matrices,
                            // but don't overwrite a previously written value unless we are
                            // closer to the edge
                            curbDist.at<float>(ownP) = newCurbDist;
                            imOrientX.at<float>(ownP) = l.domi_.x;
                            imOrientY.at<float>(ownP) = l.domi_.y;
                        }
                    }
                }
            }

        }
    }

    // Blur a little to fill in gaps
    cv::blur(imStreet, imStreet, cv::Size(final_blur,final_blur));
    cv::threshold(imStreet, imStreet, 1, 255, CV_8UC1);

}








