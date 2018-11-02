#include <cassert>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "image_proc.h"

// Deep copy a graph
void copyGraph(std::vector<MyLine>& lines_from, std::vector<MyVertex>& verts_from,
               std::vector<MyLine>& lines_to, std::vector<MyVertex>& verts_to ) {

    for (MyLine& m: lines_from) {
        lines_to.push_back(m.clone());
    }
    for (MyVertex& m: verts_from) {
        verts_to.push_back(m.clone());
    }

}

// Scale an entire graph by some factor
void scaleGraph(std::vector<MyLine>& lines, std::vector<MyVertex>& verts, float scale) {
    for (MyLine& m: lines) {
        m.scale(scale);
    }
    for (MyVertex& m: verts) {
        m.scale(scale);
    }
}

// Generate a long line for each line
void generate_long_lines(std::vector<MyLine>& lines, std::vector<LongLine>& lls) {

    for (MyLine& m: lines) {
        lls.push_back(LongLine{m});
    }

}

// Create a single vertex for each line, so we can use it for matching
void generate_dummy_vertex(std::vector<MyLine>& lines, std::vector<MyVertex>& verts) {
    for (size_t i = 0; i < lines.size(); ++i) {
        MyLine& l = lines[i];
        MyVertex v(l.mid_);
        v.vert_id = i;
        l.vert1_ = i;
        l.vert2_ = i;
        v.adj_.push_back(i);
        v.con_.push_back(1);
        verts.push_back(v);
    }
}

// Match two line segments that share a single vertex
float match_segments_score(LongLine& l1, MyLine& m1, LongLine& l2, MyLine& m2, MyVertex& v, int matching_algorithm) {
// For each algorithm, low score = better match.
// No match is greater that 1e7

    if (matching_algorithm == MATCH_ALL) { // match everything
        return 0;
    }

    if (matching_algorithm == MATCH_DOTPRODUCT) { // match based on dot product
        float dotP = m1.domi_.x * m2.domi_.x + m1.domi_.y * m2.domi_.y;
        return 1 - std::abs<float>(dotP);
    }

    if (matching_algorithm == MATCH_LENGTH) { // match any long lines
        return 0 - m1.width_ - m2.width_;
    }

    if (matching_algorithm == MATCH_VECLINES) {
        // Matching for turning vectorized line segments into long lines

        // Get endpoint vectors
        cv::Point p1 = m1.endpointVector(v.loc_);
        cv::Point p2 = m2.endpointVector(v.loc_);
        cv::Point pl1 = l1.endpointVector(v.loc_);
        cv::Point pl2 = l2.endpointVector(v.loc_);

        // Want dot product to be near -1, since that means they are
        // parallel and point in the opposite direction. Dot product is a proxy for
        // angle between vectors
        float dotP = (p1.x * p2.x + p1.y * p2.y) / (cv::norm(p1) * cv::norm(p2));
        float dotLP = (pl1.x * pl2.x + pl1.y * pl2.y) / (cv::norm(pl1) * cv::norm(pl2));

        // The lines and long lines are required to be within 30 degrees of
        // each other, and point in the opposite direction.
        float score = (dotP > -0.80 ? 1e10 : dotP) + (dotLP > -COSINE30 ? 1e10: dotLP * 0.5);
        return (score > -1.1 ? 1e10 : score);
    }

    if (matching_algorithm == MATCH_VECCHARS) {
        // Matching for turning vectorized line segments into long lines for a char

        // Get endpoint vectors
        cv::Point p1 = m1.endpointVector(v.loc_);
        cv::Point p2 = m2.endpointVector(v.loc_);
        cv::Point pl1 = l1.endpointVector(v.loc_);
        cv::Point pl2 = l2.endpointVector(v.loc_);

        // Want dot product to be near -1, since that means they are
        // parallel and point in the opposite direction. Dot product is a proxy for
        // angle between vectors
        float dotP = (p1.x * p2.x + p1.y * p2.y) / (cv::norm(p1) * cv::norm(p2));
        float dotLP = (pl1.x * pl2.x + pl1.y * pl2.y) / (cv::norm(pl1) * cv::norm(pl2));

        // The lines and long lines are required to be within 30 degrees of
        // each other, and point in the opposite direction.
        float score = (dotP > -0.73 ? 1e10 : dotP) + (dotLP > -0.8 ? 1e10: dotLP * 0.5);
        return (score > -1.1 ? 1e10 : score);
    }

    if (matching_algorithm == MATCH_EXTENDED_STREETS) {
        // Used to combine historical street segments into extended historical streets

        // Obtain endpoint vectors
        cv::Point p1 = m1.endpointVector(v.loc_);
        cv::Point p2 = m2.endpointVector(v.loc_);
        cv::Point pl1 = l1.endpointVector(v.loc_);
        cv::Point pl2 = l2.endpointVector(v.loc_);

        // Dot product is a proxy for angle between
        float dotP = (p1.x * p2.x + p1.y * p2.y) / (cv::norm(p1) * cv::norm(p2));
        float dotLP = (pl1.x * pl2.x + pl1.y * pl2.y) / (cv::norm(pl1) * cv::norm(pl2));

        // The lines and long lines are required to be within 30 degrees of
        // each other, and point in the opposite direction.
        float score = (dotP > -COSINE30 ? 1e10 : dotP) + (dotLP > -COSINE30 ? 1e10: dotLP * 0.5);
        return (score > -1.1 ? 1e10 : score);
    }

    return 1e10;
}

// Match two line segments that do not share a vertex.
// Use the algorithm provided
float match_segments_gap_score(LongLine& l1, MyLine& m1, LongLine& l2, MyLine& m2,
                               MyVertex& v1, MyVertex& v2, int matching_algorithm) {
// For each algorithm, low score = better match.
// Any number greater than 1e7 means no match

    if (matching_algorithm == MATCH_ALL) { //match everything
        return 0;
    }

    if (matching_algorithm == MATCH_DOTPRODUCT) { // match based on dot product
        float dotP = m1.domi_.x * m2.domi_.x + m1.domi_.y * m2.domi_.y;
        return 1 - std::abs<float>(dotP);
    }

    if (matching_algorithm == MATCH_LENGTH) { // match any long lines

        return 0 - m1.width_ - m2.width_;
    }

    if (matching_algorithm == MATCH_DISTANCE) { // match anything close
        return cv::norm(v1.loc_ - v2.loc_);
    }

    if (matching_algorithm == MATCH_VECLINES || matching_algorithm == MATCH_VECCHARS) {
        // Matching function for turning the vectorized lines into long lines

        // Obtain vectors for the lines and their corresponding long lines
        cv::Point p1 = m1.endpointVector(v1.loc_);
        cv::Point p2 = m2.endpointVector(v2.loc_);
        cv::Point pl1 = l1.endpointVector(v1.loc_);
        cv::Point pl2 = l2.endpointVector(v2.loc_);
        cv::Point pDiff = v2.loc_ - v1.loc_;

        // Compute dot products, which are a proxy for the signed angle between
        // the vectors
        float gap = cv::norm(pDiff);
        float dotP = (p1.x * p2.x + p1.y * p2.y) / (cv::norm(p1) * cv::norm(p2));
        float dotLP = (pl1.x * pl2.x + pl1.y * pl2.y) / (cv::norm(pl1) * cv::norm(pl2));
        float dotDiff1 = ( pDiff.x * p1.x + pDiff.y * p1.y) / (cv::norm(p1) * cv::norm(pDiff));
        float dotDiff2 = (-pDiff.x * p2.x - pDiff.y * p2.y) / (cv::norm(p2) * cv::norm(pDiff));

        // Require an angle of 15 degrees or less for a jump match
        // That is, both the lines and the long lines must be aligned correctly, near 180 degree angle
        // The distance between the vertices must be less than 100 pixels
        const float max_jump_gap = 100 * STREET_SCALE_FACTOR;
        float score = gap + (dotP > -0.95 ? 1e10: dotP * 1.0) + (dotLP > -COSINE30 ? 1e10: dotLP * 0.5)
                          + (dotDiff1 > -0.95 ? 1e10: 0) + (dotDiff2 > -0.95 ? 1e10: 0);
        return (score > max_jump_gap ? 1e10 : score);
    }

    if (matching_algorithm == MATCH_EXTENDED_STREETS) {
        // Matching function for converting historical street segments into extended
        // historical streets.

        // Vectors for the line and long lines
        cv::Point p1 = m1.endpointVector(v1.loc_);
        cv::Point p2 = m2.endpointVector(v2.loc_);
        cv::Point pl1 = l1.endpointVector(v1.loc_);
        cv::Point pl2 = l2.endpointVector(v2.loc_);
        cv::Point pDiff = v2.loc_ - v1.loc_;

        // Dot products are a proxy for the angle between the vectors
        float gap = cv::norm(pDiff);
        float dotP = (p1.x * p2.x + p1.y * p2.y) / (cv::norm(p1) * cv::norm(p2));
        float dotLP = (pl1.x * pl2.x + pl1.y * pl2.y) / (cv::norm(pl1) * cv::norm(pl2));
        float dotDiff1 = ( pDiff.x * p1.x + pDiff.y * p1.y) / (cv::norm(p1) * cv::norm(pDiff));
        float dotDiff2 = (-pDiff.x * p2.x - pDiff.y * p2.y) / (cv::norm(p2) * cv::norm(pDiff));

        // Require an angle of 15 degrees or less for a jump match
        // The distance between must be less than 250 pixels
        const float max_jump_gap = 250;
        float score = gap + (dotP > -0.95 ? 1e10: dotP * 1.0) + (dotLP > -COSINE30 ? 1e10: dotLP * 0.5)
                          + (dotDiff1 > 0.5 ? 1e10: 0) + (dotDiff2 > 0.5 ? 1e10: 0);
        return (score > max_jump_gap ? 1e10 : score);
    }

    if (matching_algorithm == MATCH_CHARS) {

        // Parameters for character matching
        const int    CHARS_TOO_FAR_APART  = 250;
        const int    CHAR_AREA_TOO_SMALL  = 30;
        const double PARA_WIDTH_RATIO     = 0.65;
        const double MIDDLE_WIDTH_SPACING = 1.10;  // Maximal spacing between chars
        const double MIDDLE_HEIGHT_CUTOFF = 0.35;  // Maximal variation in height
        const double EDGE_WIDTH_SPACING   = 0.77;
        const double EDGE_HEIGHT_CUTOFF   = 0.39;

            double dist = cv::norm( m1.mid_ - m2.mid_ );
            if (dist > CHARS_TOO_FAR_APART) return 1e10;

            // Compute various shape metrics
            int area1 = m1.pixelCnt_;
            int area2 = m2.pixelCnt_;
            bool small2 = std::min<int>(area1, area2) < CHAR_AREA_TOO_SMALL;
            float maxHt = std::max<int>(m1.height_, m2.height_);
            float maxWd = std::max<int>(m1.width_, m2.width_);
            if (area1 < CHAR_AREA_TOO_SMALL || area2 < CHAR_AREA_TOO_SMALL) return 1e10;

            // Are these chars in the middle of the street?
            bool middle = m1.charDistFromCurb_ + m2.charDistFromCurb_ > m1.charDistFromMiddle_ + m2.charDistFromMiddle_;

            // Are either of the characters dashes?
            bool dash2 = m2.height_ * 2.5 < m2.width_ || m1.height_ * 2.5 < m1.width_;

            // Do we blur parallel to the street or perpendicular?
            // We blur perpendicular only if we see high aspect ratio, due to
            // sideways characters
            bool l1Para = m1.height_ > PARA_WIDTH_RATIO * m1.width_;
            bool l2Para = m2.height_ > PARA_WIDTH_RATIO * m2.width_;
            bool blurPara = l1Para || l2Para;
            bool blurPerp = !l1Para && !l2Para && !middle;

            // What is the distances between the centroids?
            cv::Point2f diff = m1.mid_ - m2.mid_;
            double paraDist = 0, perpDist = 0;
            if (area1 > area2) { // Project onto larger one
                paraDist = std::abs(diff.ddot(m1.domi_));
                perpDist = std::abs(diff.ddot(m1.norm_));
            } else {
                paraDist = std::abs(diff.ddot(m2.domi_));
                perpDist = std::abs(diff.ddot(m2.norm_));
            }

            // Use different thresholds for street names and house numbers
            double htCutoff = middle ? MIDDLE_HEIGHT_CUTOFF : EDGE_HEIGHT_CUTOFF;
            double spacing  = middle ? MIDDLE_WIDTH_SPACING : EDGE_WIDTH_SPACING;

            // Do we have a match?
            bool paraMatch = blurPara && perpDist < maxHt * (small2 ? 0.5: htCutoff)
                            && paraDist < m1.width_ * spacing + m2.width_ * spacing
                            && !(small2 && paraDist > maxWd * 0.5)
                            && !(middle && dash2 && paraDist > maxWd * 0.3);

            bool perpMatch = blurPerp && paraDist < maxWd * htCutoff
                            && perpDist < m1.height_ * spacing + m2.height_ * spacing;

            bool charMatch = paraMatch || perpMatch;
            return charMatch ? 0 : 1e10;
    }

    return 1e10;
}

// Match lines that share a vertex
void match_at_vertex(int vert_id,
                     std::vector<MyLine>& lines, std::vector<MyVertex>& verts, std::vector<LongLine>& lls,
                     int matching_algorithm, float thresh) {

    MyVertex& v = verts[vert_id];
    int n = v.adj_.size();

    if (n < 2) return; // not enough for a match

    std::vector<std::vector<float>> scores(n, std::vector<float>(n, 0.0));

    // Compute the match score for each pair of lines adjacent to this vertex
    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            MyLine& l1 = lines[v.adj_[i]];
            MyLine& l2 = lines[v.adj_[j]];
            float score = match_segments_score(lls[l1.long_line], l1, lls[l2.long_line], l2,
                                               v, matching_algorithm);
            scores[i][j] = score;
        }
    }

    while (1) {
        float bestMatchScore = thresh;
        int bestI = -1;
        int bestJ = -1;

        // Check each pair of lines for the best score
        for (int i = 0; i < n; ++i) {
            MyLine& l1 = lines[v.adj_[i]];

            // Check if the first line is already matched, ignore if so
            int mt1 = 0, mt2 = 0;
            if (v.con_[i] == 1) mt1 = l1.matched1;
            else mt1 = l1.matched2;
            if (mt1 > 0) continue;

            for (int j = i+1; j < n; ++j) {

                MyLine& l2 = lines[v.adj_[j]];

                // Check if the second line is already matched, ignore if so
                if (v.con_[j] == 1) mt2 = l2.matched1;
                else mt2 = l2.matched2;
                if (mt2 > 0) continue;

                if (scores[i][j] < bestMatchScore) {
                    bestI = i;
                    bestJ = j;
                    bestMatchScore = scores[i][j];
                }
            }
        }

        if (bestI == -1) return; // no good enough match, exit function

        // Match the two best lines
        int idx1 = lines[v.adj_[bestI]].long_line;
        int idx2 = lines[v.adj_[bestJ]].long_line;
        lls[idx1].append(lls[idx2], lines);
        // Remember that these lines now have matches
        if (v.con_[bestI] == 1) lines[v.adj_[bestI]].matched1++;
        else  lines[v.adj_[bestI]].matched2++;
        if (v.con_[bestJ] == 1) lines[v.adj_[bestJ]].matched1++;
        else  lines[v.adj_[bestJ]].matched2++;
    }

}

// This is for matching lines that do not share a vertex
void match_between_vertex(int vert_id1, int vert_id2,
                          std::vector<MyLine>& lines, std::vector<MyVertex>& verts, std::vector<LongLine>& lls,
                          int matching_algorithm, float thresh) {

    MyVertex& v1 = verts[vert_id1];
    MyVertex& v2 = verts[vert_id2];

    int n = v1.adj_.size();
    int m = v2.adj_.size();

    std::vector<std::vector<float>> scores(n, std::vector<float>(m, 0.0));

    // Loop over all pairs of lines adjacent to the two vertices
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {

            MyLine& l1 = lines[v1.adj_[i]];
            MyLine& l2 = lines[v2.adj_[j]];

            // check if the line is already matched at this vertex
            int mt1 = 0, mt2 = 0;
            if (v1.con_[i] == 1) mt1 = l1.matched1;
            else mt1 = l1.matched2;
            if (v2.con_[j] == 1) mt2 = l2.matched1;
            else mt2 = l2.matched2;

            // Compute the matching score
            float score = match_segments_gap_score(lls[l1.long_line], l1, lls[l2.long_line], l2,
                                               v1, v2, matching_algorithm);

            if (mt1 + mt2 > 0 && matching_algorithm != MATCH_CHARS)
                score = 1e10; // Prevent lines from matching twice on the same endpoint
                              // (Except for characters into words, which is allowed to)

            if (l1.line_id == l2.line_id || l1.long_line == l2.long_line)
                score = 1e10; // Prevent redundant matches

            scores[i][j] = score;
        }
    }

    float bestMatchScore = thresh;
    int bestI = -1;
    int bestJ = -1;

    // Find the best match that is better than the threshold
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (scores[i][j] < bestMatchScore) {
                bestI = i;
                bestJ = j;
                bestMatchScore = scores[i][j];
            }
        }
    }

    if (bestI == -1) return; // no match

    // Combine the two long lines that made the match
    int idx1 = lines[v1.adj_[bestI]].long_line;
    int idx2 = lines[v2.adj_[bestJ]].long_line;
    if (idx1 != idx2) {
        lls[idx1].append(lls[idx2], lines);
        lls[idx1].gap_jump_count++;
    }
    // Remember that these lines have made matches
    if (v1.con_[bestI] == 1) lines[v1.adj_[bestI]].matched1++;
    else  lines[v1.adj_[bestI]].matched2++;
    if (v2.con_[bestJ] == 1) lines[v2.adj_[bestJ]].matched1++;
    else  lines[v2.adj_[bestJ]].matched2++;

    // Since there was one match, try to match again
    match_between_vertex(vert_id1, vert_id2, lines, verts, lls, matching_algorithm, thresh);

}

// This function takes as input a list of lines, and groups them into long lines
// based on the matching function that is passed in.
// There are two types of matching.
// -Vertex matching: matching lines that share a vertex
// -Distance matching: matching lines that do not share a vertex
void combine_long_lines(std::vector<MyLine>& lines, std::vector<MyVertex>& verts, std::vector<LongLine>& lls, cv::Mat& img,
                        int vert_matching_iter, int distance_matching_iter, int matching_algorithm, float thresh) {

    for (int iter = 0; iter < vert_matching_iter; ++iter) {
        // Loop over every vertex, and run the matching function
        for (MyVertex& v: verts) {
            match_at_vertex(v.vert_id, lines, verts, lls, matching_algorithm, thresh);
        }
    }

    for (int iter = 0; iter < distance_matching_iter; ++iter) {

        // We can't simply use a pairwise comparison of each pair of lines,
        // because it is too slow.

        // Use binning instead.
        int binSize = 100;

        switch(matching_algorithm) {
        case MATCH_VECLINES:
            binSize = 25;
            break;
        case MATCH_CHARS:
            binSize = 300;
            break;
        case MATCH_EXTENDED_STREETS:
            binSize = 1000;
            break;
        default:
            binSize = 100;
            break;
        }

        // Bins to hold the vertexes
        int xBins = img.cols / binSize + 2;
        int yBins = img.rows / binSize + 2;
        std::vector<std::vector<std::vector<int>>> bins(yBins, std::vector<std::vector<int>>(xBins, std::vector<int>(0)));

        // Put vertexes in bins
        for (MyVertex& v: verts) {
            int xb = v.loc_.x / binSize;
            int yb = v.loc_.y / binSize;
            if (xb >= 0 && yb >= 0 && xb < xBins && yb < yBins)
                (bins[yb][xb]).push_back(v.vert_id);
        }

        // Call the helper function for each pair of vertexes that are close
        // in distance. They are close if they are in adjacent bins.
        for (int yb = 0; yb < yBins - 1; ++yb) {
            for (int xb = 0; xb < xBins - 1; ++xb) {

                std::vector<int>& t00 = bins[yb][xb];
                std::vector<int>& t01 = bins[yb][xb+1];
                std::vector<int>& t10 = bins[yb+1][xb];
                std::vector<int>& t11 = bins[yb+1][xb+1];

                for (size_t i = 0; i < t00.size(); ++i) {
                    for (size_t j = i+1; j < t00.size(); ++j) {
                        match_between_vertex(t00[i], t00[j], lines, verts, lls, matching_algorithm, thresh);
                    }
                }
                for (size_t i = 0; i < t00.size(); ++i) {
                    for (size_t j = 0; j < t01.size(); ++j) {
                        match_between_vertex(t00[i], t01[j], lines, verts, lls, matching_algorithm, thresh);
                    }
                }
                for (size_t i = 0; i < t00.size(); ++i) {
                    for (size_t j = 0; j < t10.size(); ++j) {
                        match_between_vertex(t00[i], t10[j], lines, verts, lls, matching_algorithm, thresh);
                    }
                }
                for (size_t i = 0; i < t00.size(); ++i) {
                    for (size_t j = 0; j < t11.size(); ++j) {
                        match_between_vertex(t00[i], t11[j], lines, verts, lls, matching_algorithm, thresh);
                    }
                }

            }
        }
    }
}

// ------------------------ Line segmenting -------------------------

// Take a single branch of the shrink and break it up into
// further segments, if necessary.
void segment_lines_helper(int line_id, std::vector< cv::Point >& points, int ptSt, int ptEnd,
                          std::vector< MyLine >& lines, std::vector< MyVertex >& verts,
                          std::vector<std::vector<cv::Point>>& new_chains,
                          int HEIGHT_THRESHOLD, int TOO_SHORT_THRESHOLD) {

    // Find the best fit line for the point cloud "points"
    MyLine& line = lines[line_id];
    line.calculateDimensions(points, ptSt, ptEnd);

    // Too short. Don't segment any further
    bool tooSmall = ptEnd - ptSt < TOO_SHORT_THRESHOLD || ptEnd - ptSt < 4;

    // Bad line fit. Segment further.
    if (line.height_ > HEIGHT_THRESHOLD && !tooSmall) {

        // Index to partition at
        size_t partition_index = line.max_h > -line.min_h ? line.max_hi : line.min_hi;

        // Point to split at
        cv::Point mid = points[partition_index];
        //if (mid.x == 0 && mid.y == 0)
         //   std::cout << "PROBLEM!!" << std::endl;

        // Create two new lines
        int old_line_id = line.line_id;
        int new_line_id = lines.size();
        lines.push_back( MyLine{mid, line.endpt2_, false, false} );
        MyLine& line1 = lines[old_line_id];
        MyLine& line2 = lines[new_line_id];
        line2.line_id = new_line_id;
        line1.reformLine(line1.endpt1_, mid, false);

        // Remove the branch point from the first line,
        // add it to the second line
        line2.vert2_ = line1.vert2_;
        int old_vertex_pointer = line1.vert2_;
        for (int& adj_line_id: verts[old_vertex_pointer].adj_) {
            if (adj_line_id == old_line_id)
                adj_line_id = new_line_id;
        }

        // Add a corner point between the new lines
        int new_vert_id = verts.size();
        verts.push_back( MyVertex{mid} );
        verts[new_vert_id].vert_id = new_vert_id;
        verts[new_vert_id].adj_.push_back(old_line_id);
        verts[new_vert_id].con_.push_back(2);
        verts[new_vert_id].adj_.push_back(new_line_id);
        verts[new_vert_id].con_.push_back(1);
        line1.vert2_ = new_vert_id;
        line2.vert1_ = new_vert_id;

        // Recurse
        segment_lines_helper(old_line_id, points, ptSt, partition_index+1,
                             lines, verts, new_chains, HEIGHT_THRESHOLD, TOO_SHORT_THRESHOLD);
        segment_lines_helper(new_line_id, points, partition_index,  ptEnd,
                             lines, verts, new_chains, HEIGHT_THRESHOLD, TOO_SHORT_THRESHOLD);
    }

    else {
        // We are done with this line...
        // note, this happens in order of line id... (how convenient)
        // add a copy of this pixel chain to our output list.
        std::vector<cv::Point> chainCopy;
        for (int i = ptSt; i < ptEnd && i < (int)points.size(); ++i) {
            cv::Point a {points[i].x, points[i].y};
            chainCopy.push_back(a);
        }
        new_chains.push_back(chainCopy);

    }

}


// This function takes as input pixel chains, and outputs a graph
// representation of the pixel chains as lines and vertices.
// height threshold says when to create a corner point
// length threshold says when to stop segmenting
// chain start says how many pixels past the branch point to ignore (deals with "Curl")
void segment_lines(std::vector<std::vector<cv::Point>>& chains, std::vector<std::vector<cv::Point>>& new_chains,
                   std::vector<MyLine>& lines, std::vector<MyVertex>& verts,
                   int height_threshold, int length_threshold, int chain_start)
{
    size_t vert_id = 0;
    size_t branchpt_cnt = 0;
    new_chains = std::vector<std::vector<cv::Point>>();

    // Here, we will detect all of the unique branch points
    // and end points, creating a vertex for each one.
    for (size_t i = 0; i < chains.size(); ++i) {
        for (size_t j = 0; j < chains[i].size(); j += chains[i].size() - 1) {

            // Endpoints of this chain
            cv::Point chain_branchpt = chains[i][j];

            // Check if we have already seen the endpoint
            bool repeat = false;
            for (MyVertex v: verts) {
                if (v.loc_ == chain_branchpt) {
                    repeat = true;
                    break;
                }
            }

            // found a new branch point
            // Add a new vertex for it
            if (!repeat) {
                verts.push_back( MyVertex{chain_branchpt} );
                verts[vert_id].vert_id = vert_id;
                vert_id++;
                branchpt_cnt++;
            }

        }
    }

    // Segment the lines with the helper function
    for (size_t i = 0; i < chains.size(); ++i) {

        // Endpoints of this chain
        cv::Point chain_endpt1 = chains[i][0];
        cv::Point chain_endpt2 = chains[i][chains[i].size() - 1];

        // Endpoints of this line
        // Due to the "curling" effect of the branch points, we will
        // start the segments slightly later in their chain.
        int chain_start1 = std::min<int>(((int)(chains[i].size()))/2 - chain_start, chain_start);
        chain_start1 = std::max<int>(0, chain_start1);
        int chain_start2 = chains[i].size() - 1 - chain_start1;
        cv::Point line_endpt1 = chains[i][chain_start1];
        cv::Point line_endpt2 = chains[i][chain_start2];

        // Create a new line with these endpoints
        int line_id = lines.size();
        lines.push_back( MyLine{line_endpt1, line_endpt2, false, false} );
        MyLine& line = lines[line_id];
        line.line_id = line_id;

        // find vertex representation of the endpoints
        for (size_t j = 0; j < branchpt_cnt; ++j) {
            MyVertex& v = verts[j];
            if (v.loc_ == chain_endpt1) {
                line.vert1_ = j;
                v.adj_.push_back(line.line_id);
                v.con_.push_back(1);
            }
            if (v.loc_ == chain_endpt2) {
                line.vert2_ = j;
                v.adj_.push_back(line.line_id);
                v.con_.push_back(2);
            }
        }

        //std::pair<cv::Point, cv::Point> endpts = line.endpoints();
        //if ( (endpts.first.x < 10 && endpts.first.y < 10) ||  (endpts.second.x < 10 && endpts.second.y < 10)) {
        //    std::cout << endpts.first << " " << endpts.second << " " << line_id << std::endl;
         //   for (cv::Point pt: chains[i]) {
         //       std::cout << pt << ", ";
         //   }
         //   std::cout << std::endl;
        //}

        // recursively detect corner points
        segment_lines_helper(line.line_id, chains[i], chain_start1, chain_start2+1, lines, verts, new_chains, height_threshold, length_threshold);

    }

}
