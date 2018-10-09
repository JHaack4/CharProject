#pragma once
// geometry.h
// definitions for lines and streets
#include <vector>
#include "opencv2/opencv.hpp"


/**
 * @brief      A very simple line segment.
 * @todo       Integrate with MyLine class, or vice versa
 */
struct Segment {
    cv::Point p1;
    cv::Point p2;

    std::pair<cv::Point, cv::Point> endpoints() const {
        return {p1, p2};
    }

};

/**
 * @brief   Here, a street is defined by a single segment
 *          However, it is possible that a curved road is composed of many
 *          segments.  These segments will all be distinct Street objects, with
 *          distinct ids.  However, the feature_id will be common among them.
 * @todo    Perhaps change so that a Street can contain many segments, and
 *          corresponds to a single OGR feature
 *
 */
struct Street {
    uint id;    ///< the id unique to this segment
    uint feature_id; ///< the id of the present-day street
    Segment segment;
};

double segment_length(Segment s);
double segments_angle(Segment s1, Segment s2);
bool segments_intersect(Segment s1, Segment s2);
double point_segment_distance(cv::Point2f p, Segment s);
double segments_distance(Segment s1, Segment s2);
double line_to_segment_distance(Segment s1, Segment s2);
double point_segment_horizontal_proj(cv::Point2f p, Segment s);
double worst_point_segment_horizontal_proj(Segment s1, Segment s2);
cv::Point segment_midpoint(Segment s1);


