#include "image_proc.h"
#include "geometry.h"

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <cmath>

// Returns midpoint of a segment
cv::Point segment_midpoint(Segment s1) {
    cv::Point out = s1.p1 + s1.p2;
    out /= 2;
    return out;
}

// convert radians to degrees
double degrees(double radians)
{
    double degrees = radians*180/PI;
    if (degrees >= 90)
        return 180 - degrees;
    return degrees;
}

double segment_length(double x1, double y1, double x2, double y2)
{
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}
double segment_length(Segment s) {
    return cv::norm(s.p1 - s.p2);
}

double segments_angle(double x11, double y11, double x12, double y12, double x21, double y21, double x22, double y22)
{
    // turn the segments into vectors
    double xA = x11 - x12;
    double yA = y11 - y12;
    double xB = x21 - x22;
    double yB = y21 - y22;

    double dot_product = xA * xB + yA * yB;
    double lengthA = segment_length(x11, y11, x12, y12);
    double lengthB = segment_length(x21, y21, x22, y22);

    // calculate angle as acos(A * B / (||A|| || B ||))
    double arg = dot_product / (lengthA * lengthB);

    if (arg > 1) { // can happen due to floating point errors
        return 0;
    } else {
        return degrees(acos(arg));
    }
}
// Return the angle in degrees between two segments
double segments_angle(Segment s1, Segment s2) {
    return segments_angle(s1.p1.x, s1.p1.y, s1.p2.x, s1.p2.y, s2.p1.x, s2.p1.y, s2.p2.x, s2.p2.y);
}

// whether two segments in the plane intersect:
// one segment is (x11, y11) to (x12, y12)
// the other is   (x21, y21) to (x22, y22)
bool segments_intersect(double x11, double y11, double x12, double y12, double x21, double y21, double x22, double y22)
{
    double dx1 = x12 - x11;
    double dy1 = y12 - y11;
    double dx2 = x22 - x21;
    double dy2 = y22 - y21;
    double delta = dx2 * dy1 - dy2 * dx1;
    if (delta < 0.0001) {
        return false;
    }
    double s = (dx1 * (y21 - y11) + dy1 * (x11 - x21)) / delta;
    double t = (dx2 * (y11 - y21) + dy2 * (x21 - x11)) / (-delta);
    return (0 <= s) && (s <= 1) && (0 <= t) && (t <= 1);
}

bool segments_intersect(Segment s1, Segment s2) {
    return segments_intersect(s1.p1.x, s1.p1.y, s1.p2.x, s1.p2.y, s2.p1.x, s2.p1.y, s2.p2.x, s2.p2.y);
}


// distance from one point to a line segment
double point_segment_distance(double px, double py, double x1, double y1, double x2, double y2)
{
    double dx = x2 - x1;
    double dy = y2 - y1;
    if (dx == 0 && dy == 0) {  // the segment's just a point
        return segment_length(px, py, x1, y1);
    }

    // Calculate the t that minimizes the distance.
    double t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);

    // See if this represents one of the segment's
    // end points or a point in the middle.
    if (t < 0) {
        dx = px - x1;
        dy = py - y1;
    }
    else if (t > 1) {
        dx = px - x2;
        dy = py - y2;
    }
    else {
        double near_x = x1 + t * dx;
        double near_y = y1 + t * dy;
        dx = px - near_x;
        dy = py - near_y;
    }

    return segment_length(0, 0, dx, dy);
}
double point_segment_distance(cv::Point2f p, Segment s) {
    return point_segment_distance(p.x, p.y, s.p1.x, s.p1.y, s.p2.x, s.p2.y);
}

// distance between two segments in the plane:
// one segment is (x11, y11) to (x12, y12)
// the other is   (x21, y21) to (x22, y22)
double segments_distance(double x11, double y11, double x12, double y12, double x21, double y21, double x22, double y22)
{
    if (segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22)) {
        return 0;
    }
    // try each of the 4 vertices w/the other segment
    double d1 = point_segment_distance(x11, y11, x21, y21, x22, y22);
    double d2 = point_segment_distance(x12, y12, x21, y21, x22, y22);
    double d3 = point_segment_distance(x21, y21, x11, y11, x12, y12);
    double d4 = point_segment_distance(x22, y22, x11, y11, x12, y12);
    return std::min({d1, d2, d3, d4});
}
double segments_distance(Segment s1, Segment s2) {
    return segments_distance(s1.p1.x, s1.p1.y, s1.p2.x, s1.p2.y, s2.p1.x, s2.p1.y, s2.p2.x, s2.p2.y);
}

// distance between an extended line and a line segment:
// one segment is (x11, y11) to (x12, y12) (gets extended)
// the other is   (x21, y21) to (x22, y22)
double line_to_segment_distance(double x11, double y11, double x12, double y12, double x21, double y21, double x22, double y22)
{
    double xA = x11-x12;
    double yA = y11-y12;
    double mul = 3; // line extension multiplier
    return segments_distance(x11 + mul*xA, y11 + mul*yA, x12 - mul*xA, y12 - mul*yA,
                             x21, y21, x22, y22);
}
double line_to_segment_distance(Segment s1, Segment s2) {
    return line_to_segment_distance(s1.p1.x, s1.p1.y, s1.p2.x, s1.p2.y, s2.p1.x, s2.p1.y, s2.p2.x, s2.p2.y);
}

// Projects p onto the line segment s.
// Output is 0 if the projection lies on the first endpoint of s,
// Output is 1 if the projection lies on the second endpoint of s.
// Output is in (0,1) if the projection lies on s.
double point_segment_horizontal_proj(double px, double py, double x1, double y1, double x2, double y2)
{
    double dx = x2 - x1;
    double dy = y2 - y1;
    if (dx == 0 && dy == 0) { // # the segment's just a point
        return segment_length(px, py, x1, y1);
    }

    // Calculate the t that minimizes the distance.
    double t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);
    return t;
}
double point_segment_horizontal_proj(cv::Point2f p, Segment s) {
    return point_segment_horizontal_proj(p.x, p.y, s.p1.x, s.p1.y, s.p2.x, s.p2.y);
}

// Projects both endpoints of the second segment onto the first segment.
// If both endpoints lie outside of the original segment on the same side,
// A penalty is returned based on how far away the edges are.
// The purpose of this function is to help determine if two segments match.
// If the value is non zero, then the two segments do not match that well
double worst_point_segment_horizontal_proj(double x11, double y11, double x12, double y12, double x21, double y21, double x22, double y22)
{
    double cost_distance1 = 0;
    double t21 = point_segment_horizontal_proj(x21, y21, x11, y11, x12, y12);
    double t22 = point_segment_horizontal_proj(x22, y22, x11, y11, x12, y12);
    if (std::min(t21, t22) > 1) {
        cost_distance1 = (std::min(t21, t22) - 1);
    }
    if (std::max(t21, t22) < 0) {
        cost_distance1 = (0 - std::max(t21, t22));
    }
    return cost_distance1;
}
double worst_point_segment_horizontal_proj(Segment s1, Segment s2) {
    return worst_point_segment_horizontal_proj(s1.p1.x, s1.p1.y, s1.p2.x, s1.p2.y, s2.p1.x, s2.p1.y, s2.p2.x, s2.p2.y);
}

// Computes a total score for matching a modern and historical street
double street_matching_score(double x11, double y11, double x12, double y12, double x21, double y21, double x22, double y22)
{
    // angle between segments
    double angle = segments_angle(x11, y11, x12, y12, x21, y21, x22, y22);

    // closest distance between segments
    double dist = segments_distance(x11, y11, x12, y12, x21, y21, x22, y22);

    double len1 = segment_length(x11, y11, x12, y12);
    double len2 = segment_length(x21, y21, x22, y22);
    double frac1 = len1/(len1+len2); // fractions of total lengths
    double frac2 = len2/(len1+len2);

    // projection onto line 1
    double cost_distance1 = dist*2
        + 1000*len1*worst_point_segment_horizontal_proj(x11, y11, x12, y12, x21, y21, x22, y22);

    // projection onto line 2
    double cost_distance2 = dist*2
        + 1000*len1*worst_point_segment_horizontal_proj(x21, y21, x22, y22, x11, y11, x12, y12);

    // compute overall score. Lower is better
    return pow(angle, 2) + cost_distance1*frac1 + cost_distance2*frac2;
}

//
double extended_street_matching_score(double x11, double y11, double x12, double y12, double x21, double y21, double x22, double y22)
{
    // angle between segments
    double angle = segments_angle(x11, y11, x12, y12, x21, y21, x22, y22);

    double len1 = segment_length(x11, y11, x12, y12);
    double len2 = segment_length(x21, y21, x22, y22);
    double frac1 = len1/(len1+len2);
    double frac2 = len2/(len1+len2);

    double diste1 = line_to_segment_distance(x11, y11, x12, y12, x21, y21, x22, y22);
    double diste2 = line_to_segment_distance(x21, y21, x22, y22, x11, y11, x12, y12);

    return pow(angle, 2) + diste1*frac1 + diste2*frac2;
}

// Returns true if the two streets passed in match,
// relative to the matching function
bool streets_match(const MyLine& h_street, const Street& p_street)
{
    const uint matching_threshold = 1000;
    auto h_endpoints = h_street.endpoints();
    auto p_endpoints = p_street.segment.endpoints();

    // cost for a match just uses the street matching score function.
    double cost = street_matching_score(h_endpoints.first.x, h_endpoints.first.y,
                                 h_endpoints.second.x, h_endpoints.second.y,
                                 p_endpoints.first.x, p_endpoints.first.y,
                                 p_endpoints.second.x, p_endpoints.second.y);

    return cost < matching_threshold;
}

// Match all of the historical streets with the present streets
// The output is a map from historical streets to present streets that match
std::unordered_map<uint, std::vector<int>> match_streets(const std::vector<MyLine>& h_streets, const std::vector<Street>& p_streets)
{
    // a map from historical ids to present ids
    std::unordered_map<uint, std::vector<int>> h2p_matches;

    // Try to match each historical street with each present street.
    for (auto& h_street : h_streets) {
        std::vector<int> p_matches;
        for (auto& p_street: p_streets) {
            if (streets_match(h_street, p_street)) {
                p_matches.push_back(p_street.id);
            }
        }
        h2p_matches[h_street.streetId_] = p_matches;
    }

    return h2p_matches;
}

