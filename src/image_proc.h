#pragma once

#include <vector>
#include <unordered_set>
#include <math.h>
#include "geometry.h"
#include "opencv2/opencv.hpp"

typedef unsigned char pixel_t;
typedef std::vector<float> spectrum_t;

// global variables
extern std::string MIDWAY_PATH;    // files needed for OCR and matching
extern std::string DEBUG_PATH;     // Path where we write debug images
extern std::string NAME;           // Name of the map (for file writing)
extern bool DRAW_DEBUG_IMAGES;
extern double RESOLUTION;          // resolution, in meters/pixel
extern bool OUTPUT_CROPPED_IMAGES;  // Should we output cropped digits and letters to a special folder?

#define PI 3.14159265
#define COSINE30 0.866025
#define STREET_SCALE_FACTOR 0.1     // How much to downscale for street finding
#define SHRINK_SCALE_FACTOR 0.5     // How much to downscale for shrink finding
#define OWNERSHIP_SCALE_FACTOR 0.04 // Size of ownership matrices
#define SEAM_CARVE_MODERN_STREETS 0 // Use the modern streets to help find historical streets?

// For file writing
#ifdef _WIN32
const char PATH_SEP = '\\';
#else
const char PATH_SEP = '/';
#endif

class MyLine;
class MyVertex;
class LongLine;


// image_proc.cpp
void process_image(cv::Mat& img, const std::vector<cv::Point>& boundary, const std::vector<Street>& streets);
void get_mask(cv::Mat& mask, const std::vector<cv::Point> & pts);
void binarize(cv::Mat& img, cv::Mat& out);
void remove_markings(cv::Mat& img, cv::Mat& lines, cv::Mat& mark,
                     std::vector<cv::Point>& marking_centroids, std::vector<int>& marking_sizes);
void floodfill_streets(cv::Mat& img, cv::Mat& mask, cv::Mat& streets);

// shrink.cpp
std::vector<std::vector<cv::Point>> shrink(cv::Mat& filled_streets, float pThr, float T);
void draw_chains(const std::vector<std::vector<cv::Point>> chains, cv::Mat& miniChains, cv::Mat& miniBranchPoints);

// vectorize.cpp
void segment_lines(std::vector<std::vector<cv::Point>>& chains, std::vector<std::vector<cv::Point>>& new_chains, std::vector<MyLine>& lines,
                   std::vector<MyVertex>& verts, int height_threshold, int length_threshold, int chain_start);
void copyGraph(std::vector<MyLine>& lines_from, std::vector<MyVertex>& verts_from,
               std::vector<MyLine>& lines_to, std::vector<MyVertex>& verts_to );
void scaleGraph(std::vector<MyLine>& lines, std::vector<MyVertex>& verts, float scale);
void combine_long_lines(std::vector<MyLine>& lines, std::vector<MyVertex>& verts, std::vector<LongLine>& lls, cv::Mat& img,
                        int vert_matching_iter, int distance_matching_iter, int matching_algorithm, float thresh=1e7);
void generate_long_lines(std::vector<MyLine>& lines, std::vector<LongLine>& lls);
void generate_dummy_vertex(std::vector<MyLine>& lines, std::vector<MyVertex>& verts);

// seam_carve.cpp
void seam_carve_street(MyLine street, cv::Mat& imLines, cv::Mat& smStreet, std::vector<MyLine>& streets,
                       cv::Mat& outStMidd, cv::Mat& outStEdge, cv::Mat& outStSeam, cv::Mat& stOwn, cv::Mat& midDist);
const int SEAM_CARVE_EXTENSION = 500;

// find_words.cpp
void form_words_all(cv::Mat& imMarkSt, cv::Mat& midDist, cv::Mat& curbDist, cv::Mat& streetOwnership, cv::Mat& orientX, cv::Mat& orientY,
                    std::vector<MyLine>& streets, std::vector<LongLine>& extendedStreets, std::vector<MyLine>& words,
                    std::vector<std::vector<cv::Point>>& wordPixels);

// serialize.cpp
void read_args(std::string args_string, std::string& map_path, std::vector<cv::Point>& boundary, std::vector<Street>& streets);
std::vector<cv::Point> read_boundary(const std::string& boundary_string);
std::vector<Street> read_streets(const std::string& streets_string);
void write_streets(const std::vector<MyLine>& streets);
void write_words(const std::vector<MyLine>& words);
void write_matches(const std::unordered_map<uint, std::vector<int>>& matches);
void write_extensions(const std::vector<LongLine>& extended_streets);
void dump_json();
void fill_holes(cv::Mat& img, int area_threshold);


// image_util.cpp
cv::Mat my_rotate(cv::Mat& img, MyLine& line);
void cv_rotate(cv::Mat& src, cv::Mat& dst, double angle);
cv::Mat my_deskew(cv::Mat& img, double skew);
bool pixelInBounds(cv::Point p, cv::Mat& m);
bool pixelInBounds(int r, int c, cv::Mat& m);
cv::Point point_scale(cv::Point2f p, double scale);
int searchForOwner(cv::Point2f p, cv::Mat& ownership);
float searchForOwnerF(cv::Point2f p, cv::Mat& ownership);
std::vector<std::vector<cv::Point>> connectedComponentsWithIdx(cv::Mat& img, cv::Mat& labels);
std::vector<std::vector<cv::Point>> connectedComponentsSkel(cv::Mat& skel, cv::Mat& branchEndPoints);
void pixel_chain_owners(cv::Mat& img, std::vector<std::vector<cv::Point>>& pixel_chains, std::vector<std::vector<cv::Point>>& pixels_owned, int maxIter);
double averageXPos(const std::vector<cv::Point>& pixels);
cv::Mat idxToImg(std::vector<cv::Point>& points, size_t pad);
cv::Mat idxRotate(std::vector<cv::Point>& points, MyLine& line);
MyLine fitPointCloud(std::vector<cv::Point>& points);
MyLine fitPointCloud(std::vector<cv::Point>& points, cv::Point2f domi, bool useBoxCenter);
void texturepack(cv::Mat& pack, cv::Mat& img, MyLine& word, int& lastR, int& lastC, int max_height, std::string fileName, int& fileCount);

// image_debug.cpp
void package_bgr(const std::vector<cv::Mat>& layers, cv::Mat& output);
void color_connected_components(const cv::Mat& input, cv::Mat& output, cv::Vec3b background = cv::Vec3b{0,0,0}, cv::Vec3b rangeMin = cv::Vec3b{50,50,50}, cv::Vec3b rangeMax = cv::Vec3b{255,255,255});
void color_labels(const cv::Mat& input, cv::Mat& output, cv::Vec3b background = cv::Vec3b{0,0,0}, cv::Vec3b rangeMin = cv::Vec3b{50,50,50}, cv::Vec3b rangeMax = cv::Vec3b{255,255,255});
void draw_pixel_chains(cv::Mat& imBin, std::vector< std::vector<cv::Point>>& pixelChains, cv::Mat& output);
cv::Vec3b id_color(int i);
void debug_imwrite(const cv::Mat& img, const std::string& title);
void specific_imwrite(const cv::Mat& img, const std::string& type, const std::string& info = "");
void midway_imwrite(const cv::Mat& img, const std::string& title);
void make_dir(const std::string& dir, const std::string& name);

// log.cpp
enum LogLevel{ debug, info, warn, error };
LogLevel get_log_level(std::string str);
void log(std::string message, LogLevel level);
extern LogLevel LOG_LEVEL;

// match_street.cpp
std::unordered_map<uint, std::vector<int>> match_streets(const std::vector<MyLine>& h_streets, const std::vector<Street>& p_streets);
std::vector<std::unordered_set<int>> extend_streets(const std::vector<MyLine>& h_streets);

// curb_lines.cpp
void curbMatchModern(std::vector<MyLine>& lines, std::vector<LongLine>& lls, std::vector<Street>& presentStreets);
void curbFloodFill(std::vector<MyLine>& lines, std::vector<LongLine>& lls, cv::Mat& imFloodFill, cv::Mat& imMask);
void curbBranchPoints(std::vector<MyLine>& lines, std::vector<LongLine>& lls, std::vector<MyVertex>& verts);
void curbMatchMarkings(std::vector<MyLine>& lines, std::vector<LongLine>& lls,
                       std::vector<cv::Point>& markingCentroids, std::vector<int>& markingSizes,
                       cv::Mat& smLines);
void curbMatchPara(std::vector<MyLine>& lines, std::vector<LongLine>& lls);
void curbAggregate(std::vector<MyLine>& lines, std::vector<LongLine>& lls);
void curbToStreets(std::vector<MyLine>& lines, std::vector<LongLine>& lls, cv::Mat& imStreet, cv::Mat& ownOrientX, cv::Mat& ownOrientY, cv::Mat& ownCurbDist);

// Unused code
void run_word_project();
void thinning(const cv::Mat& src, cv::Mat& dst, cv::Mat& branch_points, bool skel = true, int max_iter = 1000000);
void thinning2(const cv::Mat& src, cv::Mat& dst, cv::Mat& branch_points, bool skel = true, int max_iter = 1000000);
void thinning3(const cv::Mat& src, cv::Mat& dst, cv::Mat& branch_points, bool skel = true, int max_iter = 1000000);
void thinning4(const cv::Mat& src, cv::Mat& dst, cv::Mat& branch_points, bool skel = true, int max_iter = 1000000);
void thinning5(const cv::Mat& src, cv::Mat& dst, cv::Mat& branch_points, bool skel = true, int max_iter = 1000000);
void thinning6(const cv::Mat& src, cv::Mat& dst, cv::Mat& branch_points, bool skel = true, int max_iter = 1000000);
void thinning7(const cv::Mat& src, cv::Mat& dst, cv::Mat& branch_points, bool skel = true, int max_iter = 1000000);

bool checkIfConnectedComponentsWorks(cv::Mat& img, MyLine& word);


// gu chars
void look_at_char(std::string path_to_main_directory);
void char_to_spectrum(cv::Mat& imBin, std::vector<float>& spectrum, std::vector<float>& mask, bool useAverages);
float align_spectrum(std::vector<float>& spectrum1, std::vector<float>& spectrum2, int& outAngle);
float score_spectrum(std::vector<float>& spectrum1, int angle1, std::vector<float>& spectrum2, int angle2);
void wheel_to_spectrum(std::vector<cv::Point2f>& wheel, std::vector<float>& spectrum, std::vector<float>& mask);
void find_best_representative_spectrum(std::vector<spectrum_t>& representatives, spectrum_t spectrum, int& outRep, int& outAngle);
void find_best_representative_spectrum2(std::vector<std::vector<spectrum_t>>& representatives, spectrum_t spectrum, int& outRep, int& outAngle);



// Matching codes
#define MATCH_ALL 0
#define MATCH_DOTPRODUCT 1
#define MATCH_LENGTH 2
#define MATCH_DISTANCE 3
#define MATCH_VECLINES 4
#define MATCH_CHARS 5
#define MATCH_EXTENDED_STREETS 6
#define MATCH_VECCHARS 4

// This class is used as the vertices for a graph, where the
// edges are MyLine objects and the vertices are MyVertex objects.
// The vertex remembers a location and which lines it is connected to
class MyVertex {
public:
    MyVertex(cv::Point loc) {
        loc_ = loc;
    }

    cv::Point loc_;        // Location of vertex
    int vert_id = -1;      // ID for this vertex
    std::vector<int> adj_; // line_id's of adjacent lines
    std::vector<int> con_; // which endpoint of the line this vertex is connected to

    // deep copy
    MyVertex clone() {
        MyVertex m{loc_};
        m.vert_id = vert_id;
        for (int i: adj_) m.adj_.push_back(i);
        for (int i: con_) m.con_.push_back(i);
        return m;
    }

    void scale(float scale) {
        loc_ = loc_ * scale;
    }
};

// Line class
// We represent nearly everything using this MyLine object, such as lines, streets, chars, and words.
// The object tracks the center of the line (mid_), as well as the direction it points (domi_).
// A MyLine can be turned into a bounding box by feeding it a point cloud. Then, the line also
// has width and height.
// A MyLine can be part of a graph if it is generated using vectorization. Then, the line also
// can be used in a matching function to create LongLines (poly-lines) which are groups of lines.
class MyLine {
public:

    // The definition of the line
    cv::Point2f mid_; // Midpoint aka centroid
    cv::Point2f domi_, norm_; // Unit direction vectors

    // Line with endpoints
    cv::Point endpt1_, endpt2_; // endpoints

    // Line that is part of a graph
    int vert1_, vert2_; // vert_id's of adjacent vertices
    int line_id = -1;   // ID for this line
    int long_line = -1; // pointer to ID of long line this line is part of
    int matched1 = 0, matched2 = 0; // Tracks which endpoints are matched

    // Stores the width and height of the line, after they
    // are calculated with a point cloud. We also track the
    // extremal points of the point cloud.
    bool notFixedEndpoints; // Allow the endpoints to change?
    float width_ = -1, height_ = -1;
    float varW_ = -1, varH_ = -1;
    float max_h = -1, min_h = -1, max_w = -1, min_w = -1;
    int max_hi = -1, min_hi = -1, max_wi = -1, min_wi = -1, pixelCnt_ = -1;
    cv::Point boundingBoxCenter_; // The center of the bounding box. Sometimes, this is more useful than mid_

    // Create a line with a center and a direction.
    MyLine(const cv::Point2f mid, const cv::Point2f domi) {
        mid_ = mid;
        domi_ = domi;

        norm_ = cv::Point2f(-domi_.y, domi_.x);
        notFixedEndpoints = true;
    }

    // Create a line using two endpoints.
    MyLine(const cv::Point2f endpt1, const cv::Point2f endpt2, bool pCorrectDomi, bool pNotFixedEndpoints) {
        notFixedEndpoints = pNotFixedEndpoints;
        reformLine(endpt1, endpt2, pCorrectDomi);
    }

    // Default constructor.
    MyLine() {
        domi_ = cv::Point2f{1,0};
        mid_ = cv::Point2f{-1,-1};
    }

    // Creates the line using two endpoints as input.
    void reformLine(const cv::Point2f endpt1, const cv::Point2f endpt2, bool pCorrectDomi) {
        mid_ = (endpt1 + endpt2)/2;
        domi_ = endpt2-endpt1;
        endpt1_ = endpt1;
        endpt2_ = endpt2;

        // domi_ should have norm 1.
        double len = cv::norm(domi_);
        if (len > 0)
            domi_ = domi_ / len;
        else
            domi_ = cv::Point2f{1,0};

        if (pCorrectDomi) correctDomi();
        else norm_ = cv::Point2f(-domi_.y, domi_.x);

        width_ = len;
    }

    void correctDomi() {
        // Flip the dominant vector so that the normal vector
        // points "upwards". This way, the street appears
        // right side up, assuming it was written right handed.
        const double FLIP_CUTOFF = 0.1;

        norm_ = cv::Point2f(-domi_.y, domi_.x);
        if ((domi_.x < -FLIP_CUTOFF && norm_.x > 0)
            || (domi_.x < FLIP_CUTOFF && norm_.x < 0)) {
            flipDomi();
        }
    }

    // Reverse the dominant vector, and also the endpoints
    // Note, domi_ points from endpoint1 to endpoint2
    void flipDomi() {
        domi_ *= -1;
        norm_ *= -1;
        cv::Point temp = endpt1_;
        endpt1_ = endpt2_;
        endpt2_ = temp;
    }

    // Line coordinates have the origin at one end point.
    // The x coords run parallel to the line, the y coords perpendicular to the line
    // Absolute coords are relative to the overall image

    // Coordinate transform from Line coords to absolute coords
    cv::Point absolute_tranform(cv::Point2f pt) const {
        cv::Point out{(int)(mid_.x + domi_.x*(pt.x - width_/2) + norm_.x*(pt.y)),
                      (int)(mid_.y + domi_.y*(pt.x - width_/2) + norm_.y*(pt.y))  };
        return out;
    }

    // Coordinate transform from Line coords to absolute coords
    // but use the bounding box center instead of mid_
    cv::Point absolute_tranform_boundingbox(cv::Point2f pt) const {
        cv::Point out{(int)(boundingBoxCenter_.x + domi_.x*(pt.x - width_/2) + norm_.x*(pt.y)),
                      (int)(boundingBoxCenter_.y + domi_.y*(pt.x - width_/2) + norm_.y*(pt.y))  };
        return out;
    }

    // Coordinate transform from absolute coords to line coords
    cv::Point line_transform(cv::Point2f pt) const {
        cv::Point out{(int)pointToNormal(pt), (int)pointToLine(pt)};
        return out;
    }

    // Returns the shortest distance from the point p to this line.
    // The result is signed based on which side of the line it falls on.
    double pointToLine(cv::Point2f p) const {
        // do not worry about the math here
        return domi_.x*p.y - domi_.y*p.x + domi_.y*mid_.x - domi_.x*mid_.y;
    }

    // Returns the shortest distance from the point p to the
    // normal line for this line. The result is signed.
    double pointToNormal(cv::Point2f p) const {
        // do not worry about the math here
        return domi_.y*p.y + domi_.x*p.x - domi_.x*mid_.x - domi_.y*mid_.y;
    }

    void calculateDimensions(std::vector<cv::Point> points) {
        calculateDimensions(points, 0, points.size());
    }

    // Sets the width, height, and the indexes of the extremal points of the line.
    // This essentially assigns a point cloud to the line, making it into a "box"
    void calculateDimensions(std::vector<cv::Point> points, int idxSt, int idxEnd) {

        max_h = -std::numeric_limits<double>::infinity();
        max_w = -std::numeric_limits<double>::infinity();
        min_h = std::numeric_limits<double>::infinity();
        min_w = std::numeric_limits<double>::infinity();
        varH_ = 0.0;
        varW_ = 0.0;

        // Iterate over each point in the point cloud
        for (size_t i = idxSt; i < (size_t)idxEnd && i < points.size(); ++i) {
            double h = pointToLine(points[i]);
            if (h > max_h) {
                max_h = h;
                max_hi = i;
            }
            if (h < min_h) {
                min_h = h;
                min_hi = i;
            }
            double w = pointToNormal(points[i]);
            if (w > max_w) {
                max_w = w;
                max_wi = i;
            }
            if (w < min_w) {
                min_w = w;
                min_wi = i;
            }
            varH_ += h*h;
            varW_ += w*w;
        }

        // Compute metrics
        height_ = max_h - min_h;
        pixelCnt_ = idxEnd - idxSt + 1; //(int)points.size();
        varH_ /= pixelCnt_;
        varW_ /= pixelCnt_;

        // Make aware of endpoints and bounding box center
        if (notFixedEndpoints) {
            width_ = max_w - min_w;
            endpt1_ = absolute_tranform(cv::Point2f{0, 0});
            endpt2_ = absolute_tranform(cv::Point2f{width_, 0});
        }
        cv::Point2f adj_center{(float)(width_/2 + (max_w+min_w)/2), (float)(max_h+min_h)/2};
        boundingBoxCenter_ = absolute_tranform(adj_center);
    }


    // Returns the endpoints of the line.
    std::pair<cv::Point, cv::Point> endpoints() const {
        cv::Point endpt1 = absolute_tranform(cv::Point2f{0, 0});
        cv::Point endpt2 = absolute_tranform(cv::Point2f{width_, 0});
        return {endpt1, endpt2};
    }

    // Returns the bounding box of the line.
    // order: (topLeft, bottomLeft, bottomRight, topRight)
    std::vector<cv::Point> boundingBox() const {
        cv::Point TL = absolute_tranform_boundingbox(cv::Point2f{0,       height_/2});
        cv::Point BL = absolute_tranform_boundingbox(cv::Point2f{0,      -height_/2});
        cv::Point BR = absolute_tranform_boundingbox(cv::Point2f{width_, -height_/2});
        cv::Point TR = absolute_tranform_boundingbox(cv::Point2f{width_,  height_/2});

        std::vector<cv::Point> out{TL,BL,BR,TR};
        return out;

    }

    // returns a vector that starts at p, and points
    // to whichever endpoint is further from p.
    cv::Point endpointVector(cv::Point& p) {
        cv::Point p1 = endpt1_ - p;
        cv::Point p2 = endpt2_ - p;
        if (cv::norm(p1) > cv::norm(p2)) return p1;
        return p2;
    }

    // Deep copies a line
    MyLine clone() const {
        MyLine out;
        out.mid_ = mid_;
        out.domi_ = domi_;
        out.norm_ = norm_;
        out.endpt1_ = endpt1_;
        out.endpt2_ = endpt2_;
        out.vert1_ = vert1_;
        out.vert2_ = vert2_;
        out.line_id = line_id;
        out.long_line = long_line;
        out.width_ = width_;
        out.height_ = height_;
        out.varH_ = varH_;
        out.varW_ = varW_;
        out.pixelCnt_ = pixelCnt_;
        out.boundingBoxCenter_ = boundingBoxCenter_;
        out.streetId_ = streetId_;
        out.wordId_ = wordId_;
        out.streetFromCurb_ = streetFromCurb_;
        out.streetFromModern_ = streetFromModern_;
        return out;
    }

    // Scales a line
    void scale(float scale) {
        mid_ *= scale;
        endpt1_ = endpt1_ * scale;
        endpt2_ = endpt2_ * scale;
        width_ *= scale;
        height_ *= scale;
        varH_ *= scale;
        varW_ *= scale;
        boundingBoxCenter_ = boundingBoxCenter_ * scale;
    }

    // Returns a segment for the line
    Segment segment() {
        struct Segment s;
        s.p1 = endpt1_;
        s.p2 = endpt2_;
        return s;
    }

    // Texture packing stuff
    std::vector<int>
        texturePackX, texturePackY,
        texturePackW, texturePackH,
        texturePackSheet;
    std::vector<std::string> textureName;

    // Street metrics
    int streetId_ = -1;
    bool probablyNotStreet_ = false;
    bool streetFromModern_ = false;
    bool streetFromCurb_ = false;

    // Char metrics
    float charDistFromMiddle_ = -1;
    float charDistFromCurb_ = -1;

    // Word metrics
    int wordId_ = -1;
    bool isStreetName_ = false;
    bool isHouseNumber_ = false;
    bool connectedComponentPass_ = false;
    bool probablyNotWord_ = false;
    double heightConsistencyScore = -1;
    double spacingStreetNameScore = -1;
    double spacingHouseNumScore   = -1;
    double areaStreetNameScore    = -1;
    double areaHouseNumScore      = -1;
    double connCompScore          = -1;
    cv::Point wordRelativeToStreet_;
    cv::Point wordRelativeToExtendedStreet_;

};

class LongLine {
public:
    LongLine(MyLine& m) {
        line_ids.push_back(m.line_id);
        size_ = 1;
        length_ = m.width_;
        endpt1 = m.endpt1_;
        endpt2 = m.endpt2_;
        m.long_line = m.line_id;
        id_ = m.line_id;
    }

    // Add combine two long lines into a single long line.
    void append(LongLine& ll, std::vector<MyLine>& lines_list) {

        for (int i: ll.line_ids) {
            line_ids.push_back(i);
            lines_list[i].long_line = id_;
        }
        length_ += ll.length_;
        gap_jump_count += ll.gap_jump_count;
        ll.ignore_me = 1;
        size_ += ll.size_;

        double d1 = cv::norm(endpt1 - ll.endpt1);
        double d2 = cv::norm(endpt1 - ll.endpt2);
        double d3 = cv::norm(endpt2 - ll.endpt1);
        double d4 = cv::norm(endpt2 - ll.endpt2);

        if (d1 < d2 && d1 < d3 && d1 < d4) {
            endpt1 = ll.endpt2;
        } else if (d2 < d3 && d2 < d4) {
            endpt1 = ll.endpt1;
        } else if (d3 < d4) {
            endpt2 = ll.endpt2;
        } else {
            endpt2 = ll.endpt1;
        }
    }

    cv::Point endpointVector(const cv::Point& p) {
        cv::Point p1 = endpt1 - p;
        cv::Point p2 = endpt2 - p;
        if (cv::norm(p1) > cv::norm(p2)) return p1;
        return p2;
    }

    cv::Point2f unitDirVector() const {
        cv::Point2f pt = (endpt2-endpt1);
        if (pt == cv::Point2f{0,0})
            return cv::Point2f{1,0};
        pt = pt / ((float)cv::norm(pt));
        return pt;
    }

    cv::Point normVector() const {
        cv::Point p = endpt2 - endpt1;
        cv::Point q{p.y, -p.x};
        return q;
    }

    bool normOriented(const cv::Point& vec) const {
        cv::Point norm = normVector();
        return norm.dot(vec) > 0;
    }

    Segment segment() {
        struct Segment s;
        s.p1 = endpt1;
        s.p2 = endpt2;
        return s;
    }

    void forceConsistentOrderings(std::vector<MyLine>& lines, std::vector<MyVertex>& verts) {

        if (ignore_me || endpt1 == endpt2) return; // Bad long line

        // Compute domi and normal vector for long line
        cv::Point2f domi = (endpt2-endpt1);
        domi = domi / cv::norm(domi);
        cv::Point2f norm = cv::Point2f(-domi.y, domi.x);

        // Flip the endpoints, so that the dominant vector
        // follows the right handed writing rule
        const double FLIP_CUTOFF = 0.1;
        if ((domi.x < -FLIP_CUTOFF && norm.x > 0)
            || (domi.x < FLIP_CUTOFF && norm.x < 0)) {
            cv::Point temp = endpt1;
            endpt1 = endpt2;
            endpt2 = temp;
        }

        // Sort the line_ids based on their distance
        // from the first endpoint
        cv::Point2f endpt1Copy = endpt1;
        std::sort(line_ids.begin(), line_ids.end(),
                [lines, endpt1Copy](const int& a, const int& b) -> bool
                {
                    float distA = cv::norm(lines[a].mid_ - endpt1Copy);
                    float distB = cv::norm(lines[b].mid_ - endpt1Copy);
                    return distA < distB;
                });

        // Correct the dominant vectors of every line
        // in the long line, so that the endpoints
        // are properly ordered
        for (int line_id: line_ids) {
            MyLine& l = lines[line_id];
            float dist1 = cv::norm(l.endpt1_ - endpt1);
            float dist2 = cv::norm(l.endpt2_ - endpt1);
            if (dist1 > dist2) {
                // Flip this line for a consistent ordering
                l.flipDomi();
                // Make matches aware
                int temp = l.matched1;
                l.matched1 = l.matched2;
                l.matched2 = temp;
                // Make the vertexes aware of the flip
                if (l.vert1_ != -1) {
                    MyVertex v = verts[l.vert1_];
                    for (int i = 0; i < (int)(v.adj_.size()); ++i)
                        if (v.adj_[i] == l.line_id)
                            v.con_[i] = 3 - v.con_[i];
                }
                if (l.vert2_ != -1) {
                    MyVertex v = verts[l.vert2_];
                    for (int i = 0; i < (int)(v.adj_.size()); ++i)
                        if (v.adj_[i] == l.line_id)
                            v.con_[i] = 3 - v.con_[i];
                }

            }
        }

    }

    int id_ = -1;
    int size_ = 0;
    float length_ = 0 ;
    std::vector<int> line_ids; // not ordered
    cv::Point endpt1, endpt2;
    bool ignore_me = 0;
    int gap_jump_count = 0;


    // Curb line scoring metrics
    float curbLineModernScoreP = 0;
    float curbLineParaMatchScoreP = 0;
    float curbLineBranchPointScoreP = 0;
    float curbLineEndPointScoreP = 0;
    float curbLineMarkingScoreP = 0;
    float curbLineFloodFillScoreP = 0;
    float curbLineModernScoreN = 0;
    float curbLineParaMatchScoreN = 0;
    float curbLineBranchPointScoreN = 0;
    float curbLineEndPointScoreN = 0;
    float curbLineMarkingScoreN = 0;
    float curbLineFloodFillScoreN = 0;
    float curbLineScoreP = 0;
    float curbLineScoreN = 0;
    bool curbLineP = 0;
    bool curbLineN = 0;

};
