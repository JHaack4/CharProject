#include <cassert>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "image_proc.h"

// Determines if the point p=(c,r) falls within
// the bounds of matrix m
bool pixelInBounds(cv::Point p, cv::Mat& m) {
    return p.x >= 0 && p.y >= 0 && p.x < m.cols && p.y < m.rows;
}
bool pixelInBounds(int r, int c, cv::Mat& m) {
    return c >= 0 && r >= 0 && c < m.cols && r < m.rows;
}

// Scales a point p by the scale factor
cv::Point point_scale(cv::Point2f p, double scale_factor) {
    return cv::Point{(int)(scale_factor * p.x), (int)(scale_factor * p.y)};
}

// Finds whatever owns that pixel in the ownership matrix
// Return -1 if there is no owner
int searchForOwner(cv::Point2f p, cv::Mat& ownership) {

    cv::Point q = point_scale(p, OWNERSHIP_SCALE_FACTOR);

    int pix = pixelInBounds(q, ownership) ?
                    ownership.at<int>(q) - 1 : -1;

    // Look a little harder for an owner
    if (pix == -1) {
        for (int xx = q.x - 2; xx <= q.x + 2; ++xx) {
            for (int yy = q.y - 2; yy <= q.y + 2; ++yy) {
                cv::Point q2 {xx,yy};
                int pix2 = pixelInBounds(q2, ownership) ?
                    ownership.at<int>(q2) - 1 : -1;
                if (pix2 != -1) return pix2;
            }
        }
    }

    return pix;
}

// Finds whatever owns that pixel in the ownership matrix
// Return -1 if there is no owner
float searchForOwnerF(cv::Point2f p, cv::Mat& ownership) {

    cv::Point q = point_scale(p, OWNERSHIP_SCALE_FACTOR);

    float pix = pixelInBounds(q, ownership) ?
                    ownership.at<float>(q) : -1.0;

    return pix;
}

// This method runs connected components of the img. Then, for each connected
// component (not part of the background), it creates a vector of all of the
// points that can be found inside that connected component.
std::vector< std::vector<cv::Point>> connectedComponentsWithIdx(cv::Mat& img, cv::Mat& labels) {

    size_t numCC = cv::connectedComponents(img, labels);

    // Create a pixel idx list for each connected component
    std::vector< std::vector<cv::Point>> points_list (numCC);
    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            uint pix = labels.at<uint>(r, c);
            if (pix > 0) {
                cv::Point pt {c, r}; // note: x=cols, y=rows
                points_list[pix].push_back(pt);
            }
        }
    }

    return points_list;

}

// obtains the average x-position of the pixels
double averageXPos(const std::vector<cv::Point>& pixels)
{
    double xpos = 0;
    for (auto& pt : pixels) {
        xpos += pt.x;
    }
    return xpos / pixels.size();
}


// This method is unused.
// This method runs connected components of the skeleton image skel. The connected
// components are broken up by any branch points or end points in the image branchEndPoints.
// Each connected component is returned in a reasonable ordering.
std::vector< std::vector<cv::Point>> connectedComponentsSkel(cv::Mat& skel, cv::Mat& branchEndPoints)
{

    // Create a pixel idx list for each connected component
    std::vector< std::vector<cv::Point>> points_list;

    std::vector<cv::Point> critical_points;
    int VISITED = 254; // We will slightly modify skel

    // Mark the critical points using the branch points
    // A critical point is the endpoint of any individual
    // branch of the image.
    for (int r = 0; r < skel.rows; ++r) {
        for (int c = 0; c < skel.cols; ++c) {

            // Consider every branch/end point
            pixel_t pix1 = branchEndPoints.at<pixel_t>(r, c);
            if (pix1 > 0) {

                // Consider a small radius around the branch point
                const int rad = 2;
                for (int rr = r-rad; rr <= r+rad; ++rr) {
                    for (int cc = c-rad; cc <= c+rad; ++cc) {

                        if (!pixelInBounds(rr, cc, skel))
                            continue; // pixel out of bounds

                        pixel_t pix = skel.at<pixel_t>(rr, cc);

                        // If the skeleton point is too close to the branch point,
                        // mark it as visited and continue
                        if (std::abs(rr-r) <= rad-1 && std::abs(cc-c) <= rad-1) {
                            skel.at<pixel_t>(rr, cc) = VISITED;
                            continue;
                        }

                        if (pix > 0) { // Otherwise, it is a critical point
                            cv::Point pt {cc, rr};
                            critical_points.push_back(pt);
                        }
                    }
                }
            }

        }
    }

    // Now, for each critical point, we launch a breadth first search
    // to find every other pixel in this branch, in order of closeness.
    for (size_t i = 0; i < critical_points.size(); ++i) {
        pixel_t pix1 = skel.at<pixel_t>(critical_points[i]);
        if (pix1 > 0 && pix1 != VISITED) {

            std::vector<cv::Point> points;
            std::queue<cv::Point> bfs;
            bfs.push(critical_points[i]);

            while (!bfs.empty()) {

                cv::Point pt = bfs.front();
                bfs.pop();

                if (!pixelInBounds(pt, skel))
                    continue; // OOB

                pixel_t pix = skel.at<pixel_t>(pt);
                if (pix == 0 || pix == VISITED) {
                    continue; // Already visited
                }

                skel.at<pixel_t>(pt) = VISITED;
                points.push_back(pt);

                // Now, search the neighbors
                bfs.push(cv::Point {pt.x+1, pt.y+0} );
                bfs.push(cv::Point {pt.x-1, pt.y+0} );
                bfs.push(cv::Point {pt.x+0, pt.y+1} );
                bfs.push(cv::Point {pt.x+0, pt.y-1} );
                bfs.push(cv::Point {pt.x+1, pt.y+1} );
                bfs.push(cv::Point {pt.x+1, pt.y-1} );
                bfs.push(cv::Point {pt.x-1, pt.y+1} );
                bfs.push(cv::Point {pt.x-1, pt.y-1} );
            }

            points_list.push_back(points);
        }
    }

    return points_list;

}

// Crops and rotates out a section of img using
// the bounding box of the MyLine line.
cv::Mat my_rotate(cv::Mat& img, MyLine& line) {

    if (line.width_ < 0.5 || line.height_ < 0.5) {
        return cv::Mat{1,1,img.type(), 0.0};
    }

    int width = (int)(line.width_ + 0.5);
    int height = (int)(line.height_ + 0.5);

    cv::Mat output(height, width, img.type(), 0.0);

    for (int r = 0; r < height; ++r) {
      for (int c = 0; c < width; ++c) {

        // Apply a coordinate transform (line coords to absolute)
        // to know what pixel to look at
        cv::Point p = line.absolute_tranform_boundingbox(cv::Point{c, r - height/2});

        if (pixelInBounds(p, img)) {
            pixel_t pix = img.at<pixel_t>(p);
            output.at<pixel_t>(r,c) = pix;
        }

      }
    }

    return output;
}

// Deskews an image by factor skew
cv::Mat my_deskew(cv::Mat& img, double skew) {

    cv::Mat out(img.rows, img.cols, img.type(), 0.0);

    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            double nc = c + (img.rows/2.0 - r) * skew; // apply skew
            int sc = (int)(nc + 0.5);
            int sr = r;

            if (pixelInBounds(sr, sc, img)) {
                pixel_t pix = img.at<pixel_t>(sr,sc);
                out.at<pixel_t>(r,c) = pix;
            }
        }
    }

    return out;
}

// Takes in a list of white pixels "points" and a padding, and turns those
// points into a padded image
cv::Mat idxToImg(std::vector<cv::Point>& points, const size_t pad) {

    int min_r = 1000000;
    int max_r = -1000000;
    int min_c = 1000000;
    int max_c = -1000000;

    for (cv::Point pt: points) {
        min_r = std::min<int>(min_r, pt.y);
        max_r = std::max<int>(max_r, pt.y);
        min_c = std::min<int>(min_c, pt.x);
        max_c = std::max<int>(max_c, pt.x);
    }

    size_t rows = max_r - min_r + 1 + 2*pad,
           cols = max_c - min_c + 1 + 2*pad;
    cv::Mat m (rows, cols, CV_8UC1, 0.0);

    for (cv::Point pt: points) {
        int rr = pt.y - min_r + pad;
        int cc = pt.x - min_c + pad;
        m.at<pixel_t>(rr, cc) = 255;
    }

    return m;
}

// This function is unused
// Creates an image using the white pixel list provided,
// relative to the line.
cv::Mat idxRotate(std::vector<cv::Point>& points, MyLine& line) {

    if (line.width_ < 0.5 || line.height_ < 0.5) {
        return cv::Mat{1,1,CV_8U, 0.0};
    }

    int width = (int)(line.width_ + 0.5);
    int height = (int)(line.height_ + 0.5);

    cv::Mat output(height, width, CV_8U, 0.0);

    for (cv::Point p: points) {
        cv::Point q = line.line_transform(p);
        cv::Point trQ1 {q.x + width/2    , q.y + height/2    };
        cv::Point trQ5 {q.x + width/2    , q.y + height/2 + 1};

        if (pixelInBounds(trQ1, output)) {
            output.at<pixel_t>(trQ1) = 255;
        }
        if (pixelInBounds(trQ5, output)) {
            output.at<pixel_t>(trQ5) = 255;
        }
    }

    return output;
}


// In house fit line implementation using LSRL. Not used currently.
void my_fit_line(std::vector<cv::Point>& points, cv::Point2f& centroid, cv::Point2f& domi) {

    double X = 0, Y = 0, XY = 0, X2 = 0, Y2 = 0;
    int N = 0;

    for (cv::Point2f pt: points) {
        X  += pt.x;
        Y  += pt.y;
        XY += pt.x * pt.y;
        X2 += pt.x * pt.x;
        Y2 += pt.y * pt.y;
        N++;
    }

    centroid.x = X / N;
    centroid.y = Y / N;

    double A  = XY - X*Y/N;
    double Sx = X2 - X*X/N;
    double Sy = Y2 - Y*Y/N;

    double t = atan2( 2 * A, Sx - Sy ) / 2;
    domi.x = cos( t );
    domi.y = sin( t );


}

// Fit a MyLine object to a point cloud.
MyLine fitPointCloud(std::vector<cv::Point>& points) {

    if (points.size() < 2) {
        cv::Point2f p {-1.0,-1.0};
        cv::Point2f p2 {-1.0,-1.0};
        return MyLine {p,p2};
    }


    // Find the best fit line for the point cloud "points"
    cv::Vec4f fitline;
    cv::fitLine(points, fitline, CV_DIST_L2, 0, 0.001, 0.001);

    cv::Point2f midpoint{fitline[2], fitline[3]};
    cv::Point2f domi{fitline[0], fitline[1]};
    MyLine line{midpoint, domi};

    // Compute height, width, etc of the cloud
    line.calculateDimensions(points);

    return line;

}

// Fit a MyLine object to a point cloud. Instead of doing a fitLine
// call, use the domi vector provided. If useBoxCenter, use the center
// of the bounding box, instead of the centroid.
MyLine fitPointCloud(std::vector<cv::Point>& points, cv::Point2f domi, bool useBoxCenter) {

    if (points.size() < 1) {
        cv::Point2f p {-1.0,-1.0};
        cv::Point2f p2 {-1.0,-1.0};
        return MyLine {p,p2};
    }

    float X = 0, Y = 0;

    for (cv::Point2f pt: points) {
        X += pt.x;
        Y += pt.y;
    }

    cv::Point2f centroid{X/points.size(), Y/points.size()};
    MyLine line {centroid, domi};

    line.calculateDimensions(points);

    if (useBoxCenter) {
        line.mid_ = line.boundingBoxCenter_;
    }

    return line;

}


// Creates a texture pack image, which is essentially a compilation of
// a bunch of images into a single image.
void texturepack(cv::Mat& pack, cv::Mat& img, MyLine& word, int& lastR, int& lastC, int max_height, std::string fileName, int& fileCount) {

    // This row is too full. Move on to the next row
    if (img.cols + lastC > pack.cols) {
        lastC = 0;
        lastR += max_height + 1;
    }

    // This image is too full. Output this image and
    // move on to a new texture pack
    if (lastR + max_height > pack.rows) {
        midway_imwrite(pack, fileName + std::to_string(fileCount));
        fileCount++;
        lastR = 0;
        lastC = 0;
        pack = cv::Mat::zeros( pack.size(), pack.type() );
    }

    // Record the location for the JSON
    word.texturePackH.push_back(std::min<int>(img.rows, max_height));
    word.texturePackW.push_back(std::min<int>(img.cols, pack.cols));
    word.texturePackX.push_back(lastC);
    word.texturePackY.push_back(lastR);
    word.texturePackSheet.push_back(fileCount);
    word.textureName.push_back("idk");

    // Copies the image to the texture pack image
    for (int r = 0; r < img.rows && r < max_height; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            pixel_t pix = img.at<pixel_t>(r,c);
            int sr = lastR + r;
            int sc = lastC + c;
            if (pixelInBounds(sr,sc,pack))
                pack.at<pixel_t>(sr, sc) = pix;
        }
    }

    lastC += img.cols + 1;

}





