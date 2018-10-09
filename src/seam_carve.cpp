#include <cassert>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "image_proc.h"

// ------------- SEAM CARVING ------------------


// Seam carves 1/2 of a street.
void seam_carve_street(MyLine street, cv::Mat& imLines, cv::Mat& smStreet, std::vector<MyLine>& streets,
                       cv::Mat& outStMidd, cv::Mat& outStEdge, cv::Mat& outStSeam, cv::Mat& stOwn, cv::Mat& midDist) {

    // --------- HEIGHT CALCULATION -------------
    // First, we want to figure out how tall we expect the street to be at
    // each sample point.

    const int hHtSpacing = 20;
    const int vHtSpacing = 20;
    const int vSearchLimit = 500; // should be in meters

    const int xHtSamples = street.width_ / hHtSpacing;
    const int yHtSamples = vSearchLimit / vHtSpacing;

    std::vector<int> streetHeights (xHtSamples);

    for (int x = 0; x < xHtSamples; ++x) {
        bool seenStreet = false;

        for (int y = 0; y < yHtSamples; ++y) {

            cv::Point p {x*hHtSpacing, y*vHtSpacing};
            cv::Point q = street.absolute_tranform(p);
            cv::Point stQ = point_scale(q, STREET_SCALE_FACTOR);

            pixel_t pix = pixelInBounds(stQ, smStreet) ? smStreet.at<pixel_t>(stQ) : 0;

            // We will travel away from the middle until we see a transition from street
            // to not street. This measures the approximate height of the street.
            if (pix > 0 && !seenStreet) seenStreet = true;
            if (pix == 0 && seenStreet) {
                streetHeights[x] = p.y; // Records the height of the street for this sample
                break; // breaks inner loop
            }

            if (y == yHtSamples - 1) {
                streetHeights[x] = seenStreet ? vSearchLimit : 0;
            }
        }
    }


    // --------- ENERGY CALCULATION -------------
    // Now, we run the dynamic programming algorithm.

    // The slope of the allowed seam is vSpacing/hSpacing
    const int hSpacing = 1 + std::min<int>(6, (int)(street.width_ / 250));
    const int vSpacing = 2;

    const int xSamples = street.width_ / hSpacing;
    const int ySamples = vSearchLimit / vSpacing;

    // How far away from the edge of the street
    // to incur an energy penalty of 1/2 (in meters)
    const double streetCloseness = 7.0 / RESOLUTION;
    // How far past the edge of the street to consider for seam carving
    const float maxHeightRatio = 1.7;

    cv::Mat energy (xSamples, ySamples, CV_64F, 0.0);

    for (int x = 1; x < xSamples; ++x) {
        for (int y = 1; y < ySamples - 1; ++y) {

            cv::Point p{x*hSpacing, y*vSpacing};
            cv::Point q = street.absolute_tranform(p);

            int streetHeight = streetHeights[x * xHtSamples / xSamples];
            if (p.y > streetHeight*maxHeightRatio && p.y > vSearchLimit/10) {
                break; // We are too far from the edge of the street
            }

            // Is there a black pixel where we are testing
            pixel_t pix = pixelInBounds(q, imLines) ? imLines.at<pixel_t>(q) : 0;

            // Energy starts as the maximum of the previous three energies
            double e0  = energy.at<double>(x-1, y  );
            double e1  = energy.at<double>(x-1, y+1);
            double en1 = energy.at<double>(x-1, y-1);
            double e = std::max<double>({e0, e1, en1});

            // Note, we desire the seam to be close to the height of
            // the street we expect. So, that is what this penalty function does.
            double ls = 1.0/((std::abs(p.y - streetHeight)) / streetCloseness + 1);

            // The energy increase by ls, but only if there
            // is a black pixel where we are checking
            energy.at<double>(x, y) = e + (pix > 0) * ls;
        }
    }

    // --------- SEAM RECONSTRUCTION -------------
    // Reconstructs the seam with the highest energy.
    // This seam tends to lie on the inside edge of the line
    // that demarcates the edge of the street.

    double maxE = -1, prevE = -1;
    int maxY = -1;
    int seamPixelCount = 0;

    // A bonus function that gives a bonus to the part of the seam that
    // is closest to the street. This way, we can clip as close to the
    // house numbers as possible.
    const double INSIDE_ALLOWANCE = 0.05 * vSpacing;

    // Seam that we reconstruct
    std::vector<int> seam (xSamples);

    // Look in the last column for max energy
    for (int y = 1; y < ySamples - 1; ++y) {
        double e = energy.at<double>(xSamples - 1, y);

        if (e > maxE) {
            maxE = e;
            maxY = y;
            prevE = e;
        }
    }

    // Iterate backwards over the DP table
    for (int x = xSamples - 1; x >= 1; --x) {

        seam[x] = maxY;

        double e0  = energy.at<double>(x-1, maxY  );
        double e1  = energy.at<double>(x-1, maxY+1);
        double en1 = energy.at<double>(x-1, maxY-1);

        // Move the pointer backwards one step, to the one of the
        // previous three that is the largest
        if (en1 + INSIDE_ALLOWANCE > e0 && en1 + 2*INSIDE_ALLOWANCE > e1) {
            maxY -= 1;
        } else if (e0 + INSIDE_ALLOWANCE > e1) {

        } else {
            maxY += 1;
        }

        double tMax = std::max<double>({e0, e1, en1});
        if (tMax + 0.01 < prevE) {// Did the energy increase (i.e. is there a black pixel)
            seamPixelCount++; // Count the number of black pixels in the seam (goodness of fit)
        }
        prevE = tMax;

        if (maxY < 1) maxY = 1;
        if (maxY > ySamples - 1) maxY = ySamples - 1;
    }
    seam[0] = maxY;


    // --------- STREET FILLING -------------
    // Fills in the output images. We define the middle, edge, and seam of
    // the street here.

    // How many pixels are assigned to the edge of the street
    // which is the region where we will keep lines that are seam carved
    const int edgeSize = 25;
    bool goodSeam = seamPixelCount > 0.4 * xSamples;

    cv::Point prev;

    // Move horizontally across the line
    for (int x = 0; x < xSamples; ++x) {

        // Determine heights for the seam, edge, and street height
        int seamHt = seam[x];
        int edgeHt = std::max(0, seam[x] - edgeSize);
        int streetHeight = streetHeights[x * xHtSamples / xSamples];

        // Fill in the middle of the street
        for (int y = 0; (y < edgeHt && y < seamHt && y < ySamples)
                            || (!goodSeam && y < streetHeight + 20/vSpacing && y < ySamples); ++y) {
            cv::Point p{x*hSpacing, y*vSpacing};
            cv::Point q = street.absolute_tranform(p);

            if (pixelInBounds(q, outStMidd))
                outStMidd.at<pixel_t>(q) = 255;

            // Add this pixel to the street ownership matrix
            // and assign a distance from the middle
            cv::Point smQ = point_scale(q, OWNERSHIP_SCALE_FACTOR);
            if (pixelInBounds(smQ, stOwn)) {
                int curOwner = stOwn.at<int>(smQ) - 1;
                // note, we prefer to give ownership to longer streets, since they are more reliable.
                if (curOwner == -1 || streets[curOwner].width_ < street.width_ - SEAM_CARVE_EXTENSION) {
                    stOwn.at<int>(smQ) = street.streetId_ + 1;
                    float curMidDist = midDist.at<float>(smQ);
                    midDist.at<float>(smQ) = curMidDist == 0 ? y + 0.01f : std::min<float>(y+0.01f, curMidDist);
                }

            }

        }

        // If the seam we find is not good enough, just assign the pixels to the
        // middle of the street. Don't create an edge or a seam.
        if (!goodSeam) continue;

        for (int y = std::max<int>(edgeHt, 0); y < seamHt - 1 && y < ySamples; ++y) {
            cv::Point p{x*hSpacing, y*vSpacing};
            cv::Point q = street.absolute_tranform(p);

            if (pixelInBounds(q, outStEdge))
                outStEdge.at<pixel_t>(q) = 255;

            // Duplicate code from above
            cv::Point smQ = point_scale(q, OWNERSHIP_SCALE_FACTOR);
            if (pixelInBounds(smQ, stOwn)) {
                int curOwner = stOwn.at<int>(smQ) - 1;
                if (curOwner == -1 || streets[curOwner].width_ < street.width_ - SEAM_CARVE_EXTENSION) {
                    stOwn.at<int>(smQ) = street.streetId_ + 1;
                    float curMidDist = midDist.at<float>(smQ);
                    midDist.at<float>(smQ) = curMidDist == 0 ? y + 0.01f : std::min<float>(y+0.01f, curMidDist);
                }

            }

        }

        // Fill in the seam
        cv::Point p{x*hSpacing, seamHt*vSpacing};
        cv::Point q = street.absolute_tranform(p);
        const int seam_thickness = 4;
        if (x > 0) // Draw a line for the seam.
            cv::line(outStSeam, q, prev, cv::Scalar{255}, seam_thickness);
        prev = q;
    }

}
