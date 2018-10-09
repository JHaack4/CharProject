// This file is unused

/**
 * An implementation of the Zhang Suen thinning algorithm, taken from
 * https://github.com/bsdnoobz/zhang-suen-thinning
 */
#include "opencv2/opencv.hpp"

// These are precomputed neighborhood matrices
// A neighborhood is the 3x3 region around a white pixel
// There is a mapping from neighborhood to uchar (byte)
// as follows: LSB=no,ne,ea,se,so,sw,we,nw=MSB
// The value in each of these arrays gives information
// about what to do with a pixel, given its neighborhood.

const bool skel0[256] = {
                    0,0,0,1,0,0,1,1,0,0,0,0,1,0,1,1,
                    0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
                    0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0 };
const bool skel1[256] = {
                    0,0,0,1,0,0,1,1,0,0,0,0,1,0,1,1,
                    0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,
                    0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0 };
const bool shrink0[256] = {
                    0,1,1,1,1,0,1,1,1,0,0,0,1,0,1,1,
                    1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,
                    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,
                    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0 };
const bool shrink1[256] = {
                    0,1,1,1,1,0,1,1,1,0,0,0,1,0,1,1,
                    1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,
                    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,
                    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,
                    1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0 };
const bool branch[256] = {
                    0,1,1,1,1,0,1,1,1,0,0,0,1,0,1,1,
                    1,0,0,0,0,1,0,0,1,0,0,0,1,0,1,1,
                    1,0,0,0,0,1,0,0,0,1,1,1,0,1,0,0,
                    1,0,0,0,0,1,0,0,1,0,0,0,1,0,1,1,
                    1,0,0,0,0,1,0,0,0,1,1,1,0,1,0,0,
                    0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,
                    1,0,0,0,0,1,0,0,0,1,1,1,0,1,0,1,
                    1,0,0,0,0,1,0,1,1,0,0,1,1,1,1,0,
                    1,1,0,1,0,0,0,1,0,0,1,0,0,0,0,1,
                    0,0,1,0,1,1,1,0,0,0,1,0,0,0,0,1,
                    0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,
                    0,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,
                    1,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,
                    0,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,
                    1,1,0,1,0,0,0,0,0,0,1,1,0,1,1,0,
                    1,0,0,0,0,1,1,0,1,0,1,0,1,0,0,0 };

const int connected_nbrs[256] = {
  0,1,1,1,1,1,1,1,
  1,2,2,2,1,1,1,1,
  1,2,2,2,1,1,1,1,
  1,2,2,2,1,1,1,1,
  1,2,2,2,2,2,2,2,
  2,3,3,3,2,2,2,2,
  1,2,2,2,1,1,1,1,
  1,2,2,2,1,1,1,1,
  1,1,2,1,2,1,2,1,
  2,2,3,2,2,1,2,1,
  1,1,2,1,1,1,1,1,
  1,1,2,1,1,1,1,1,
  1,1,2,1,2,1,2,1,
  2,2,3,2,2,1,2,1,
  1,1,2,1,1,1,1,1,
  1,1,2,1,1,1,1,1,
  1,1,2,1,2,1,2,1,
  2,2,3,2,2,1,2,1,
  2,2,3,2,2,1,2,1,
  2,2,3,2,2,1,2,1,
  2,2,3,2,3,2,3,2,
  3,3,4,3,3,2,3,2,
  2,2,3,2,2,1,2,1,
  2,2,3,2,2,1,2,1,
  1,1,2,1,2,1,2,1,
  2,2,3,2,2,1,2,1,
  1,1,2,1,1,1,1,1,
  1,1,2,1,1,1,1,1,
  1,1,2,1,2,1,2,1,
  2,2,3,2,2,1,2,1,
  1,1,2,1,1,1,1,1,
  1,1,2,1,1,1,1,1
};

const int nbr_branches[256] = {
  0,1,1,1,1,2,1,2,
  1,2,2,2,1,2,2,2,
  1,2,2,2,2,3,2,3,
  1,2,2,2,2,3,2,3,
  1,2,2,2,2,3,2,3,
  2,3,3,3,2,3,3,3,
  1,2,2,2,2,3,2,3,
  2,3,3,3,2,3,3,3,
  1,2,2,2,2,3,2,3,
  2,3,3,3,2,3,3,3,
  2,3,3,3,3,4,3,4,
  2,3,3,3,3,4,3,4,
  1,2,2,2,2,3,2,3,
  2,3,3,3,2,3,3,3,
  2,3,3,3,3,4,3,4,
  2,3,3,3,3,4,3,4,
  1,1,2,2,2,2,2,2,
  2,2,3,3,2,2,3,3,
  2,2,3,3,3,3,3,3,
  2,2,3,3,3,3,3,3,
  2,2,3,3,3,3,3,3,
  3,3,4,4,3,3,4,4,
  2,2,3,3,3,3,3,3,
  3,3,4,4,3,3,4,4,
  1,2,2,2,2,3,2,3,
  2,3,3,3,2,3,3,3,
  2,3,3,3,3,4,3,4,
  2,3,3,3,3,4,3,4,
  2,2,3,3,3,3,3,3,
  3,3,4,4,3,3,4,4,
  2,3,3,3,3,4,3,4,
  3,3,4,4,3,4,4,4,
};

const bool jhskel[256] = {
                    0,0,0,1,0,1,1,1,0,0,0,0,1,1,1,1,
                    0,0,0,0,1,0,1,0,1,0,0,0,1,0,1,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,1,
                    0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
                    1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,
                    0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,
                    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,0,1,1,0,0,1,0,0,0 };

const bool jhshrink[256] = {
                    0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,
                    1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,1,
                    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,1,
                    1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
                    1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,
                    1,1,0,1,0,1,0,1,0,0,0,0,0,0,0,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,
                    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,0,1,1,0,0,1,0,0,0 };

// skel=true implies use Zhang-Suen skeleton
// skel=false implies use Zhang-Suen shrink
// if jhmod=true, use Jordan's modification to Z-S
bool check_neighborhood(uchar hood, int itertype, bool skel, bool jhmod) {

    int no = (hood & 1) > 0;
    int ne = (hood & 2) > 0;
    int ea = (hood & 4) > 0;
    int se = (hood & 8) > 0;
    int so = (hood & 16) > 0;
    int sw = (hood & 32) > 0;
    int we = (hood & 64) > 0;
    int nw = (hood & 128) > 0;

    if (jhmod)
    {
        int A = (no == 0 && ne != 0) + (ne == 0 && ea != 0) +
                (ea == 0 && se != 0) + (se == 0 && so != 0) +
                (so == 0 && sw != 0) + (sw == 0 && we != 0) +
                (we == 0 && nw != 0) + (nw == 0 && no != 0);
        int B  = (no!=0) + (ne!=0) + (ea!=0) + (se!=0) + (so!=0) + (sw!=0) + (we!=0) + (nw!=0);
            //int m1 = ;
            //int m2 = itertype == 0 ? (ea * so * we) : (no * so * we);

        bool discard1 = (A==1 && B > 1 && B <= 6)
                || (A==2 && B==2 && ((no!=0)*(ea!=0) | (no!=0)*(we!=0) | (so!=0)*(we!=0) | (so!=0)*(ea!=0)) != 0)
                || (A==2 && B==3 && ((no!=0)*(ea!=0)*(ne==0) | (no!=0)*(we!=0)*(nw==0) | (so!=0)*(we!=0)*(sw==0) | (so!=0)*(ea!=0)*(se==0)) != 0);

        bool discard2 = (A==1 && B >= 1 && B <= 6)
                || (A==2 && B==2 && ((no!=0)*(ea!=0) | (no!=0)*(we!=0) | (so!=0)*(we!=0) | (so!=0)*(ea!=0)) != 0)
                || (A==2 && B==3 && ((no!=0)*(ea!=0)*(ne==0) | (no!=0)*(we!=0)*(nw==0) | (so!=0)*(we!=0)*(sw==0) | (so!=0)*(ea!=0)*(se==0)) != 0);

        if (skel) return discard1;
        else return discard2;
 // idk if this helps, but tries to handle T's
 //               || (A==3 && B==3 && ((no!=0)*(ea!=0)*(so!=0) | (no!=0)*(we!=0)*(ea!=0) | (so!=0)*(we!=0)*(no!=0) | (so!=0)*(ea!=0)*(we!=0)) != 0);

    }
    int A = (no == 0 && ne == 1) + (ne == 0 && ea == 1) +
            (ea == 0 && se == 1) + (se == 0 && so == 1) +
            (so == 0 && sw == 1) + (sw == 0 && we == 1) +
            (we == 0 && nw == 1) + (nw == 0 && no == 1);
    int B  = no + ne + ea + se + so + sw + we + nw;
    int m1 = itertype == 0 ? (no * ea * so) : (no * ea * we);
    int m2 = itertype == 0 ? (ea * so * we) : (no * so * we);

    if (itertype == 2 && (A >= 3 || (A == 2 && B == 6)))
        return 1;
    else if (skel) { //skeleton
        if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
            return 1;
    } else { //shrink
        if (A == 1 && (B >= 1 && B <= 6) && m1 == 0 && m2 == 0)
            return 1;
    }
    return 0;
}

// Print out those pretty arrays like above
void print_nbhd_array(int itertype, bool skel, bool jhmod) {

    std::cout << "\nconst bool arr[256] = {\n                    ";

    for (int i = 0; i < 256; ++i) {
        std::cout << (check_neighborhood(i, itertype, skel, jhmod) ? "1" : "0") << ",";
        if (i%16==15) std::cout << "\n                    ";
    }
    std::cout << "\n                    };\n\n";
}

void generate_nbhd_array(uchar arr[], int itertype, bool skel, bool jhmod) {

    for (int i = 0; i < 256; ++i) {
        arr[i] = check_neighborhood(i, itertype, skel, jhmod);
    }

}

// This is the old and slow version. Don't use

// iterations 0 and 1 are correspond to the thinning/shrinking
// algorithm.  Iteration 2 is for finding branchpoints
void thinningIteration(cv::Mat& img, int iter, bool skel)
{
    if (iter == 2 && skel) return; // This line is probably incorrect

    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);
    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

    int nRows = img.rows;
    int nCols = img.cols;

    if (img.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }

    int x, y;
    uchar *pAbove;
    uchar *pCurr;
    uchar *pBelow;
    uchar *nw, *no, *ne;    // north (pAbove)
    uchar *we, *me, *ea;
    uchar *sw, *so, *se;    // south (pBelow)
    uchar *pDst;
    // uchar *pBranchPoints;

    // initialize row pointers
    pAbove = NULL;
    pCurr  = img.ptr<uchar>(0);
    pBelow = img.ptr<uchar>(1);

    for (y = 1; y < img.rows-1; ++y) {
        // shift the rows up by one
        pAbove = pCurr;
        pCurr  = pBelow;
        pBelow = img.ptr<uchar>(y+1);

        pDst = marker.ptr<uchar>(y);
        // pBranchPoints = bpmarker.ptr<uchar>(y);

        // initialize col pointers
        no = &(pAbove[0]);
        ne = &(pAbove[1]);
        me = &(pCurr[0]);
        ea = &(pCurr[1]);
        so = &(pBelow[0]);
        se = &(pBelow[1]);

        for (x = 1; x < img.cols-1; ++x) {
            // shift col pointers left by one (scan left to right)
            nw = no;
            no = ne;
            ne = &(pAbove[x+1]);
            we = me;
            me = ea;
            ea = &(pCurr[x+1]);
            sw = so;
            so = se;
            se = &(pBelow[x+1]);

            //if (*me == 0) continue;

            int A  = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
                     (*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
                     (*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
                     (*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
            int B  = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
            int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
            int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

            if (skel) {
                if (iter < 2 && A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                    pDst[x] = 1;
            } else { //shrink
                if (iter < 2 && A == 1 && (B >= 1 && B <= 6) && m1 == 0 && m2 == 0)
                    pDst[x] = 1;
                // TODO: this is a HACK.  Figure out actual requirements for branchpoints
                else if (iter == 2 && (A >= 3 || (A == 2 && B == 6)))
                    pDst[x] = 1;
            }
        }
    }
    if (iter == 2)
        img &= marker;
    else
        img &= ~marker;
}

/*
 * Function for thinning the given binary image
 *
 * Parameters:
 *      src  The source image, binary with range = [0,255]
 *      dst  The destination image
*/
void thinning(const cv::Mat& src, cv::Mat& dst, cv::Mat& branch_points, bool skel, int max_iter)
{
    dst = src.clone();
    dst /= 255;         // convert to binary image

    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::Mat diff;
    do {
        thinningIteration(dst, 0, skel);
        thinningIteration(dst, 1, skel);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
        --max_iter;
        std::cout << "here" << std::endl;
    }
    while (cv::countNonZero(diff) > 0 && max_iter > 0);

    branch_points = dst.clone();
    thinningIteration(branch_points, 2, false);

    dst *= 255;
    branch_points *= 255;
}


// This is Jordan's modified version of Z-S

// iteration 0  corresponds to the thinning/shrinking
// algorithm.  Iteration 2 is for finding branchpoints
void thinningIteration4(cv::Mat& img, int itertype, bool skel, bool& loopAgain,
                        cv::Mat& nextWhiteX, cv::Mat& nextWhiteY,
                        int& firstWhiteX, int& firstWhiteY, bool firstRun)
{
    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 2 && img.cols > 2);

    uchar *pAbove;
    uchar *pCurr;
    uchar *pBelow;
    uchar nw, no, ne;    // north (pAbove)
    uchar we, me, ea;
    uchar sw, so, se;    // south (pBelow)
    uchar *pDst;
    int *pWhiteX, *pWhiteY;

    // initialize white pixel trackers
    int* lastWhitePixelX = &firstWhiteX;
    int* lastWhitePixelY = &firstWhiteY;
    int x = firstWhiteX, y = firstWhiteY;

    int prevY = -1;

    while (x < img.cols-1 && y < img.rows-1) {


            if (y != prevY) {
                pAbove = img.ptr<uchar>(y-1);
                pCurr  = img.ptr<uchar>(y  );
                pBelow = img.ptr<uchar>(y+1);

                pDst = pCurr;
                pWhiteX = nextWhiteX.ptr<int>(y);
                pWhiteY = nextWhiteY.ptr<int>(y);
            }
            prevY = y;

            me = pCurr[x  ];
            if (me == 0) { // This code should only be hit on the first run
                x += x%2==0 ? 3 : (x==1?2:-1);
                if (x >= img.cols-1) {
                    x = 1;
                    y += y%2==0 ? 3 : (y==1?2:-1);
                }
                continue;
            };

            // shift col pointers left by one (scan left to right)
            nw = pAbove[x-1];
            no = pAbove[x  ];
            ne = pAbove[x+1];
            we = pCurr[x-1];

            ea = pCurr[x+1];
            sw = pBelow[x-1];
            so = pBelow[x  ];
            se = pBelow[x+1];

            uchar neighborhood = ((no != 0) << 0)
                               + ((ne != 0) << 1)
                               + ((ea != 0) << 2)
                               + ((se != 0) << 3)
                               + ((so != 0) << 4)
                               + ((sw != 0) << 5)
                               + ((we != 0) << 6)
                               + ((nw != 0) << 7);

            if (itertype == 2 && branch[neighborhood])
                pDst[x] = 1;
            else if (skel && jhskel[neighborhood]) //skeleton
                pDst[x] = 0;
            else if (!skel && jhshrink[neighborhood])
                pDst[x] = 0; // Shrink doesn't work so don't use it

            if (pDst[x] == 0) // This pixel is turning black
                loopAgain = true;
            else {
                *lastWhitePixelX = x;
                *lastWhitePixelY = y;
                lastWhitePixelX = &(pWhiteX[x]);
                lastWhitePixelY = &(pWhiteY[x]);

            }

            if (firstRun) {
                x += x%2==0 ? 3 : (x==1?2:-1);
                if (x >= img.cols-1) {
                    x = 1;
                    y += y%2==0 ? 3 : (y==1?2:-1);
                }
            }
            else { // jump directly to next white pixel
                y = pWhiteY[x];
                x = pWhiteX[x];
            }

    }

    *lastWhitePixelX = img.cols;
    *lastWhitePixelY = img.rows;
}

/**
 * Function for thinning the given binary image
 *
 * Parameters:
 *      src  The source image, binary with range = [0,255]
 *      dst  The destination image
 */
void thinning4(const cv::Mat& src, cv::Mat& dst, cv::Mat& branch_points, bool skel, int max_iter)
{
    dst = src.clone();

    cv::Mat nextWhiteX = cv::Mat::zeros(dst.size(), CV_32SC1);
    cv::Mat nextWhiteY = cv::Mat::zeros(dst.size(), CV_32SC1);
    int firstWhiteX = 1, firstWhiteY = 1;

    bool loopAgain1 = false, firstRun = true;
    do {
        loopAgain1 = false;
        thinningIteration4(dst, 0, skel, loopAgain1,  nextWhiteX, nextWhiteY, firstWhiteX, firstWhiteY, firstRun);
        firstRun = false;
        --max_iter;
        std::cout << "here4" << std::endl;
    }
    while (loopAgain1 && max_iter > 0);

    branch_points = dst.clone();
    thinningIteration4(branch_points, 2, false, loopAgain1,  nextWhiteX, nextWhiteY, firstWhiteX, firstWhiteY, false);

    dst = dst > 0;
    branch_points = branch_points == 1;
}

// This is the exact same algorithm as Z-S, but much faster

// iterations 0 and 1 are correspond to the thinning/shrinking
// algorithm.  Iteration 2 is for finding branchpoints
void thinningIteration5(cv::Mat& img, int itertype, bool skel, bool& loopAgain,
                        cv::Mat& markerMask, cv::Mat& nextWhiteX, cv::Mat& nextWhiteY,
                        int& firstWhiteX, int& firstWhiteY, bool firstRun)
{
    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 2 && img.cols > 2);

    uchar *pAbove;
    uchar *pCurr;
    uchar *pBelow;
    uchar nw, no, ne;    // north (pAbove)
    uchar we, me, ea;
    uchar sw, so, se;    // south (pBelow)
    uchar *pDst;
    int *pWhiteX, *pWhiteY;

    // initialize row pointers
    pAbove = NULL;
    pCurr  = img.ptr<uchar>(0);
    pBelow = img.ptr<uchar>(1);

    // initialize white pixel trackers
    int* lastWhitePixelX = &firstWhiteX;
    int* lastWhitePixelY = &firstWhiteY;
    int x = firstWhiteX, y = firstWhiteY;

    int prevY = -1;


        while (x < img.cols-1 && y < img.rows-1) {

            if (y != prevY) {
                pAbove = img.ptr<uchar>(y-1);
                pCurr  = img.ptr<uchar>(y  );
                pBelow = img.ptr<uchar>(y+1);

                pDst = markerMask.ptr<uchar>(y);
                pWhiteX = nextWhiteX.ptr<int>(y);
                pWhiteY = nextWhiteY.ptr<int>(y);
            }
            prevY = y;

            me = pCurr[x  ];
            if (me == 0) { // This code should only be hit on the first run
                x++;
                if (x == img.cols-1) {
                    x = 1;
                    y++;
                }
                continue;
            };

            // shift col pointers left by one (scan left to right)
            nw = pAbove[x-1];
            no = pAbove[x  ];
            ne = pAbove[x+1];
            we = pCurr[x-1];

            ea = pCurr[x+1];
            sw = pBelow[x-1];
            so = pBelow[x  ];
            se = pBelow[x+1];

            uchar neighborhood = ((no != 0) << 0)
                               + ((ne != 0) << 1)
                               + ((ea != 0) << 2)
                               + ((se != 0) << 3)
                               + ((so != 0) << 4)
                               + ((sw != 0) << 5)
                               + ((we != 0) << 6)
                               + ((nw != 0) << 7);

            if (itertype == 2 && branch[neighborhood])
                pDst[x] = 255;
            else if (skel && itertype == 0 && skel0[neighborhood]) //skeleton
                pDst[x] = 0;
            else if (skel && itertype == 1 && skel1[neighborhood]) //skeleton
                pDst[x] = 0;
            else if (!skel && itertype == 0 && shrink0[neighborhood]) //shrink
                pDst[x] = 0;
            else if (!skel && itertype == 1 && shrink1[neighborhood]) //shrink
                pDst[x] = 0;

            if (pDst[x] == 0) // This pixel is turning black
                loopAgain = true;
            else {
                *lastWhitePixelX = x;
                *lastWhitePixelY = y;
                lastWhitePixelX = &(pWhiteX[x]);
                lastWhitePixelY = &(pWhiteY[x]);

            }

            if (firstRun) {
                x++;
                if (x == img.cols-1) {
                    x = 1;
                    y++;
                }
            }
            else { // jump directly to next white pixel
                y = pWhiteY[x];
                x = pWhiteX[x];
            }

        }

    *lastWhitePixelX = img.cols;
    *lastWhitePixelY = img.rows;
    img &= markerMask;
}

/**
 * Function for thinning the given binary image
 *
 * Parameters:
 *      src  The source image, binary with range = [0,255]
 *      dst  The destination image
 */
void thinning5(const cv::Mat& src, cv::Mat& dst, cv::Mat& branch_points, bool skel, int max_iter)
{
    dst = src.clone();
    //dst /= 255;         // convert to binary image

    cv::Mat markerMask = cv::Mat::zeros(dst.size(), CV_8UC1);
    markerMask = 255-markerMask;
    cv::Mat nextWhiteX = cv::Mat::zeros(dst.size(), CV_32SC1);
    cv::Mat nextWhiteY = cv::Mat::zeros(dst.size(), CV_32SC1);
    int firstWhiteX = 1, firstWhiteY = 1;

    bool loopAgain1 = false, loopAgain2 = false, firstRun = true;
    do {
        loopAgain1 = false; loopAgain2 = false;
        thinningIteration5(dst, 0, skel, loopAgain1, markerMask, nextWhiteX, nextWhiteY, firstWhiteX, firstWhiteY, firstRun);
        firstRun = false;
        //std::cout << "here5a" << std::endl;
        thinningIteration5(dst, 1, skel, loopAgain2, markerMask, nextWhiteX, nextWhiteY, firstWhiteX, firstWhiteY, false);
        --max_iter;
        //std::cout << "here5" << std::endl;
    }
    while ((loopAgain1 || loopAgain2) && max_iter > 0);

    branch_points = dst.clone();
    cv::Mat bpMarkerMask = cv::Mat::zeros(dst.size(), CV_8UC1);
    thinningIteration5(branch_points, 2, false, loopAgain1, bpMarkerMask, nextWhiteX, nextWhiteY, firstWhiteX, firstWhiteY, false);

    dst = dst > 0;
    branch_points = branch_points > 0;
}

