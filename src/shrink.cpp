// Find the shrink of an image by calling the java executable

#include "image_proc.h"

// Fill small holes in an image
void fill_holes(cv::Mat& img, int area_thresh)
{
    // Look for connected components of the white space
    cv::Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(~img, labels, stats, centroids, 4);

    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            uint cc = labels.at<uint>(row, col);
            // Fill in small components
            if (stats.at<uint>(cc, cv::CC_STAT_AREA) < (size_t)area_thresh) {
                img.at<pixel_t>(row, col) = 255;
            }
        }
    }
}

// Read the output of the Java executable, which a text file
// containing the pixel chains
std::vector<std::vector<cv::Point>> read_chains()
{
    std::string chains_file = "chains.txt";

    std::ifstream infile;
    infile.open((MIDWAY_PATH + PATH_SEP + NAME + PATH_SEP + "chains.txt" ));


    if (!infile) {
        std::cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }

    // read in number of chains
    size_t num_chains;
    infile >> num_chains;

    // read in chain sizes
    std::vector<size_t> chain_sizes(num_chains);

    for (size_t i = 0; i < num_chains; ++i) {
        size_t chain_size;
        infile >> chain_size;
        chain_sizes[i] = chain_size;
    }

    // read individual chains
    std::vector<std::vector<cv::Point>> chains;
    int x, y;
    for (size_t chain_size : chain_sizes) {
        std::vector<cv::Point> chain(chain_size);
        for (size_t i = 0; i < chain_size; ++i) {
            infile >> y >> x;
            chain[i] = {x, y};
        }
        chains.push_back(chain);
    }

    infile.close();

    return chains;
}

// Shrink fails when there are white pixels up against the edges of the image.
// Here we make all the border pixels black
void remove_border(cv::Mat& filled_streets)
{
    for (int row = 0; row < filled_streets.rows; ++row) {
        filled_streets.at<pixel_t>(row, 0) = 0;
        filled_streets.at<pixel_t>(row, filled_streets.cols - 1) = 0;
    }
    for (int col = 0; col < filled_streets.cols; ++col) {
        filled_streets.at<pixel_t>(0, col) = 0;
        filled_streets.at<pixel_t>(filled_streets.rows - 1, col) = 0;
    }
}

// shrink streets by calling java executable
std::vector<std::vector<cv::Point>> shrink(cv::Mat& filled_streets, float pThr, float T)
{
    // mixed results when fillinig the holes of the streets.  We should compare results
    // between closing the streets in the previous image and filling holes with this method
    //fill_holes(filled_streets);

    remove_border(filled_streets);

    // draw image so it can be read by java executable
    midway_imwrite(filled_streets, "streets");
    // Path to the image that we feed into the java executable
    std::string input_path = MIDWAY_PATH + PATH_SEP + NAME;

    // arguments for java subprocess:
    // first argument: name of image in midway file, representing streets
    // second argument: pThr represents the pruning constant pruning factor at the end.  Good default value: 4
    // third argument: T represents the contour tolerance.  T=1 means contour must be _very_ accurate.  Good default value: 25

    std::string command = "java -cp src/ImageProc/src/skeletonPruning/bin viewer.EmptyViewer \"" + input_path + "\"" + " " + std::to_string(pThr) + " " + std::to_string(T);

    // Yikes this is not secure
    system(command.c_str());

    return read_chains();

}

// Debug image for drawing pixel chains
void draw_chains(const std::vector<std::vector<cv::Point>> chains, cv::Mat& miniChains, cv::Mat& miniBranchPoints)
{
    for (auto& chain : chains) {
        miniBranchPoints.at<pixel_t>(chain[0]) = 255;
        miniBranchPoints.at<pixel_t>(chain[chain.size() - 1]) = 255;
        for (auto pt : chain) {
            miniChains.at<pixel_t>(pt) = 255;
        }
    }
}

