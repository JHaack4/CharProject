
// Jordan's machine learning project. Please ignore
// remember to comment out the include here, and the line in main.

#include <stdlib.h>
#include <queue>
#include <windows.h>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "image_proc.h"

std::string fileName = "C:\\Users\\Jordan Haack\\Desktop\\projectML\\rawdigits\\";
std::string fileNum = "C:\\Users\\Jordan Haack\\Desktop\\projectML\\rawnumbers\\";
std::string fileOut = "C:\\Users\\Jordan Haack\\Desktop\\projectML\\outdigits\\";

void trainTest(int seed, std::vector<std::vector<std::string>>& digitPaths,
    std::vector<std::vector<int>>& train_index,
    std::vector<std::vector<int>>& test_index,
    std::vector<std::vector<std::vector<MyLine>>>& train_vec,
    std::vector<std::vector<std::vector<MyLine>>>& test_vec, int trainSize, int testSize) {

    srand(seed);

    for (int dig = 0; dig < 10; ++dig) {

        std::vector<int> left(0,0);
        for (int i = 0; i < (int)(digitPaths[dig].size()); ++i)
            left.push_back(i);

        for (int i = 0; i < testSize; ++i) {
            int r = std::abs<int>(rand()) % (int)(left.size());
            test_index[dig].push_back(left[r]);
            left.erase(left.begin() + r);
            //std::cout << dig << " " << test_index[dig][i] << std::endl;
        }

        for (int i = 0; i < trainSize; ++i) {
            int r = std::abs<int>(rand()) % (int)(left.size());
            train_index[dig].push_back(left[r]);
            left.erase(left.begin() + r);
            //std::cout << dig << " " << train_index[dig][i] << std::endl;
        }

        for (int i = 0; i < trainSize; ++i) {
            std::vector<MyLine> lines;
            int idx = train_index[dig][i];
            std::ifstream inputFile;
            inputFile.open((fileOut + "vec" + PATH_SEP + digitPaths[dig][idx] + "vec.txt"));
            std::string line;
            std::getline(inputFile, line);
            int rows = atoi(line.c_str());
            std::getline(inputFile, line);
            int cols = atoi(line.c_str());
            //std::cout << rows << " " << cols << std::endl;
            while (std::getline(inputFile, line))
            {
                int ef1 = atoi(line.c_str());
                std::getline(inputFile, line);
                int ef2 = atoi(line.c_str());
                std::getline(inputFile, line);
                int es1 = atoi(line.c_str());
                std::getline(inputFile, line);
                int es2 = atoi(line.c_str());

                cv::Point2f p1 {(float)ef1, (float)ef2};
                cv::Point2f p2 {(float)es1, (float)es2};
                p1 = p1 * 16.0 / rows;
                p2 = p2 * 16.0 / rows;
                p1.x = p1.x + (int)((16 - 16.0*cols/rows)/2);
                p2.x = p2.x + (int)((16 - 16.0*cols/rows)/2);
                MyLine l (p1, p2, false, false);

                lines.push_back(l);
            }
            inputFile.close();

            train_vec[dig].push_back( lines );

            cv::Mat zeros16 = cv::Mat(cv::Size(16,16), CV_8U, 0.0);
            for (size_t j = 0; j < lines.size(); ++j) {
                MyLine tline = lines[j];
                std::pair<cv::Point, cv::Point> endpts = tline.endpoints();
                cv::Point p1 = endpts.first;
                cv::Point p2 = endpts.second;
                cv::line(zeros16, p1, p2, CV_RGB(255,255,255),1);
            }
            //cv::imwrite((fileOut + "train" + PATH_SEP + std::to_string(dig) + "_" + std::to_string(i) + "_" + digitPaths[dig][idx]), zeros16);
        }

        for (int i = 0; i < testSize; ++i) {
            std::vector<MyLine> lines;
            int idx = test_index[dig][i];
            std::ifstream inputFile;
            inputFile.open((fileOut + "vec" + PATH_SEP + digitPaths[dig][idx] + "vec.txt"));
            std::string line;
            std::getline(inputFile, line);
            int rows = atoi(line.c_str());
            std::getline(inputFile, line);
            int cols = atoi(line.c_str());
            //std::cout << rows << " " << cols << std::endl;
            while (std::getline(inputFile, line))
            {
                int ef1 = atoi(line.c_str());
                std::getline(inputFile, line);
                int ef2 = atoi(line.c_str());
                std::getline(inputFile, line);
                int es1 = atoi(line.c_str());
                std::getline(inputFile, line);
                int es2 = atoi(line.c_str());

                cv::Point2f p1 {(float)ef1, (float)ef2};
                cv::Point2f p2 {(float)es1, (float)es2};
                p1 = p1 * 16.0 / rows;
                p2 = p2 * 16.0 / rows;
                p1.x = p1.x + (int)((16 - 16.0*cols/rows)/2);
                p2.x = p2.x + (int)((16 - 16.0*cols/rows)/2);
                MyLine l (p1, p2, false, false);

                lines.push_back(l);
            }
            inputFile.close();

            test_vec[dig].push_back( lines );

            cv::Mat zeros16 = cv::Mat(cv::Size(16,16), CV_8U, 0.0);
            for (size_t j = 0; j < lines.size(); ++j) {
                MyLine tline = lines[j];
                std::pair<cv::Point, cv::Point> endpts = tline.endpoints();
                cv::Point p1 = endpts.first;
                cv::Point p2 = endpts.second;
                cv::line(zeros16, p1, p2, CV_RGB(255,255,255),1);
            }
            //cv::imwrite((fileOut + "test" + PATH_SEP + std::to_string(dig) + "_" + std::to_string(i) + "_" + digitPaths[dig][idx]), zeros16);
        }

    }

}

//int trainSize = 100;
//int testSize = 10;

double distanceMatch(MyLine& l, MyLine& m, float angleThresh, float distThresh, float distWeight) {
    Segment s1 = l.segment();
    Segment s2 = m.segment();

    if (cv::norm(s1.p1 - s1.p2) <= 0.01) return 1;
    if (cv::norm(s2.p1 - s2.p2) <= 0.01) return 1;

    float angle = segments_angle(s1, s2);
    float tdist = segments_distance(s1, s2);
    float tproj = worst_point_segment_horizontal_proj(s1, s2)
                                    + worst_point_segment_horizontal_proj(s2, s1);

    if (tproj > 0.2 || angle > angleThresh || tdist > distThresh) return 1;

    float angle_score = angle / angleThresh;
    float dist_score = tdist / distThresh;
    //if (dist_score > 0.5) dist_score = 0.5;
    return dist_score * distWeight + angle_score * (1-distWeight);
}

double distanceMatch(std::vector<MyLine>& l1, MyLine& l2, double thresh, float angleThresh, float distThresh, float distWeight) {

    int n1 = l1.size();

    for (int i = 0; i < n1; ++i) {
            double tdist = distanceMatch(l1[i], l2, angleThresh, distThresh, distWeight);
            if (tdist < thresh) return 1;

    }
    return 0;
}

double entropy(std::vector<int> counts) {

    int uniqueCnt = 0;
    int uniqueIdx = 0;
    double entropy = 0.0;

    int totalCnt = 0;
    for (int i = 0; i < 10; ++i) {
        totalCnt += counts[i];
        if (counts[i]>0) {
            uniqueCnt++;
            uniqueIdx = i;
        }
    }

    if (uniqueCnt == 0) return -11;
    if (uniqueCnt == 1) return -uniqueIdx - 1;

    for (int i = 0; i < 10; ++i) {
        double p = counts[i] * 1.0 / totalCnt;
        if (p>0)
            entropy += p * log2(p);
    }

    return 0-entropy;
}

int mostCommon(std::vector<int> counts) {

    int uniqueCnt = 0;
    int uniqueIdx = 0;

    for (int i = 0; i < 10; ++i) {
        if (counts[i] > uniqueCnt) {
            uniqueCnt = counts[i];
            uniqueIdx = i;
        }
    }

    return uniqueIdx;
}

class DecisionTree;

class DecisionTree {
public:

    int max_depth;
    int num_gen;

    std::vector<MyLine> lines;
    std::vector<double> threshs;
    std::vector<double> angle_threshs;
    std::vector<double> dist_threshs;
    std::vector<double> dist_weights;

    std::vector<int> left_idxs; // match
    std::vector<int> right_idxs; // no match
    std::vector<int> depths;
    std::vector<int> guesses;

    void trainrec(std::vector<std::vector<std::vector<MyLine>>>& train_vec , int cur_depth) {

        std::vector<int> cnts (10, 0);

        // check if empty

        int totalCnt = 0;
        for (int i = 0; i < 10; ++i) {
            cnts[i] = train_vec[i].size();
            totalCnt += train_vec[i].size();
        }

        if (totalCnt == 0) {
            lines.push_back(MyLine());
            threshs.push_back(-1);
            angle_threshs.push_back(-1);
            dist_threshs.push_back(-1);
            dist_weights.push_back(-1);
            left_idxs.push_back(-1);
            right_idxs.push_back(-1);
            depths.push_back(cur_depth);
            guesses.push_back(0);
            return;
        }

        // check entropy
        double starten = entropy(cnts);

        if (starten < 0 || cur_depth > max_depth) {
                int pred = mostCommon(cnts);

            lines.push_back(MyLine());
            threshs.push_back(-1);
            angle_threshs.push_back(-1);
            dist_threshs.push_back(-1);
            dist_weights.push_back(-1);
            left_idxs.push_back(-1);
            right_idxs.push_back(-1);
            depths.push_back(cur_depth);
            guesses.push_back(pred);
            return;

        }


        bool noGood = true;

        double best_entropy = 1e10;
        MyLine best_line;
        double best_thresh, best_angle_thresh, best_dist_thresh, best_dist_weight;

        std::vector<int> best_countsM, best_countsN;

        for (int i = 0; i < num_gen; ++i) {
            int x1 = std::abs<int>(rand()) % 17;
            int x2 = std::abs<int>(rand()) % 17;
            int y1 = std::abs<int>(rand()) % 17;
            int y2 = std::abs<int>(rand()) % 17;

            cv::Point e1{x1,y1};
            cv::Point e2{x2,y2};


            if (e1 == e2 || cv::norm(e1-e2) > 7) {
                //--i;
                //continue;
                int randdig = std::abs<int>(rand()) % 10;
                if (train_vec[randdig].size() == 0) {
                    --i;
                    continue;
                }
                int randnum = std::abs<int>(rand()) % ((int)(train_vec[randdig].size()));
                if (train_vec[randdig][randnum].size() == 0) {
                    --i;
                    continue;
                }
                int randseg = std::abs<int>(rand()) % ((int)(train_vec[randdig][randnum].size()));
                MyLine mm = train_vec[randdig][randnum][randseg];
                std::pair<cv::Point, cv::Point> endpt = mm.endpoints();
                e1 = endpt.first;
                e2 = endpt.second;
                if (cv::norm(e1-e2) <= 0.1) {
                    --i;
                    continue;
                }
            }

            MyLine l{e1, e2, true, false};

            double thresh = (std::abs<int>(rand()) % 1000) * 0.99 / 1000;
            double angle_thresh = (std::abs<int>(rand()) % 1000) * 45.0 / 1000;
            double dist_thresh = (std::abs<int>(rand()) % 1000) * 8.0 / 1000;
            double dist_weight = (std::abs<int>(rand()) % 1000) * 0.8 / 1000 + 0.10;

            std::vector<int> countMatch (10, 0);
            std::vector<int> countNoMatch (10, 0);
            int totMatch = 0;
            int totNoMatch = 0;

            for (int dig = 0; dig < 10; dig++) {
                for (int j = 0; j < (int)(train_vec[dig].size()); ++j) {
                    double distM = distanceMatch(train_vec[dig][j], l, thresh, angle_thresh, dist_thresh, dist_weight);

                    if (distM > 0) {
                        totMatch++;
                        countMatch[dig]++;
                    }
                    else {
                        totNoMatch++;
                        countNoMatch[dig]++;
                    }
                }
            }

            double en1 = entropy(countMatch);
            double en2 = entropy(countNoMatch);

            if (en1<0) en1=0;
            if (en2<0) en2=0;

            double en = en1 * totMatch * 1.0 / (totMatch + totNoMatch) + en2 * totNoMatch * 1.0 / (totMatch + totNoMatch);

            if (totMatch != 0 && totNoMatch != 0) {
                noGood = false;
            }

            /*std::pair<cv::Point, cv::Point> endpt = l.endpoints();
            std::cout << "try" << endpt.first << " "
                << endpt.second << " t"
                << thresh << " at"
                << angle_thresh << " dt"
                << dist_thresh << " dw"
                << dist_weight <<std::endl ;
            std::cout << "en" << en;
            for (int q = 0; q < 10; ++q) {
                std::cout  << " " << q << ": " << countMatch[q] << "," << countNoMatch[q];
            }
            std::cout << std::endl;*/

            if (en < best_entropy) {
                best_angle_thresh = angle_thresh;
                best_dist_thresh = dist_thresh;
                best_entropy = en;
                best_line = l;
                best_dist_weight = dist_weight;
                best_thresh = thresh;
                best_countsM = countMatch;
                best_countsN = countNoMatch;
            }

        }

        if (noGood) {
                int pred = mostCommon(cnts);

            lines.push_back(MyLine());
            threshs.push_back(-1);
            angle_threshs.push_back(-1);
            dist_threshs.push_back(-1);
            dist_weights.push_back(-1);
            left_idxs.push_back(-1);
            right_idxs.push_back(-1);
            depths.push_back(cur_depth);
            guesses.push_back(pred);
            return;

        }

        // train recursively

        std::vector<std::vector<std::vector<MyLine>>> train_vecM (10, std::vector<std::vector<MyLine>>(0, std::vector<MyLine>(0, MyLine())));
        std::vector<std::vector<std::vector<MyLine>>> train_vecN (10, std::vector<std::vector<MyLine>>(0, std::vector<MyLine>(0, MyLine())));

        for (int dig = 0; dig < 10; dig++) {
                for (int j = 0; j < (int)(train_vec[dig].size()); ++j) {
                    double distM = distanceMatch(train_vec[dig][j], best_line, best_thresh, best_angle_thresh, best_dist_thresh, best_dist_weight);

                    if (distM > 0) {
                        train_vecM[dig].push_back(train_vec[dig][j]);
                        //std::cout << "   " << dig << " " << j << " match" << std::endl;
                    }
                    else {
                        train_vecN[dig].push_back(train_vec[dig][j]);
                        //std::cout << "   " << dig << " " << j << " no match" << std::endl;
                    }
                }
        }

        int my_loc = lines.size();
        lines.push_back(best_line);
        threshs.push_back(best_thresh);
        angle_threshs.push_back(best_angle_thresh);
        dist_threshs.push_back(best_dist_thresh);
        dist_weights.push_back(best_dist_weight);
        depths.push_back(cur_depth);
        guesses.push_back(-1);
        left_idxs.push_back(-2);
        right_idxs.push_back(-2);
        {
            int i = my_loc;
            std::pair<cv::Point, cv::Point> endpt = lines[i].endpoints();
            std::cout << i <<  ": g" << guesses[i] << " l" << endpt.first << " "
                << endpt.second << " t"
                << threshs[i] << " at"
                << angle_threshs[i] << " dt"
                << dist_threshs[i] << " dw"
                << dist_weights[i] << " l"
                << left_idxs[i] << " r"
                << right_idxs[i] << " d"
                << depths[i] <<std::endl ;

            std::cout << "---en" << best_entropy;
            for (int q = 0; q < 10; ++q) {
                std::cout  << " " << q << ": " << best_countsM[q] << "," << best_countsN[q];
            }
            std::cout << std::endl;
        }


        int left_child_loc = lines.size();
        trainrec(train_vecM, cur_depth + 1);

        int right_child_loc = lines.size();
        trainrec(train_vecN, cur_depth + 1);

        left_idxs[my_loc] = left_child_loc;
        right_idxs[my_loc] = right_child_loc;

        return;


    }

    void train(std::vector<std::vector<std::vector<MyLine>>>& train_vec) {
        trainrec(train_vec, 0);
    }

    int predict(std::vector<MyLine>& l) {

        int idx = 0;

        while(true) {
            int g = guesses[idx];
            //std::cout << idx << " ";

            if (g < 0) {

                double m = distanceMatch(l, lines[idx], threshs[idx], angle_threshs[idx], dist_threshs[idx], dist_weights[idx]);

                if (m > 0) {
                    idx = left_idxs[idx];

                } else {
                    idx = right_idxs[idx];
                }

            } else {
                 //std::cout << " g:" << g << std::endl;
                return g;

            }
        }

    }

    void printInfo () {
        std::cout << "Decision tree" << std::endl;
        for (int i = 0; i < (int)(lines.size()); ++i) {
            std::pair<cv::Point, cv::Point> endpt = lines[i].endpoints();
            std::cout << i <<  ": g" << guesses[i] << " l" << endpt.first << " "
                << endpt.second << " t"
                << threshs[i] << " at"
                << angle_threshs[i] << " dt"
                << dist_threshs[i] << " dw"
                << dist_weights[i] << " l"
                << left_idxs[i] << " r"
                << right_idxs[i] << " d"
                << depths[i] <<std::endl ;

            cv::Mat zeros16 = cv::Mat(cv::Size(16,16), CV_8U, 0.0);

            MyLine tline = lines[i];
            std::pair<cv::Point, cv::Point> endpts = tline.endpoints();
            cv::Point p1 = endpts.first;
            cv::Point p2 = endpts.second;
            cv::line(zeros16, p1, p2, CV_RGB(255,255,255),1);
            //cv::imwrite((fileOut + "dt" + PATH_SEP + std::to_string(i) + "_" + std::to_string(depths[i]) + "_" + std::to_string(guesses[i]) + ".png"), zeros16);

        }


    }


};

double dt(int max_depth, int num_gen, int trainSize, int testSize, std::vector<std::vector<std::string>>& digitPaths) {
    std::cout << "dt, depth=" << max_depth << std::endl;

    int correct = 0;
    int total = 0;

    std::vector<std::vector<int>> confusion (10, std::vector<int>(10, 0));

    std::vector<std::vector<int>> train_index (10, std::vector<int>(0,0));
    std::vector<std::vector<int>> test_index (10, std::vector<int>(0,0));


    std::vector<std::vector<std::vector<MyLine>>> train_vec (10, std::vector<std::vector<MyLine>>(0, std::vector<MyLine>(0, MyLine())));
    std::vector<std::vector<std::vector<MyLine>>> test_vec (10, std::vector<std::vector<MyLine>>(0, std::vector<MyLine>(0, MyLine())));

    trainTest(1234, digitPaths, train_index, test_index, train_vec, test_vec, trainSize, testSize);

    //int k = 3;

    DecisionTree dt;
    dt.max_depth = max_depth;
    dt.num_gen = num_gen;

    dt.train(train_vec);

    //dt.printInfo();

    for (int dig = 0; dig < 10; dig++) {
            std::cout << dig << std::endl;

        for (int i = 0; i < testSize; ++i) {

            std::vector<MyLine>& l = test_vec[dig][i];


            int pred = dt.predict(l);


            confusion[dig][pred] ++;
            correct += (dig == pred);
            total++;

        }

    }

    std::cout << correct << " out of " << testSize*10 << std::endl;

    //for (int i = 0; i < 10; ++i) {
    //    std::cout << i << ": " ;
    //    for (int j = 0; j < 10; ++j) {
    //        std::cout << confusion[i][j] << " " ;
    //    }
    //    std::cout << std::endl;
    //}

    return correct * 1.0 / total;
}

double rforest(int num_trees, int max_depth, int num_gen, int trainSize, int testSize, std::vector<std::vector<std::string>>& digitPaths) {
    std::cout << "rf, depth=" << max_depth << std::endl;

    int correct = 0;
    int total = 0;

    std::vector<std::vector<int>> confusion (10, std::vector<int>(10, 0));

    std::vector<std::vector<int>> train_index (10, std::vector<int>(0,0));
    std::vector<std::vector<int>> test_index (10, std::vector<int>(0,0));


    std::vector<std::vector<std::vector<MyLine>>> train_vec (10, std::vector<std::vector<MyLine>>(0, std::vector<MyLine>(0, MyLine())));
    std::vector<std::vector<std::vector<MyLine>>> test_vec (10, std::vector<std::vector<MyLine>>(0, std::vector<MyLine>(0, MyLine())));

    trainTest(1234, digitPaths, train_index, test_index, train_vec, test_vec, trainSize, testSize);

    //int k = 3;

    std::vector<DecisionTree> dts;

    for (int q = 0; q < num_trees; ++q) {
            std::cout<< "training" << q << std::endl;
        DecisionTree dt;
        dt.max_depth = max_depth;
        dt.num_gen = num_gen;

        std::vector<std::vector<std::vector<MyLine>>> train_vec2 (10, std::vector<std::vector<MyLine>>(0, std::vector<MyLine>(0, MyLine())));

        for (int dig = 0; dig < 10; ++dig) {
        for (int k = 0; k < (int)(train_vec[dig].size()); ++k) {
            int r = std::abs<int>(rand()) % 8;
            if (r == 0) {
                train_vec2[dig].push_back(train_vec[dig][k]);
            }
        }
        }

        std::cout<< "copied train" << q << std::endl;

        dt.train(train_vec2);

        dts.push_back(dt);
    }





    //dt.printInfo();

    for (int dig = 0; dig < 10; dig++) {
            std::cout << dig << std::endl;

        for (int i = 0; i < testSize; ++i) {

            std::vector<MyLine>& l = test_vec[dig][i];


            std::vector<int> counts(10, 0);

            for(int q= 0; q < num_trees; ++q) {
                int pred = dts[q].predict(l);
                counts[pred]++;
            }

            int pred = mostCommon(counts);


            confusion[dig][pred] ++;
            correct += (dig == pred);
            total++;

            std::string img_path_in = fileOut + "img16" + PATH_SEP + digitPaths[dig][test_index[dig][i]];
            cv::Mat inimg = cv::imread(img_path_in, CV_LOAD_IMAGE_GRAYSCALE);
            cv::resize(inimg, inimg, cv::Size(0,0), 4, 4, cv::INTER_NEAREST);
            //v::GaussianBlur(inimg, inimg, cv::Size(3,3),0,0);
            //cv::blur(img, img, cv::Size(2,2));
            //cv::blur(img, img, cv::Size(2,2));
            //cv::blur(img, img, cv::Size(3,3));
            cv::threshold(inimg, inimg, 70, 255, CV_8UC1);
            fill_holes(inimg, 8);
            cv::Mat bgr;
            package_bgr({inimg, inimg, inimg}, bgr);

            //cv::Mat zeros16 = cv::Mat(cv::Size(16,16), CV_8U, 0.0);
            for (size_t j = 0; j < l.size(); ++j) {
                MyLine tline = l[j];
                std::pair<cv::Point, cv::Point> endpts = tline.endpoints();
                cv::Point p1 = endpts.first;
                cv::Point p2 = endpts.second;
                p1 *= 4;
                p2 *= 4;
                p1.x += 2;
                p1.y += 2;
                p2.x += 2;
                p2.y += 2;
                cv::line(bgr, p1, p2, CV_RGB(255,120,0),2);
            }

            if (dig == pred) {
                cv::imwrite((fileOut + "correct" + PATH_SEP + std::to_string(dig) + "_" + digitPaths[dig][i]), bgr);

            } else {
                cv::imwrite((fileOut + "mistake" + PATH_SEP + std::to_string(dig) + "a" + std::to_string(pred) + "p_" + digitPaths[dig][i]), bgr);
            }

        }

    }

    std::cout << correct << " out of " << testSize*10 << std::endl;

    for (int i = 0; i < 10; ++i) {
        std::cout << i << ": " ;
        for (int j = 0; j < 10; ++j) {
            std::cout << confusion[i][j] << " " ;
        }
        std::cout << std::endl;
    }

    return correct * 1.0 / total;
}



double distance(MyLine& l, MyLine& m) {
    Segment s1 = l.segment();
    Segment s2 = m.segment();

    if (cv::norm(s1.p1 - s1.p2) == 0) return 1;
    if (cv::norm(s2.p1 - s2.p2) == 0) return 1;

    float angle = segments_angle(s1, s2);
    float tdist = segments_distance(s1, s2);
    float tproj = worst_point_segment_horizontal_proj(s1, s2)
                                    + worst_point_segment_horizontal_proj(s2, s1);

    if (tproj > 0.2 || angle > 24 || tdist > 3) return 1;

    float angle_score = angle / 48.0;
    float dist_score = tdist / 6.0;
    if (dist_score > 0.5) dist_score = 0.5;
    return dist_score + angle_score;
}
int a = 0;
double distance(std::vector<MyLine>& l1, std::vector<MyLine>& l2) {

    int n1 = l1.size();
    int n2 = l2.size();

    std::vector<std::vector<double>> dists (n1, std::vector<double>(n2, 0.0));

    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            dists[i][j] = 0.5 * (distance(l1[i], l2[j]) + distance(l2[j], l1[i]));
        }
    }

    std::vector<double> lengths1 (n1, 0.0);
    std::vector<double> lengths2 (n2, 0.0);

    double totalLen1 = 0.0;
    double totalLen2 = 0.0;

    for (int i = 0; i < n1; ++i) {
        Segment s = l1[i].segment();
        double len = cv::norm(s.p1 - s.p2);
        lengths1[i] = len;
        totalLen1 += len;
    }

    for (int i = 0; i < n2; ++i) {
        Segment s = l2[i].segment();
        double len = cv::norm(s.p1 - s.p2);
        lengths2[i] = len;
        totalLen2 += len;
    }

    double score1 = 0;
    double score2 = 0;

    cv::Mat zeros16 = cv::Mat(cv::Size(64,64), CV_8UC1, 0.0);
    cv::Mat rgb;
    package_bgr({zeros16, zeros16, zeros16}, rgb);

    for (int i = 0; i < n1; ++i) {
        double minDist = 1;
        for (int j = 0; j < n2; ++j) {
            minDist = std::min<double>(minDist, dists[i][j]);
        }
        score1 += lengths1[i] * minDist;

        MyLine tline = l1[i];
        std::pair<cv::Point, cv::Point> endpts = tline.endpoints();
        cv::Point p1 = endpts.first;
        cv::Point p2 = endpts.second;
        p1*=4;
        p2*=4;
        cv::line(rgb, p1, p2, CV_RGB(0,255,(int)(255.0*minDist)),3);
    }
    score1 /= totalLen1;

    for (int i = 0; i < n2; ++i) {
        double minDist = 1;
        for (int j = 0; j < n1; ++j) {
            minDist = std::min<double>(minDist, dists[j][i]);
        }
        score2 += lengths2[i] * minDist;

        MyLine tline = l2[i];
        std::pair<cv::Point, cv::Point> endpts = tline.endpoints();
        cv::Point p1 = endpts.first;
        cv::Point p2 = endpts.second;
        p1*=4;
        p2*=4;
        cv::line(rgb, p1, p2, CV_RGB(255,0,(int)(255.0*minDist)),2);
    }
    score2 /= totalLen2;

    double score = (score1+score2)/2.0;
    std::cout << "here" << std::endl;
    cv::putText(rgb, std::to_string(score).substr(0,5), {0,60}, 0, 0.3, CV_RGB(255,255,255));
    cv::imwrite((fileOut + "distinfo" + PATH_SEP + std::to_string(a) + ".png"), rgb);
    std::cout << "here2" << std::endl;
    a++;

    return score;
}

double knn(int k, int trainSize, int testSize, std::vector<std::vector<std::string>>& digitPaths) {
    std::cout << "k nearest neighbors" << std::endl;

    int correct = 0;
    int total = 0;

    std::vector<std::vector<int>> confusion (10, std::vector<int>(10, 0));

    std::vector<std::vector<int>> train_index (10, std::vector<int>(0,0));
    std::vector<std::vector<int>> test_index (10, std::vector<int>(0,0));


    std::vector<std::vector<std::vector<MyLine>>> train_vec (10, std::vector<std::vector<MyLine>>(0, std::vector<MyLine>(0, MyLine())));
    std::vector<std::vector<std::vector<MyLine>>> test_vec (10, std::vector<std::vector<MyLine>>(0, std::vector<MyLine>(0, MyLine())));

    trainTest(1234, digitPaths, train_index, test_index, train_vec, test_vec, trainSize, testSize);

    //int k = 3;

    for (int dig = 0; dig < 10; dig++) {
            std::cout << dig << std::endl;

        for (int i = 0; i < testSize; ++i) {

            std::vector<MyLine>& l = test_vec[dig][i];


            auto cmp = [](std::pair<double, int> left, std::pair<double, int> right) { return (left.first) > (right.first);};
            std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, decltype(cmp)> q3(cmp);

            for (int dig2 = 0; dig2 < 10; dig2++) {
                for (int j = 0; j < trainSize; ++j) {

                    std::vector<MyLine>& m = train_vec[dig2][j];
                    std::pair<double, int> thisd = {distance(l, m), dig2};

                    q3.push(thisd);

                }
            }

            //std::cout << "here" << std::endl;
            //while (!q3.empty()) {
            //    std::pair<double, int> thisd = q3.top();
            //    std::cout << thisd.first << " " << thisd.second << std::endl;
            //    q3.pop();
            //}
            //continue;

            std::vector<double> counts;
            for (int j = 0; j < 10; ++j)
                counts.push_back(0.0);

            for (int j = 0; j < k; ++j) {
                std::pair<double, int> thisd = q3.top();
                counts[thisd.second] += 1 - thisd.first / k / 10.0;
                q3.pop();
                //std::cout << thisd.second << std::endl;
            }
            //for (double q: counts)
            //    std::cout << q << " ";
            //std::cout << std::endl;

            double maxCnt = -1;
            int pred = -1;
            for (int j = 0; j < 10; ++j) {
                if (counts[j] > maxCnt) {
                    pred = j;
                    maxCnt = counts[j];
                }
            }


            confusion[dig][pred] ++;
            correct += (dig == pred);
            total++;
            //std::cout << dig << " pred " << pred << std::endl;

        }

    }

    std::cout << correct << " out of " << testSize*10 << std::endl;

    //for (int i = 0; i < 10; ++i) {
    //    std::cout << i << ": " ;
    //    for (int j = 0; j < 10; ++j) {
    //        std::cout << confusion[i][j] << " " ;
    //    }
    //    std::cout << std::endl;
    //}

    return correct * 1.0 / total;
}

std::vector<std::string> glob(std::string pat){
    WIN32_FIND_DATA data;
    HANDLE hFind = FindFirstFile(pat.c_str(), &data);      // DIRECTORY
    std::vector<std::string> ret;

    if ( hFind != INVALID_HANDLE_VALUE ) {
        do {
            ret.push_back( data.cFileName );
        } while (FindNextFile(hFind, &data));
        FindClose(hFind);
    }
    return ret;
}

void run_word_project()
{
    std::cout << "Begin word processing" << std::endl;

    if (0)
    {
        std::string path = fileNum + PATH_SEP + "*.png";
        std::vector<std::string> numPaths = glob(path);

        for (size_t i = 0; i < 5 && i < numPaths.size(); ++i) {
            std::cout << numPaths[i] << std::endl;

            std::string img_path_in = fileNum + numPaths[i];
            cv::Mat img = cv::imread(img_path_in, CV_LOAD_IMAGE_GRAYSCALE);
            cv::GaussianBlur(img, img, cv::Size(5,5), 1, 1);
            cv::resize(img, img, cv::Size(0,0), 5, 5, cv::INTER_LINEAR);
            cv::GaussianBlur(img, img, cv::Size(9,9),0,0);
            //cv::blur(img, img, cv::Size(2,2));
            //cv::blur(img, img, cv::Size(2,2));
            //cv::blur(img, img, cv::Size(3,3));
            cv::threshold(img, img, 70, 255, CV_8UC1);
            fill_holes(img, 10);

            std::vector<std::vector<cv::Point>> line_chains = shrink(img, 1.2, 0.1);

            cv::Mat zeros = cv::Mat(img.size(), img.type(), 0.0);
            cv::Mat vecImg = cv::Mat(img.size(), img.type(), 0.0);
            cv::Mat miniChains = cv::Mat(img.size(), img.type(), 0.0);
            cv::Mat miniBranchPoints = cv::Mat(img.size(), img.type(), 0.0);
            draw_chains(line_chains, miniChains, miniBranchPoints);
            cv::Mat bgrMini;
            cv::imwrite((fileOut + numPaths[i] + "copy.png"), img);
            package_bgr({img & ~miniBranchPoints, miniChains & ~miniBranchPoints, miniBranchPoints}, bgrMini);
            cv::imwrite((fileOut + numPaths[i] + "skel.png"), bgrMini);
            package_bgr({zeros, zeros, zeros}, vecImg);

            std::vector<MyLine> smVecLines;
            std::vector<MyVertex> smVecVerts;
            std::vector<std::vector<cv::Point>> new_chains2;
            segment_lines(line_chains, new_chains2, smVecLines, smVecVerts, /*ht*/3, /*len*/3, /*curl*/0);

            for (size_t j = 0; j < smVecLines.size(); ++j) {
                MyLine street = smVecLines[j];
                std::pair<cv::Point, cv::Point> endpts = street.endpoints();
                cv::Point pm {(int) street.mid_.x, (int) street.mid_.y};
                cv::line(bgrMini, endpts.first, endpts.second, CV_RGB(255,0,100),2);
                cv::line(vecImg, endpts.first, endpts.second, CV_RGB(255,0,100),2);
            }
            //log("Testing vectorization of small lines", LogLevel::debug);

            cv::imwrite((fileOut + numPaths[i] + "info.png"), bgrMini);
            cv::imwrite((fileOut + numPaths[i] + "vec.png"), vecImg);


        }
    }

    // Vectorize all of the digits, write each to a file
    /*for (int dig = 0; dig < 10; ++dig) {

        std::string path = fileName + std::to_string(dig) + PATH_SEP + "*.png";
        std::vector<std::string> digitPaths = glob(path);

        for (size_t i = 0; i < digitPaths.size(); ++i) {
            std::cout << digitPaths[i] << std::endl;
            std::cout << dig << " " << i << " " << digitPaths.size() << std::endl;

            std::string img_path_in = fileName + std::to_string(dig) + PATH_SEP + digitPaths[i];
            cv::Mat imgraw = cv::imread(img_path_in, CV_LOAD_IMAGE_GRAYSCALE);
            cv::Mat img;
            cv::GaussianBlur(imgraw, img, cv::Size(5,5), 1, 1);
            cv::resize(img, img, cv::Size(0,0), 5, 5, cv::INTER_LINEAR);
            cv::GaussianBlur(img, img, cv::Size(9,9),0,0);
            //cv::blur(img, img, cv::Size(2,2));
            //cv::blur(img, img, cv::Size(2,2));
            //cv::blur(img, img, cv::Size(3,3));
            cv::threshold(img, img, 70, 255, CV_8UC1);
            fill_holes(img, 10);

            std::vector<std::vector<cv::Point>> line_chains = shrink(img, 1.1, 2); // best 1.2, 10

            cv::Mat zeros = cv::Mat(img.size(), img.type(), 0.0);
            cv::Mat vecImg = cv::Mat(img.size(), img.type(), 0.0);
            cv::Mat miniChains = cv::Mat(img.size(), img.type(), 0.0);
            cv::Mat miniBranchPoints = cv::Mat(img.size(), img.type(), 0.0);
            draw_chains(line_chains, miniChains, miniBranchPoints);
            cv::Mat bgrMini;
            cv::imwrite((fileOut + "copy" + PATH_SEP + digitPaths[i]), img);
            package_bgr({img & ~miniBranchPoints, miniChains & ~miniBranchPoints, miniBranchPoints}, bgrMini);
            cv::imwrite((fileOut + "skel" + PATH_SEP + digitPaths[i]), bgrMini);
            package_bgr({zeros, zeros, zeros}, vecImg);

            std::vector<MyLine> smVecLines;
            std::vector<MyVertex> smVecVerts;
            segment_lines(line_chains, smVecLines, smVecVerts, 3, 3, 0); // ht len curl

            std::ofstream outputFile;
            outputFile.open((fileOut + "vec" + PATH_SEP + digitPaths[i] + "vec.txt"));

            outputFile << img.rows << std::endl;
            outputFile << img.cols << std::endl;

            for (size_t j = 0; j < smVecLines.size(); ++j) {
                MyLine line = smVecLines[j];
                std::pair<cv::Point, cv::Point> endpts = line.endpoints();
                cv::Point pm {(int) line.mid_.x, (int) line.mid_.y};
                cv::line(bgrMini, endpts.first, endpts.second, CV_RGB(255,0,100),2);
                cv::line(vecImg, endpts.first, endpts.second, CV_RGB(255,0,100),2);
                outputFile << endpts.first.x << std::endl;
                outputFile << endpts.first.y << std::endl;
                outputFile << endpts.second.x << std::endl;
                outputFile << endpts.second.y << std::endl;
            }
            outputFile.close();
            //log("Testing vectorization of small lines", LogLevel::debug);

            cv::imwrite((fileOut + "info" + PATH_SEP + digitPaths[i]), bgrMini);
            cv::imwrite((fileOut + "vecimg" + PATH_SEP + digitPaths[i] ), vecImg);


            // convert to 16x16
            cv::resize(imgraw, imgraw, cv::Size((int)(16.0*imgraw.cols/imgraw.rows),16), 0, 0, cv::INTER_LINEAR);
            if (imgraw.cols < 16) {
                int pad = (int)((16.0-imgraw.cols)/2.0);
                cv::copyMakeBorder(imgraw, imgraw, 0, 0, pad, 16-imgraw.cols-pad, cv::BORDER_CONSTANT, 0);
            }
            if (imgraw.cols > 16) {
                cv::Rect rect {(imgraw.cols - 16)/2, 0, 16, 16};
                imgraw = imgraw(rect);
                std::cout << "here" << std::endl;
            }
            cv::threshold(imgraw, imgraw, 70, 255, CV_8UC1);
            cv::imwrite((fileOut + "img16" + PATH_SEP + digitPaths[i]), imgraw);

            cv::Mat zeros16 = cv::Mat(cv::Size(16,16), img.type(), 0.0);
            for (size_t j = 0; j < smVecLines.size(); ++j) {
                MyLine line = smVecLines[j];
                std::pair<cv::Point, cv::Point> endpts = line.endpoints();
                cv::Point p1 = endpts.first;
                cv::Point p2 = endpts.second;
                p1 = p1 * 16.0 / img.rows;
                p2 = p2 * 16.0 / img.rows;
                p1.x = p1.x + (int)((16 - 16.0*img.cols/img.rows)/2);
                p2.x = p2.x + (int)((16 - 16.0*img.cols/img.rows)/2);
                cv::line(zeros16, p1, p2, CV_RGB(255,255,255),1);
            }
            cv::imwrite((fileOut + "vec16" + PATH_SEP + digitPaths[i]), zeros16);

        }


    }*/

    std::cout << "digit paths" << std::endl;

    std::vector<std::vector<std::string>> digitPaths (10, std::vector<std::string>(0, ""));

    for (int dig = 0; dig < 10; ++dig) {
        std::string path = fileName + std::to_string(dig) + PATH_SEP + "*.png";
        digitPaths[dig] = glob(path);
    }

    //knn(9, 10, 5, digitPaths);
    //return;

    /*int testSize = 100;
    //int k = 9;
    std::vector<int> train_sizes = {10, 20, 30, 50, 75, 100, 200, 500, 1000, 2000};
    std::vector<int> ks = {1,3,5,7,9,11,15,21,31,41,51,101};

    std::vector<double> accuracies;

    for (int kk : ks) {
        accuracies.push_back(knn(kk, 200, testSize, digitPaths));
    }

    std::cout << "k, accuracy" << std::endl;
    for (int i = 0; i < (int)(ks.size()); ++i) {
        std::cout << ks[i] << "\t" << accuracies[i] << std::endl;
    }*/

    //int max_depth = 13;
    //std::vector<int> depths = { 3, 4, 6, 8, 10, 12, 14, 16,18, 20 ,25};
    //int num_gen = 40;
    //std::vector<int> num_gens = {750, 1000};
    //int trainSize = 1500;
    //std::vector<int> trainSizes = {100, 200, 300, 500, 750, 1000, 1500, 2000};
    //int testSize = 200;
    //std::vector<double> accuracies;

    //for (int ts: num_gens) {

        //double acc = dt(max_depth, ts, trainSize, testSize, digitPaths);
       // accuracies.push_back(acc);
    //}
    //std::cout << "k, accuracy" << std::endl;
    //for (int i = 0; i < (int)(num_gens.size()); ++i) {
    //    std::cout << num_gens[i] << "\t" << accuracies[i] << std::endl;
    //}

    rforest(150, 13, 100, 2500, 790, digitPaths);
    //rforest(5, 13, 10, 10, 20, digitPaths);

    /*
    std::cout << "distance test" << std::endl;
    //std::vector<std::vector<std::vector<MyLine>>> trainVec;
    //std::vector<std::vector<std::vector<MyLine>>> test_index;

    for (int dig = 0; dig < 10; ++dig) {

        for (int i = 0; i < trainSize; ++i) {
            for (int j = 0; j < trainSize; ++j) {
                std::cout << dig << " " << i << " " << j << " " << distance(train_vec[dig][i], train_vec[dig][j]) << std::endl;
            }
        }

    }

    std::cout << "diff -------------" << std::endl;

    for (int dig1 = 0; dig1 < 10; ++dig1) {
      for (int dig2 = 0; dig2 < 10; ++dig2) {

        for (int i = 0; i < trainSize; ++i) {
            for (int j = 0; j < trainSize; ++j) {
                std::cout << dig1 << " " << dig2 << " "
                    << i << " " << j << " " << distance(train_vec[dig1][i], train_vec[dig2][j]) << std::endl;
            }
        }
      }

    }*/



    return;
}

