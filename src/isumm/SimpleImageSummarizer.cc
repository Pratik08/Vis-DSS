/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    Simple Image Collection Summarizer using Color Histogram
 *
 */
#include "SimpleImageSummarizer.h"

std::string IntToString(int a) {
    stringstream ss;
    ss << a;
    string str = ss.str();
    return str;
}

SimpleImageSummarizer::SimpleImageSummarizer(std::vector<cv::Mat>& ImageCollection, int summaryFunction) : ImageCollection(ImageCollection), summaryFunction(summaryFunction) {
    n = ImageCollection.size();
    for (int i = 0; i < n; i++) {
        costList.push_back(1);
    }
}

void SimpleImageSummarizer::extractFeatures(double resizeParam) {
    collectionHist.clear();
    for (int i = 0; i < n; i++) {
        cv::Mat frame = ImageCollection[i].clone();
        cv::Mat hist;
        cv::resize(frame, frame, cv::Size(frame.cols * resizeParam, frame.rows * resizeParam));
        colorHist(frame, hist);
        collectionHist.push_back(hist);
    }
}

void SimpleImageSummarizer::computeKernel(int compare_method, double gamma) {
    // compare_method is the comparision method for histogram similarity
    // gamma is a power of the similarity function: s_{ij} = sim(H_i, H_j)^{\gamma}
    float max = 0;
    for (int i = 0; i < n; i++) {
        std::vector<float> currvector;
        for (int j = 0; j < n; j++) {
            float val = cv::compareHist(collectionHist[i], collectionHist[j], compare_method);
            if (max < val)
                max = val;
            currvector.push_back(val);
        }
        kernel.push_back(currvector);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if ((compare_method == 0) || (compare_method == 2))
                kernel[i][j] = kernel[i][j] / max;
            else if ((compare_method == 1) || (compare_method == 3))
                kernel[i][j] = (max - kernel[i][j]) / max;
            if (compare_method == 0)
                kernel[i][j] = (kernel[i][j] + 1) / 2;
            kernel[i][j] = pow(kernel[i][j], gamma);
        }
    }
}

void SimpleImageSummarizer::summarizeBudget(int budget) {
    Set optSet;
    if (summaryFunction == 0) {
        DisparityMin dM(n, kernel);
        int inititem = 1;  // rand()%n;
        optSet.insert(inititem);
        naiveGreedyMaxKnapsack(dM, costList, budget, optSet, 1, false, true);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 1) {
        MMR m(n, kernel);
        int inititem = rand() % n;
        optSet.insert(inititem);
        naiveGreedyMaxKnapsack(m, costList, budget, optSet, 1, false, true);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 2) {
        FacilityLocation fL(n, kernel);
        lazyGreedyMaxKnapsack(fL, costList, budget, optSet, 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 3) {
        GraphCutFunctions gC(n, kernel, 0.5);
        lazyGreedyMaxKnapsack(gC, costList, budget, optSet, 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 4) {
        SaturateCoverage sC(n, kernel, 0.1);
        lazyGreedyMaxKnapsack(sC, costList, budget, optSet, 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    }
    // cout << "Done with summarization\n" << flush;
}

void SimpleImageSummarizer::summarizeStream(double epsilon) {
    Set optSet;
    if (summaryFunction == 0) {
        DisparityMin dM(n, kernel);
        optSet.insert(0);
        vector<int> order(n, 1);
        for (int i = 0; i < n; i++)
            order[i] = i;
        streamGreedy(dM, epsilon, optSet, order);
        optSet.insert(n - 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 1) {
        MMR m(n, kernel);
        optSet.insert(0);
        vector<int> order(n, 1);
        for (int i = 0; i < n; i++)
            order[i] = i;
        streamGreedy(m, epsilon, optSet, order);
        optSet.insert(n - 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 2) {
        FacilityLocation fL(n, kernel);
        optSet.insert(0);
        vector<int> order(n, 1);
        for (int i = 0; i < n; i++)
            order[i] = i;
        streamGreedy(fL, epsilon, optSet, order);
        optSet.insert(n - 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 3) {
        GraphCutFunctions gC(n, kernel, 0.5);
        optSet.insert(0);
        vector<int> order(n, 1);
        for (int i = 0; i < n; i++)
            order[i] = i;
        streamGreedy(gC, epsilon, optSet, order);
        optSet.insert(n - 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 4) {
        SaturateCoverage sC(n, kernel, 0.1);
        optSet.insert(0);
        vector<int> order(n, 1);
        for (int i = 0; i < n; i++)
            order[i] = i;
        streamGreedy(sC, epsilon, optSet, order);
        optSet.insert(n - 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    }
}

void SimpleImageSummarizer::summarizeCover(double coverage) {
    Set optSet;
    if (summaryFunction == 0) {
        std::cout << "Cover Summarization is not supported for Disparity Min Function\n";
    } else if (summaryFunction == 1) {
        std::cout << "Cover Summarization is not supported for MMR Function\n";
    } else if (summaryFunction == 2) {
        FacilityLocation fL(n, kernel);
        lazyGreedyMaxSC(fL, costList, coverage, optSet, 0);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 3) {
        GraphCutFunctions gC(n, kernel, 0.5);
        lazyGreedyMaxSC(gC, costList, coverage, optSet, 0);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 4) {
        SaturateCoverage sC(n, kernel, 0.1);
        lazyGreedyMaxSC(sC, costList, coverage, optSet, 0);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    }
}

void SimpleImageSummarizer::playAndSaveSummaryVideo(char* videoFileSave, int frameSize) {
    cv::VideoWriter videoWriter;
    if (videoFileSave != "") {
        videoWriter = cv::VideoWriter(videoFileSave, CV_FOURCC('M', 'J', 'P', 'G'), 1, cv::Size(frameSize, frameSize));
    }
    for (std::set<int>::iterator it = summarySet.begin(); it != summarySet.end(); it++) {
        cv::Mat frame = ImageCollection[*it];
        cv::Mat frameSq = GetSquareImage(frame, frameSize);
        cv::putText(frame, "Frame Number: " + IntToString(*it), cvPoint(30, 30),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
        if (frameSq.data) {
            cv::imshow("Summary Video", frameSq);
        }
        if (videoFileSave != "") {
            videoWriter.write(frameSq);
        }
        // Press  ESC on keyboard to exit
        char c = static_cast<char>(cv::waitKey(500));
        if (c == 27) {
            break;
        }
    }
}

void SimpleImageSummarizer::displayAndSaveSummaryMontage(char* imageFileSave, int image_size) {
    int summary_x = ceil(sqrt(summarySet.size()));
    int summary_y = ceil(static_cast<double>(summarySet.size() / summary_x));
    std::vector<cv::Mat> summaryimages = std::vector<cv::Mat>();
    for (set<int>::iterator it = summarySet.begin(); it != summarySet.end(); it++) {
        summaryimages.push_back(ImageCollection[*it]);
    }
    cv::Mat collagesummary = cv::Mat::zeros(cv::Size(image_size * summary_x, image_size * summary_y), CV_8UC3);
    tile(summaryimages, collagesummary, summary_x, summary_y, summaryimages.size());
    cv::imshow("Summary Collage", collagesummary);
    if (imageFileSave != "") {
        cv::imwrite(imageFileSave, collagesummary);
    }
    char c = static_cast<char>(cv::waitKey(0));
    if (c == 27) {
        return;
    }
}
