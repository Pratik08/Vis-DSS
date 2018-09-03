/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    Deep Image Collection Summarizer using Similarity based SFs with Deep Features
 *
 */
#include "DeepSimImageSummarizer.h"

static std::string IntToString(int a) {
    std::stringstream ss;
    ss << a;
    std::string str = ss.str();
    return str;
}

float DotProduct(std::vector<float> vec1, std::vector<float> vec2) {
    if (vec1.size() != vec2.size()) {
        std::cout << "Error: Both vectors need to be of the same size\n";
    }
    float sim = 0;
    float norm1 = 0;
    float norm2 = 0;
    int n = vec1.size();
    for (int i = 0; i < n; i++) {
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
        for (int j = 0; j < n; j++) {
            sim += vec1[i] * vec2[j];
        }
    }
    return sim / (sqrt(norm1) * sqrt(norm2));
}

float GaussianSimilarity(std::vector<float> vec1, std::vector<float> vec2) {
    if (vec1.size() != vec2.size()) {
        std::cout << "Error: Both vectors need to be of the same size\n";
    }
    float diff = 0;
    float norm1 = 0;
    float norm2 = 0;
    int n = vec1.size();
    for (int i = 0; i < n; i++) {
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < vec2.size(); j++) {
            diff += vec1[i] / sqrt(norm1) - vec2[j] / sqrt(norm2);
        }
    }
    return exp(-diff / 2);
}

DeepSimImageSummarizer::DeepSimImageSummarizer(std::vector<cv::Mat>& ImageCollection, CaffeClassifier& cc, std::string featureLayer, int summaryFunction, bool debugMode) : ImageCollection(ImageCollection), cc(cc), featureLayer(featureLayer), summaryFunction(summaryFunction), debugMode(debugMode) {
    n = ImageCollection.size();
    for (int i = 0; i < n; i++) {
        costList.push_back(1);
    }
}

void DeepSimImageSummarizer::extractFeatures() {
    classifierFeatures.clear();
    for (int i = 0; i < n; i++) {
        cv::Mat frame = ImageCollection[i].clone();
        std::vector<float> feat = cc.Predict(frame, featureLayer);
        classifierFeatures.push_back(feat);
        if (debugMode) {
            std::vector<std::pair<std::string, float> > res = cc.Classify(frame);
            std::string labels = "";
            for (int i = 0; i < res.size() - 1; i++) {
                labels = labels + res[i].first + ", ";
            }
            labels = labels + res[res.size() - 1].first;
            cv::putText(frame, labels, cvPoint(30, 30),
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
            cv::imshow("Debug Image", frame);
            // Press  ESC on keyboard to exit
            char c = static_cast<char>(cv::waitKey(25));
            if (c == 27) {
                break;
            }
        }
    }
}

void DeepSimImageSummarizer::computeKernel(int compare_method) {
    // compare_method is the comparision method for similarity (0: DotProduct, 1:GaussianSimilarity)
    float max = 0;
    for (int i = 0; i < n; i++) {
        std::vector<float> currvector;
        for (int j = 0; j < n; j++) {
            float val;
            if (compare_method == 0) {
                val = DotProduct(classifierFeatures[i], classifierFeatures[j]);
            } else {
                val = GaussianSimilarity(classifierFeatures[i], classifierFeatures[j]);
            }
            currvector.push_back(val);
        }
        kernel.push_back(currvector);
    }
}

void DeepSimImageSummarizer::summarizeBudget(int budget) {
    std::cout << "Begin summarization\n" << std::flush;
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
    for (std::set<int>::iterator it = summarySet.begin(); it != summarySet.end(); it++) {
        std::cout << *it << "\n" << std::flush;
    }
    std::cout << "Summarization is done with n = " << n << " and budget = " << budget << "\n" << std::flush;
}

void DeepSimImageSummarizer::summarizeStream(double epsilon) {
    Set optSet;
    if (summaryFunction == 0) {
        DisparityMin dM(n, kernel);
        optSet.insert(0);
        std::vector<int> order(n, 1);
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        streamGreedy(dM, epsilon, optSet, order);
        optSet.insert(n - 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 1) {
        MMR m(n, kernel);
        optSet.insert(0);
        std::vector<int> order(n, 1);
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        streamGreedy(m, epsilon, optSet, order);
        optSet.insert(n - 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 2) {
        FacilityLocation fL(n, kernel);
        optSet.insert(0);
        std::vector<int> order(n, 1);
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        streamGreedy(fL, epsilon, optSet, order);
        optSet.insert(n - 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 3) {
        GraphCutFunctions gC(n, kernel, 0.5);
        optSet.insert(0);
        std::vector<int> order(n, 1);
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        streamGreedy(gC, epsilon, optSet, order);
        optSet.insert(n - 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 4) {
        SaturateCoverage sC(n, kernel, 0.1);
        optSet.insert(0);
        std::vector<int> order(n, 1);
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        streamGreedy(sC, epsilon, optSet, order);
        optSet.insert(n - 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    }
}

void DeepSimImageSummarizer::summarizeCover(double coverage) {
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

void DeepSimImageSummarizer::playAndSaveSummaryVideo(char* videoFileSave, int frameSize) {
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

void DeepSimImageSummarizer::displayAndSaveSummaryMontage(char* imageFileSave, int image_size) {
    int summary_x = ceil(sqrt(summarySet.size()));
    int summary_y = ceil(static_cast<double>(summarySet.size() / summary_x));
    std::vector<cv::Mat> summaryimages = std::vector<cv::Mat>();
    for (std::set<int>::iterator it = summarySet.begin(); it != summarySet.end(); it++) {
        summaryimages.push_back(ImageCollection[*it]);
    }
    std::cout << summaryimages.size() << "\n";
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
