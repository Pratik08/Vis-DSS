/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    Deep Image Collection Summarizer using Coverage based SFs with Deep Features
 *
 */
#include "DeepCoverImageSummarizer.h"

static std::string IntToString(int a) {
    std::stringstream ss;
    ss << a;
    std::string str = ss.str();
    return str;
}

DeepCoverImageSummarizer::DeepCoverImageSummarizer(std::vector<cv::Mat>& ImageCollection, CaffeClassifier& cc, std::string featureLayer, int summaryFunction, int FeatureBasedFnType, bool debugMode) : ImageCollection(ImageCollection), cc(cc), featureLayer(featureLayer), summaryFunction(summaryFunction), FeatureBasedFnType(FeatureBasedFnType), debugMode(debugMode) {
    n = ImageCollection.size();
    for (int i = 0; i < n; i++) {
        costList.push_back(1);
    }
}

void DeepCoverImageSummarizer::extractFeatures() {
    cv::Mat frame;
    std::map <std::string, int> labelMap;
    std::vector<std::vector<std::string> > labelSegmentList = std::vector<std::vector<std::string> >();
    for (int i = 0; i < n; i++) {
        cv::Mat frame = ImageCollection[i].clone();
        std::vector<float> feat;
        SparseFeature sfeat;
        std::vector<float> p;
        feat = cc.Predict(frame, featureLayer);
        std::cout << "\n";
        sfeat = getSparseFeature(feat);
        p = cc.Predict(frame);
        std::vector<std::pair<std::string, float> > res = cc.Classify(frame);
        std::vector<std::string> currLabels = std::vector<std::string>();
        for (int i = 0; i < res.size(); i++) {
            currLabels.push_back(res[i].first);
            if (labelMap.count(res[i].first) == 0)
                labelMap[res[i].first] = labelMap.size();
        }
        labelSegmentList.push_back(currLabels);
        classifierFeatures.push_back(sfeat);
        pVec.push_back(p);
        if (debugMode) {
            std::vector<std::pair<std::string, float> > res = cc.Classify(frame);
            std::string labels = "";
            for (int i = 0; i < res.size() - 1; i++) {
                labels = labels + res[i].first + ", ";
            }
            labels = labels + res[res.size() - 1].first;
            cv::putText(frame, labels, cvPoint(30, 30),
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
            if (frame.data) {
                cv::imshow("Debug Image", frame);
            }
            char c = static_cast<char>(cv::waitKey(25));
            if (c == 27)
                break;
        }
    }
    coverSet = std::vector<Set>();   // The sets U_i, i \in V
    for (int i = 0; i < labelSegmentList.size(); i++) {
        Set currCoverSet = Set();
        for (int j = 0; j < labelSegmentList[i].size(); j++) {
            currCoverSet.insert(labelMap[labelSegmentList[i][j]]);
        }
        coverSet.push_back(currCoverSet);
    }
}

void DeepCoverImageSummarizer::summarizeBudget(int budget) {
    Set optSet = Set();
    if (summaryFunction == 0) {
        std::vector<double> featureWeights = std::vector<double>(nFeatures, 1);
        for (int i = 0; i < featureWeights.size(); i++) {
            std::cout << featureWeights[i] << " ";
        }
        FeatureBasedFunctions ff(n, FeatureBasedFnType, classifierFeatures, featureWeights);
        lazyGreedyMaxKnapsack(ff, costList, budget, optSet, 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 1) {
        SetCover sc(n, coverSet);
        lazyGreedyMaxKnapsack(sc, costList, budget, optSet, 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 2) {
        ProbabilisticSetCover psc(n, nConcepts, pVec);
        lazyGreedyMaxKnapsack(psc, costList, budget, optSet, 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    }
}

void DeepCoverImageSummarizer::summarizeCover(double coverage) {
    Set optSet;
    if (summaryFunction == 0) {
        std::vector<double> featureWeights = std::vector<double>(nFeatures, 1);
        FeatureBasedFunctions ff(n, FeatureBasedFnType, classifierFeatures, featureWeights);
        lazyGreedyMaxSC(ff, costList, coverage, optSet, 0);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 1) {
        SetCover sc(n, coverSet);
        lazyGreedyMaxSC(sc, costList, coverage, optSet, 0);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 2) {
        ProbabilisticSetCover psc(n, nConcepts, pVec);
        lazyGreedyMaxSC(psc, costList, coverage, optSet, 0);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    }
}

void DeepCoverImageSummarizer::playAndSaveSummaryVideo(char* videoFileSave, int frameSize) {
    cv::VideoWriter videoWriter;
    if (videoFileSave != "")
        videoWriter = cv::VideoWriter(videoFileSave, CV_FOURCC('M', 'J', 'P', 'G'), 1,
                                      cv::Size(frameSize, frameSize));
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
        if (c == 27)
            break;
    }
}

void DeepCoverImageSummarizer::displayAndSaveSummaryMontage(char* imageFileSave, int image_size) {
    int summary_x = ceil(sqrt(summarySet.size()));
    int summary_y = ceil(summarySet.size() / summary_x);
    std::vector<cv::Mat> summaryimages = std::vector<cv::Mat>();
    for (std::set<int>::iterator it = summarySet.begin(); it != summarySet.end(); it++) {
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
