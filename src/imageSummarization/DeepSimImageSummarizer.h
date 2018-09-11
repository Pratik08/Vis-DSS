/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    Deep Image Collection Summarizer using Similarity based SFs with Deep Features
 *
 */

#ifndef SRC_ISUMM_DEEPSIMIMAGESUMMARIZER_H_
#define SRC_ISUMM_DEEPSIMIMAGESUMMARIZER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <set>
#include <string>
#include "set.h"
#include "ColorUtils.h"
#include "ImageUtils.h"
#include "ShotDetector.h"
#include "DisparityMin.h"
#include "FacilityLocation.h"
#include "GraphCutFunctions.h"
#include "ProbabilisticSetCover.h"
#include "mmr.h"
#include "SaturateCoverage.h"
#include "naiveGreedyMax.h"
#include "SetCover.h"
#include "streamGreedy.h"
#include "lazyGreedyMax.h"
#include "caffeClassifier.h"

class DeepSimImageSummarizer {
 protected:
    std::vector<cv::Mat> ImageCollection;
    int summaryFunction;  // 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage
    int n;  // ground truth size
    std::vector< std::vector<float> > classifierFeatures;
    std::set<int> summarySet;
    std::vector<double> costList;
    std::vector<std::vector<float> > kernel;
    CaffeClassifier cc;
    std::string featureLayer;
    bool debugMode;

 public:
    DeepSimImageSummarizer(std::vector<cv::Mat>& ImageCollection, CaffeClassifier& cc, std::string featureLayer,
                           int summaryFunction = 0, bool debugMode = true);
    void extractFeatures();
    void computeKernel(int compare_method = 0);
    void summarizeBudget(int budget);
    void summarizeStream(double threshold);
    void summarizeCover(double coverage);
    void playAndSaveSummaryVideo(char* videoFileSave, int frameSize = 500);
    void displayAndSaveSummaryMontage(char* imageFileSave, int image_size);
};
#endif  // SRC_ISUMM_DEEPSIMIMAGESUMMARIZER_H_
