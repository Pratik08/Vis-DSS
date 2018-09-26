/*
    Copyright to be added.
 *
 */

#ifndef SRC_ISUMM_QUERYSIMIMAGESUMMARIZER_H_
#define SRC_ISUMM_QUERYSIMIMAGESUMMARIZER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <set>
#include <string>
#include "set.h"
#include "ImageUtils.h"
#include "ColorUtils.h"
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

class QuerySimImageSummarizer {
 protected:
    std::vector<cv::Mat> imageCollection;
    int summaryFunction;  // 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage
    int n;  // ground truth size
    std::vector<std::pair<std::set<std::string>, std::vector<float> > > classifiedImage;
    std::vector<cv::Mat> queryImages;
    std::vector<std::vector<float> > queryFeatures;
    std::set<int> summarySet;
    std::vector<double> costList;
    std::vector<std::vector<float> > kernel;
    CaffeClassifier ccLabel;
    CaffeClassifier ccFeature;
    std::string featureLayer;
    bool debugMode;

 public:
    QuerySimImageSummarizer(std::vector<cv::Mat>& imageCollection, CaffeClassifier & ccLabel, CaffeClassifier & ccFeature, std::string featureLayer, int summaryFunction = 0, bool debugMode = true);
    void extractFeatures();
    void processQuery(std::string queryInput);
    void computeKernel(int compareMethod = 0);
    void summarizeBudget(int budget);
    void summarizeStream(double threshold);
    void summarizeCover(double coverage);
    void playAndSaveSummaryVideo(char* videoFileSave);
    void displayAndSaveSummaryMontage(char* imageFileSave, int image_size);
};
#endif  // SRC_ISUMM_QUERYSIMIMAGESUMMARIZER_H_
