/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    Deep Image Collection Summarizer using Similarity based SFs with Deep Features
 *
 */

#ifndef DEEP_SIM_IMAGE_SUMMARIZER_h
#define DEEP_SIM_IMAGE_SUMMARIZER_h

#include "set.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <set>
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
    // protected:
 protected:
   std::vector<cv::Mat> ImageCollection;
   int summaryFunction;
   // 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage
   int n; // ground truth size
   std::vector< std::vector<float> > classifierFeatures;
   std::set<int> summarySet;
   std::vector<double> costList;
   std::vector<std::vector<float> > kernel;
   CaffeClassifier cc;
   std::string featureLayer;
   bool debugMode;

 public:
    // Constructors
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
#endif
