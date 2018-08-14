/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    Deep Image Collection Summarizer using Coverage based SFs with Deep Features
 *
 */

#ifndef DEEP_COVER_IMAGE_SUMMARIZER_h
#define DEEP_COVER_IMAGE_SUMMARIZER_h

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
#include "FeatureBasedFunctions.h"
#include "SetCover.h"
#include "ProbabilisticSetCover.h"
#include "mmr.h"
#include "SaturateCoverage.h"
#include "naiveGreedyMax.h"
#include "lazyGreedyMax.h"
#include "caffeClassifier.h"
#include "SparseFeature.h"

class DeepCoverImageSummarizer {
    // protected:
 protected:
   std::vector<cv::Mat> ImageCollection;
   int summaryFunction;
   // 0: Feature Based sqrt, 1: Set Cover, 4: Probabilistic Cover
   int FeatureBasedFnType;
   int n; // ground truth size
   std::set<int> summarySet;
   std::vector<double> costList;
   std::vector<struct SparseFeature> classifierFeatures;
   std::vector<std::vector<float> > pVec;
   std::vector<Set> coverSet;  // The sets U_i, i \in V
   CaffeClassifier cc;
   std::string featureLayer;
   int nFeatures;
   int nConcepts;
   bool debugMode;

 public:
    // Constructors
    DeepCoverImageSummarizer(std::vector<cv::Mat>& ImageCollection, CaffeClassifier& cc, std::string featureLayer,
      int summaryFunction = 0, int FeatureBasedFnType = 0, bool debugMode = true);
    void extractFeatures();
    void summarizeBudget(int budget);
    void summarizeCover(double coverage);
    void playAndSaveSummaryVideo(char* videoFileSave, int frameSize = 500);
    void displayAndSaveSummaryMontage(char* imageFileSave, int image_size);
};
#endif
