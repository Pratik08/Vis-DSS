/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    Simple Video Summarizer using Color Histogram
 *
 */

#ifndef DEEP_SIM_VIDEO_SUMMARIZER_h
#define DEEP_SIM_VIDEO_SUMMARIZER_h

#include "set.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <set>
#include "ImageUtils.h"
#include "ColorUtils.h"
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
#include "dnnClassifier.h"
#include "dlibClassifier.h"

class EntitySimVideoSummarizer {
    // protected:
 protected:
   char* videoFile;
   int frameRate;
   int videoLength; // Length of the video in seconds
   int summaryFunction;
   // 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage
   std::vector<cv::Mat> entityVector;
   DNNClassifier dnnc;
   CaffeClassifier cc;
   DlibClassifier dlibc;
   int n; // ground truth size
   std::vector< std::vector<float> > entityFeatures;
   std::set<int> summarySet;
   std::vector<double> costList;
   std::vector<std::vector<float> > kernel;
   std::string featureLayer;
   bool debugMode;
   int featMode;

 public:
    // Constructors
    EntitySimVideoSummarizer(char* videoFile, CaffeClassifier& cc, DNNClassifier& dnnc, std::string featureLayer,
      int summaryFunction = 0, bool debugMode = true);
    EntitySimVideoSummarizer(char* videoFile, DlibClassifier& dlibc, DNNClassifier& dnnc, std::string featureLayer,
        int summaryFunction = 0, bool debugMode = true);
    void extractFeatures();
    void computeKernel(int compare_method = 0);
    void summarizeBudget(int budget);
    void summarizeStream(double threshold);
    void summarizeCover(double coverage);
    void playAndSaveSummaryVideo(char* videoFileSave);
    void displayAndSaveSummaryMontage(char* imageFileSave, int image_size);
};
#endif
