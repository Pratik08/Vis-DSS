/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    Simple Image Collection Summarizer using Color Histogram
 *
 */

#ifndef SIMPLE_IMAGE_SUMMARIZER_h
#define SIMPLE_IMAGE_SUMMARIZER_h

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
#include "math.h"

class SimpleImageSummarizer {
    // protected:
 protected:
   std::vector<cv::Mat> ImageCollection;
   int summaryFunction;
   // 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage
   int n; // ground truth size
   std::set<int> summarySet;
   std::vector<double> costList;
   std::vector<std::vector<float> > kernel;
   std::vector<cv::Mat> collectionHist;
 public:
    // Constructors
    SimpleImageSummarizer(std::vector<cv::Mat>& ImageCollection, int summaryFunction = 0);
    void extractFeatures(double resizeParam = 0.25);
    void computeKernel(int compare_method = 0, double gamma = 1);
    void summarizeBudget(int budget);
    void summarizeStream(double threshold);
    void summarizeCover(double coverage);
    void playAndSaveSummaryVideo(char* videoFileSave, int frameSize = 500);
    void displayAndSaveSummaryMontage(char* imageFileSave, int image_size);
};
#endif
