/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    Simple Video Summarizer using Color Histogram
 *
 */

#ifndef DEEP_COVER_VIDEO_SUMMARIZER_h
#define DEEP_COVER_VIDEO_SUMMARIZER_h

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
#include "FeatureBasedFunctions.h"
#include "SetCover.h"
#include "ProbabilisticSetCover.h"
#include "mmr.h"
#include "SaturateCoverage.h"
#include "naiveGreedyMax.h"
#include "lazyGreedyMax.h"
#include "caffeClassifier.h"
#include "SparseFeature.h"

class DeepCoverVideoSummarizer {
    // protected:
 protected:
   char* videoFile;
   int frameRate;
   int videoLength; // Length of the video in seconds
   int summaryFunction;
   // 0: Feature Based sqrt, 1: Set Cover, 4: Probabilistic Cover
   int FeatureBasedFnType;
   int segmentType;
   // 0: Fixed Length Segments, 1: Segments based on Shot Detectors
   int snippetLength; // in case of fixed length snippets, the length of the snippetHist
   std::vector<int> segmentStartTimes; // start times of the individual segments (each segment is an element in the ground set)
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
   double SmallShotPenalty = 10;
   bool debugMode;

 public:
    // Constructors
    DeepCoverVideoSummarizer(char* videoFile, CaffeClassifier& cc, std::string featureLayer,
      int summaryFunction = 0, int FeatureBasedFnType = 0, int segmentType = 0, int snippetLength = 2, bool debugMode = true);
    void extractFeatures();
    void summarizeBudget(int budget);
    void summarizeCover(double coverage);
    void playAndSaveSummaryVideo(char* videoFileSave);
    void displayAndSaveSummaryMontage(char* imageFileSave, int image_size);
};
#endif
