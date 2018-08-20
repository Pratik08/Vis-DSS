/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    Simple Video Summarizer using Color Histogram
 *
 */

#ifndef SRC_VSUMM_DEEPSIMVIDEOSUMMARIZER_H_
#define SRC_VSUMM_DEEPSIMVIDEOSUMMARIZER_H_

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

class DeepSimVideoSummarizer {
    // protected:
 protected:
    char* videoFile;
    int frameRate;
    int videoLength;  // Length of the video in seconds
    int summaryFunction;
    // 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage
    int segmentType;
    // 0: Fixed Length Segments, 1: Segments based on Shot Detectors
    int snippetLength;  // in case of fixed length snippets, the length of the snippetHist
    std::vector<int> segmentStartTimes;  // start times of the individual segments (each segment is an element in the ground set)
    int n;  // ground truth size
    std::vector< std::vector<float> > classifierFeatures;
    std::set<int> summarySet;
    std::vector<double> costList;
    std::vector<std::vector<float> > kernel;
    CaffeClassifier cc;
    std::string featureLayer;
    double SmallShotPenalty = 10;
    bool debugMode;

 public:
    // Constructors
    DeepSimVideoSummarizer(char* videoFile, CaffeClassifier& cc, std::string featureLayer,
                           int summaryFunction = 0, int segmentType = 0, int snippetLength = 2, bool debugMode = true);
    void extractFeatures();
    void computeKernel(int compare_method = 0);
    void summarizeBudget(int budget);
    void summarizeStream(double threshold);
    void summarizeCover(double coverage);
    void playAndSaveSummaryVideo(char* videoFileSave);
    void displayAndSaveSummaryMontage(char* imageFileSave, int image_size);
};
#endif  // SRC_VSUMM_DEEPSIMVIDEOSUMMARIZER_H_
