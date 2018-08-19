/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    Simple Video Summarizer using Color Histogram
 *
 */

#ifndef SIMPLE_VIDEO_SUMMARIZER_h
#define SIMPLE_VIDEO_SUMMARIZER_h

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

class SimpleVideoSummarizer {
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
    std::vector<cv::Mat> snippetHist;
    std::set<int> summarySet;
    std::vector<double> costList;
    std::vector<std::vector<float> > kernel;
 public:
     // Constructors
    SimpleVideoSummarizer(char* videoFile, int summaryFunction = 0, int segmentType = 0, int snippetLength = 2);
    void extractFeatures(double resizeParam = 0.25);
    void computeKernel(int compare_method = 0, double gamma = 1);
    void summarizeBudget(int budget);
    void summarizeStream(double threshold);
    void summarizeCover(double coverage);
    void playAndSaveSummaryVideo(char* videoFileSave);
    void displayAndSaveSummaryMontage(char* imageFileSave, int image_size);
};
#endif
