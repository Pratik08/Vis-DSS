/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    A simple shot segmentation algorithm using color histogram
 *
 */
 #ifndef SHOT_DETECTOR_H
 #define SHOT_DETECTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <iostream>

std::vector<int> shotDetector(cv::VideoCapture& capture, double thresh = 0.8, int samplingRate = 1, int compare_method = 0,
   int maxShotDuration = 5);
   /*
        thresh: Threshold for Color Histogram detection to segment shots. If the color histogram similarity between
                consequtive shots is less than a threshold, we break the shots
        samplingRate: Sampling Rate of the algorithm for considering frames for splitting
        compare_method: Comparison method for color histograms
        maxShotDuration: Maximum continuous shot length. If the shot is more than this length, we force cut the shot.
   */
#endif
