/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#ifndef COLOR_UTILS
#define COLOR_UTILS

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <iostream>

// Compute the average color histogram for frames between startframe and endframe
void colorHist(std::vector<cv::Mat>& frames, int startframe, int endframe, cv::MatND& hist);
void colorHist(cv::Mat image, cv::MatND& hist);
#endif // SUMMARIZER
