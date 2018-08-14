/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#ifndef IMAGE_UTILS
#define IMAGE_UTILS

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <iostream>

// Compute the average color histogram for frames between startframe and endframe
cv::Mat GetSquareImage( const cv::Mat& img, int target_width = 500);
void tile(std::vector<cv::Mat> &src, cv::Mat &dst, int grid_x, int grid_y);
#endif // SUMMARIZER
