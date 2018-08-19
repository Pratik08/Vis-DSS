/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#ifndef SRC_UTILS_IMAGEUTILS_H_
#define SRC_UTILS_IMAGEUTILS_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <iostream>

// Compute the average color histogram for frames between startframe and endframe
cv::Mat GetSquareImage(const cv::Mat& img, int target_width = 500);
void tile(std::vector<cv::Mat> &src, cv::Mat &dst, int grid_x, int grid_y);
#endif   // SRC_UTILS_IMAGEUTILS_H_
