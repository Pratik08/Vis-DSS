/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#include "ColorUtils.h"
// Compute the average color histogram for frames between startframe and endframe
void colorHist(std::vector<cv::Mat>& frames, int startframe, int endframe, cv::MatND& hist) {
    int h_bins = 10;
    int s_bins = 10;
    int histSize[] = { h_bins, s_bins };
     // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };
    cv::MatND histcurr;
     /// Calculate the histograms for the HSV images
    cv::calcHist(&frames[startframe], 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    for (int j = startframe + 1; j <= endframe; j++) {
        cv::calcHist(&frames[j], 1, channels, cv::Mat(), histcurr, 2, histSize, ranges, true, false);
        cv::normalize(histcurr, histcurr, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
        hist += histcurr;
    }
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
}

// Compare the color histogram similarity between two images.
void colorHist(cv::Mat image, cv::MatND& hist) {
    int h_bins = 10;
    int s_bins = 10;
    int histSize[] = { h_bins, s_bins };
     // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };
     /// Calculate the histograms for the HSV images
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
}
