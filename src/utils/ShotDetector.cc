/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    A simple shot segmentation algorithm using color histogram
 *
 */

#include "ShotDetector.h"
#include "ColorUtils.h"

std::vector<int> shotDetector(cv::VideoCapture& capture, double thresh, int samplingRate, int compare_method,
                              int maxShotDuration) {
    cv::Mat frame;
    cv::Mat currFrame;
    cv::Mat prevFrame;
    cv::MatND histcurr;
    cv::MatND histprev;
    std::vector<int> shotStartTimes = std::vector<int>();
    int FRAME_RATE = static_cast<int>(capture.get(CV_CAP_PROP_FPS));
    if (!capture.isOpened()) {
        throw "Error when reading steam";
    }
    int frame_count = 0;
    int sampling_count = 0;
     // read the video and compute the histogram at 1 FPS
    int frac = FRAME_RATE / samplingRate;
    shotStartTimes.push_back(0);
    capture.set(CV_CAP_PROP_POS_FRAMES, 0);
    while (true) {
         // FullVideo.push_back(frame.clone());  // Read the video at 10 FPS
        if (frame_count % frac == 0) {
            capture.set(CV_CAP_PROP_POS_FRAMES, frame_count);
            capture >> frame;
            if (frame.empty()) {
                break;
            }
            if (frame_count != 0) {
                prevFrame = currFrame.clone();
            } else {
                prevFrame = frame;
            }
            currFrame = frame.clone();
            colorHist(prevFrame, histprev);
            colorHist(currFrame, histcurr);
            float val = cv::compareHist(histcurr, histprev, compare_method);
            if (val < thresh) {
                shotStartTimes.push_back(sampling_count);
            } else if (sampling_count - shotStartTimes[shotStartTimes.size() - 1] > maxShotDuration) {
                shotStartTimes.push_back(sampling_count);
            }
            sampling_count++;
        }
        frame_count++;
    }
    return shotStartTimes;
}
