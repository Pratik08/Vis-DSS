#ifndef DNN_CLASSIFIER_H
#define DNN_CLASSIFIER_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/dnn.hpp"
#include "opencv2/dnn/shape_utils.hpp"

class DNNClassifier {
 public:
    DNNClassifier();
    DNNClassifier(const std::string& analyticsType, const std::string& analyticsAlgo, const std::string& network_file, const std::string& trained_file, const std::string& mean_file = "", const std::string& label_file = "", float threshold = 0.3);
    DNNClassifier(const std::string& analyticsType, const std::string& analyticsAlgo, const std::string& config_file, const std::string& weights_file, const std::string& label_file = "", float threshold = 0.2);
    void Classify(const cv::Mat &rawFrame, std::vector<std::pair<std::string, cv::Rect> > &results);
    bool isEmpty();
 private:
    cv::Mat Preprocess(const cv::Mat &rawFrame);
    cv::Mat Predict(const cv::Mat &rawFrame);
    std::vector<std::string> labels;
    float dnnThreshold;
    cv::dnn::Net dnnNet;
    bool empty = true;
    std::string analyticsType;
    std::string analyticsAlgo;
};

#endif  // DNN_CLASSIFIER_H
