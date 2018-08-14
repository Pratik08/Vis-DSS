#ifndef CAFFE_CLASSIFIER_H
#define CAFFE_CLASSIFIER_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <iosfwd>
#include <string>
#include <vector>
#include <utility>
#include <memory>

#include <caffe/caffe.hpp>

// typedef std::pair<std::string, float> Prediction;

class CaffeClassifier {
 public:
    // CaffeClassifier(const std::string &network_file, const std::string &trained_file);
    CaffeClassifier();
    CaffeClassifier(const std::string& network_file,
        const std::string& trained_file,
        const std::string& mean_file = "",
        const std::string& label_file = "");

    std::vector<std::pair<std::string, float>> Classify(const cv::Mat& img,int N = 2);
    std::vector<float> Predict(const cv::Mat &img, std::string layer = "", const bool &verbose = false);
    std::vector<std::pair<std::string, float>> Classify(const std::vector<cv::Mat>& imgvec,int N = 2);
    std::vector<float> Predict(const std::vector<cv::Mat>& imgvec, std::string layer = "", const bool &verbose = false);
    bool isEmpty();

 private:
    void WrapInputLayer(std::vector<cv::Mat> *input_channels);
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat> *input_channels);
    void SetMean(const std::string& mean_file);
    std::vector<float> getLayerOutput(std::string layer);
    caffe::shared_ptr<caffe::Net<float>> net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::vector<std::string> labels_;
    bool empty = true;
};

#endif  // CAFFE_CLASSIFIER_H
