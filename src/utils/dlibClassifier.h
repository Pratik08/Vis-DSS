/*
    Copyright 2018 AITOE
 */

#ifndef DLIB_CLASSIFIER_H
#define DLIB_CLASSIFIER_H

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "dlib/dnn.h"
#include "dlib/opencv.h"
#include "dlib/string.h"
#include "dlib/image_io.h"
#include "dlib/image_processing/frontal_face_detector.h"

// ----------------------------------------------------------------------------------------

// This block of statements defines the resnet network

template<
    template <int, template<typename>class, int, typename>
    class block, int N, template<typename>
    class BN, typename SUBNET
    >
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET> > >;

template<
    template<int, template<typename>class, int, typename >
    class block, int N, template<typename>
    class BN, typename SUBNET
    >
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET> > > > > >;

template<
    int N, template<typename> class BN, int stride, typename SUBNET
    >
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET> > > > >;

template <int N, typename SUBNET>
using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET> >;

template <int N, typename SUBNET>
using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET> >;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET> > >;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET> > >;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET> > > >;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET> > >;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2, dlib::input_rgb_image_sized<150> > > > > > > > > > > > >;

// ----------------------------------------------------------------------------------------

class DlibClassifier {
 public:
    DlibClassifier(const std::string& landmarking_model_file, const std::string& pretrained_resnet_file);
    DlibClassifier();
    ~DlibClassifier();
    dlib::rectangle openCvRectToDlib(cv::Rect originalFaceRect);
    std::vector<float> Predict(const cv::Mat &img, cv::Rect originalFaceRect);
    std::vector<float> Predict(const cv::Mat &img);
    std::vector<float> dlibMatToVect(dlib::matrix<float, 0, 1> faceDescriptor);
 private:
    dlib::shape_predictor shapePredictor;
    anet_type network;
    dlib::rectangle dlibFaceRect;
    dlib::full_object_detection shape;
    dlib::matrix<dlib::rgb_pixel>face_chip;
    dlib::matrix<float, 0, 1> face_descriptor;
};
#endif  // DLIB_CLASSIFIER_H
