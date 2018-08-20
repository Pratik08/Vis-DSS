/*
    Copyright 2018 AITOE
 */

#include "dlibClassifier.h"

DlibClassifier::DlibClassifier(const std::string& landmarking_model_file, const std::string& pretrained_resnet_file) {
    dlib::deserialize(landmarking_model_file) >> this->shapePredictor;  // Loading landmarking model to align face
    dlib::deserialize(pretrained_resnet_file) >> this->network;  // Loading Pretrained RESNET Network
    std::cout << "Landmarking and RESNET model loaded for Dlib" << std::endl;
}


DlibClassifier::~DlibClassifier() {
    std::cout << "DlibClassifier destructor called" << std::endl;
}

DlibClassifier::DlibClassifier() {}


dlib::rectangle DlibClassifier::openCvRectToDlib(cv::Rect originalFaceRect) {
    return dlib::rectangle((long)originalFaceRect.tl().x, (long)originalFaceRect.tl().y, (long)originalFaceRect.br().x - 1, (long)originalFaceRect.br().y - 1);
}


std::vector<float> DlibClassifier::Predict(const cv::Mat &img, cv::Rect originalFaceRect) {
    dlib::cv_image<dlib::bgr_pixel> dlibImg(img);  // Converting OpenCV Marix to Dlib Matrix
    this->dlibFaceRect = openCvRectToDlib(originalFaceRect);
    this->shape = this->shapePredictor(dlibImg, this->dlibFaceRect);
    dlib::extract_image_chip(dlibImg, dlib::get_face_chip_details(this->shape, 150, 0.25), this->face_chip);
    this->face_descriptor = network(this->face_chip);
    return dlibMatToVect(this->face_descriptor);
}


std::vector<float> DlibClassifier::Predict(const cv::Mat &img) {
    return (this->Predict(img, cv::Rect(0, 0, img.size().width, img.size().height)));
}


std::vector<float> DlibClassifier::dlibMatToVect(dlib::matrix<float, 0, 1> faceDescriptor) {
    std::vector<float> faceVector(faceDescriptor.begin(), faceDescriptor.end());
    return faceVector;
}
