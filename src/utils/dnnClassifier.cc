#include "dnnClassifier.h"

DNNClassifier::DNNClassifier() {
    this->empty = true;
}


DNNClassifier::DNNClassifier(const std::string& analyticsType, const std::string& analyticsAlgo, const std::string& network_file, const std::string& trained_file, const std::string& mean_file, const std::string& label_file, float threshold) {
    this->analyticsType = analyticsType;
    this->analyticsAlgo = analyticsAlgo;
    this->dnnThreshold = threshold;
    cv::String modelPrototxt = network_file;
    cv::String modelCaffemodel = trained_file;
    // Initialize network
    this->dnnNet = cv::dnn::readNetFromCaffe(modelPrototxt, modelCaffemodel);
    if (this->dnnNet.empty()) {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << modelPrototxt << std::endl;
        std::cerr << "caffemodel: " << modelCaffemodel << std::endl;
    }

    std::ifstream file(label_file);
    if (file.is_open()) {
        std::string label;
        while (!file.eof()) {
            file >> label;
            this->labels.push_back(label);
        }
    }
}


// Constructor for DARKNET_CPU
DNNClassifier::DNNClassifier(const std::string& analyticsType, const std::string& analyticsAlgo, const std::string& config_file, const std::string& weights_file, const std::string& label_file, float threshold) {
    std::cout << "In constructor for DARKNET_CPU" << std::endl;
    this->analyticsType = analyticsType;
    this->analyticsAlgo = analyticsAlgo;
    this->dnnThreshold = threshold;
    cv::String modelConfiguration = config_file;
    cv::String modelBinary = weights_file;
    // Initialize network
    this->dnnNet = cv::dnn::readNetFromDarknet(modelConfiguration, modelBinary);
    if (this->dnnNet.empty()) {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "Configuration File:   " << modelConfiguration << std::endl;
        std::cerr << "Weights File: " << modelBinary << std::endl;
    }

    std::ifstream file(label_file);
    if (file.is_open()) {
        std::string label;
        while (!file.eof()) {
            file >> label;
            this->labels.push_back(label);
        }
    }
}


void DNNClassifier::Classify(const cv::Mat &rawFrame, std::vector<std::pair<std::string, cv::Rect> > &results) {
    cv::Mat detectionMat = this->Predict(rawFrame);

    if (this->analyticsType == "OBJECT") {
        if (this->analyticsAlgo == "MOBILENET") {
            for (int i = 0; i < detectionMat.rows; i++) {
                float confidence = detectionMat.at<float>(i, 2);
                if (confidence > this->dnnThreshold) {
                    std::pair<std::string, cv::Rect> result;
                    size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));
                    int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * rawFrame.cols);
                    int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * rawFrame.rows);
                    int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * rawFrame.cols);
                    int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * rawFrame.rows);
                    cv::Rect bb((int)xLeftBottom, (int)yLeftBottom,
                                (int)(xRightTop - xLeftBottom),
                                (int)(yRightTop - yLeftBottom));
                    result.first = cv::String(this->labels[objectClass]);
                    result.second = bb;
                    results.push_back(result);
                }
            }
        } else if (this->analyticsAlgo == "DARKNET_CPU") {
            std::ostringstream ss;
            cv::Mat annotatedFrame;
            for (int i = 0; i < detectionMat.rows; i++) {
                // std::cout << "Number of rows in the detectionMat is : " << detectionMat.rows << std::endl;
                const int probability_index = 5;
                const int probability_size = detectionMat.cols - probability_index;
                float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
                size_t objectClass = (size_t)(std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr);
                float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
                if (confidence > this->dnnThreshold) {
                    std::pair<std::string, cv::Rect> result;
                    float x = detectionMat.at<float>(i, 0);
                    float y = detectionMat.at<float>(i, 1);
                    float width = detectionMat.at<float>(i, 2);
                    float height = detectionMat.at<float>(i, 3);
                    int xLeftBottom = static_cast<int>((x - width / 2) * rawFrame.cols);
                    int yLeftBottom = static_cast<int>((y - height / 2) * rawFrame.rows);
                    int xRightTop = static_cast<int>((x + width / 2) * rawFrame.cols);
                    int yRightTop = static_cast<int>((y + height / 2) * rawFrame.rows);
                    cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
                                    (int)(xRightTop - xLeftBottom),
                                    (int)(yRightTop - yLeftBottom));
                    result.first = cv::String(this->labels[objectClass]);
                    result.second = object;
                    results.push_back(result);
                }
            }
        }
    } else if (this->analyticsType == "FACE") {
        if (this->analyticsAlgo == "RESNET") {
            for (int i = 0; i < detectionMat.rows; i++) {
                float confidence = detectionMat.at<float>(i, 2);
                if (confidence > this->dnnThreshold) {
                    std::pair<std::string, cv::Rect> result;
                    size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));
                    int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * rawFrame.cols);
                    int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * rawFrame.rows);
                    int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * rawFrame.cols);
                    int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * rawFrame.rows);
                    cv::Rect bb((int)xLeftBottom, (int)yLeftBottom,
                                (int)(xRightTop - xLeftBottom),
                                (int)(yRightTop - yLeftBottom));
                    result.first = cv::String("face");  // cv::String(this->labels[objectClass]);
                    result.second = bb;
                    results.push_back(result);
                }
            }
        }
    }
}


cv::Mat DNNClassifier::Predict(const cv::Mat &rawFrame) {
    // Preprocess frame
    cv::Mat inputBlob = this->Preprocess(rawFrame);

    if (this->analyticsType == "OBJECT") {
        if (this->analyticsAlgo == "MOBILENET") {
            // Set input blob
            this->dnnNet.setInput(inputBlob, "data");  // set the network input
            cv::Mat detection = this->dnnNet.forward("detection_out");  // compute output
            cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
            return detectionMat;
        } else if (this->analyticsAlgo == "DARKNET_CPU") {
            // Set input blob
            this->dnnNet.setInput(inputBlob, "data");  // set the network input
            // Make forward pass
            cv::Mat detectionMat = this->dnnNet.forward("detection_out");  // compute output
            return detectionMat;
        }
    } else if (this->analyticsType == "FACE") {
        if (this->analyticsAlgo == "RESNET") {
            // Set input blob
            this->dnnNet.setInput(inputBlob);  // set the network input
            cv::Mat detection = this->dnnNet.forward();  // compute output
            cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
            return detectionMat;
        }
    }
}


cv::Mat DNNClassifier::Preprocess(const cv::Mat &rawFrame) {
    if (this->analyticsType == "OBJECT") {
        if (this->analyticsAlgo == "MOBILENET") {
            const size_t inWidth = 300;
            const size_t inHeight = 300;
            const float WHRatio = inWidth / (float)inHeight;
            const float inScaleFactor = 0.007843f;
            const float meanVal = 127.5;

            if (rawFrame.channels() == 4) {
                cvtColor(rawFrame, rawFrame, cv::COLOR_BGRA2BGR);
            }

            // Prepare blob
            cv::Mat inputBlob = cv::dnn::blobFromImage(rawFrame, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false);  // Convert Mat to batch of images
            return inputBlob;
        } else if (this->analyticsAlgo == "DARKNET_CPU") {
            if (rawFrame.channels() == 4) {
                cvtColor(rawFrame, rawFrame, cv::COLOR_BGRA2BGR);
            }

            cv::Mat inputBlob = cv::dnn::blobFromImage(rawFrame, 1 / 255.F, cv::Size(416, 416), cv::Scalar(), true, false);  // Convert Mat to batch of images
            return inputBlob;
        }
    } else if (this->analyticsType == "FACE") {
        if (this->analyticsAlgo == "RESNET") {
            // const size_t inWidth = 300;
            // const size_t inHeight = 300;
            const float inScaleFactor = 1.0f;
            //
            // cv::Mat resizedFrame;
            // cv::resize(rawFrame, resizedFrame, cv::Size(300, 300));
            //
            if (rawFrame.channels() == 4) {
                cvtColor(rawFrame, rawFrame, cv::COLOR_BGRA2BGR);
            }
            //
            //// Prepare blob
            cv::Mat inputBlob = cv::dnn::blobFromImage(rawFrame, inScaleFactor, cv::Size(rawFrame.cols, rawFrame.rows), cv::Scalar(104.0, 177.0, 123.0));  // Convert Mat to batch of images
            return inputBlob;
        }
    }
}


bool DNNClassifier::isEmpty() {
    return this->empty;
}
