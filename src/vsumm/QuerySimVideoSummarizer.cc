/*
    Copyright to be added.
 *
 */

#include "QuerySimVideoSummarizer.h"

static std::string IntToString(int a) {
    std::stringstream ss;
    ss << a;
    std::string str = ss.str();
    return str;
}

float DotProduct(std::vector<float> vec1, std::vector<float> vec2) {
    if (vec1.size() != vec2.size()) {
        std::cout << "Error: Both vectors need to be of the same size\n";
    }
    float sim = 0;
    float norm1 = 0;
    float norm2 = 0;
    int n = vec1.size();
    for (int i = 0; i < n; i++) {
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
        for (int j = 0; j < n; j++) {
            sim += vec1[i] * vec2[j];
        }
    }
    return sim / (sqrt(norm1) * sqrt(norm2));
}

float GaussianSimilarity(std::vector<float> vec1, std::vector<float> vec2) {
    if (vec1.size() != vec2.size()) {
        std::cout << "Error: Both vectors need to be of the same size\n";
    }
    float diff = 0;
    float norm1 = 0;
    float norm2 = 0;
    int n = vec1.size();
    for (int i = 0; i < n; i++) {
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < vec2.size(); j++) {
            diff += vec1[i] / sqrt(norm1) - vec2[j] / sqrt(norm2);
        }
    }
    return exp(-diff / 2);
}

QuerySimVideoSummarizer::QuerySimVideoSummarizer(char* videoFile, CaffeClassifier& cc, std::string featureLayer, int summaryFunction, bool debugMode) : videoFile(videoFile), cc(cc), featureLayer(featureLayer), summaryFunction(summaryFunction), debugMode(debugMode) {
    cv::VideoCapture capture(videoFile);
    frameRate = static_cast<int>(capture.get(CV_CAP_PROP_FPS));
    videoLength = capture.get(CV_CAP_PROP_FRAME_COUNT) / frameRate;
    capture.release();
    featMode = 0;
}

void QuerySimVideoSummarizer::extractFeatures() {
    classifiedImageVector = std::vector<cv::Mat>();
    classifiedFeatureVector = std::vector<std::pair<std::string, std::vector<float> > >();
    cv::VideoCapture capture(videoFile);
    frameRate = static_cast<int>(capture.get(CV_CAP_PROP_FPS));
    cv::Mat frame;
    int count = 0;
    if ( !capture.isOpened() ) {
        throw "Error when reading video file\n";
    }
    while (1) {
        capture >> frame;
        if (frame.empty()) {
            break;
        }
        if (count % frameRate == 0) {
            std::vector<std::pair<std::string, float> > framePredictions = std::vector<std::pair<std::string, float> >();
            std::vector<float> frameFeature = std::vector<float> ();
            framePredictions = cc.Classify(frame);
            frameFeature = cc.Predict(frame, featureLayer);
            classifiedImageVector.push_back(frame);
            classifiedFeatureVector.push_back(std::make_pair(framePredictions[0].first, frameFeature));
            if (debugMode) {
                cv::imshow("Debug", frame);
                cv::waitKey(300);  // key press to close window
            }
        }
        count++;
    }
}

void QuerySimVideoSummarizer::processQuery(std::string queryInput) {
    queryVector = std::vector<cv::Mat>();
    queryFeatures = std::vector<std::vector<float> >();
    for (int i = 0; i < classifiedFeatureVector.size(); i++) {
        if (classifiedFeatureVector[i].first == queryInput) {
            queryVector.push_back(classifiedImageVector[i]);
            queryFeatures.push_back(classifiedFeatureVector[i].second);
            costList.push_back(1);
        }
    }
}

void QuerySimVideoSummarizer::computeKernel(int compareMethod) {
    // compareMethod is the comparision method for similarity (0: DotProduct, 1:GaussianSimilarity)
    float max = 0;
    std::cout << "Started Compute Kernel\n";
    kernel = std::vector<std::vector <float> >();
    for (int i = 0; i < n; i++) {
        std::vector<float> currvector = std::vector<float>();
        for (int j = 0; j < n; j++) {
            float val;
            if (compareMethod == 0) {
                val = DotProduct(queryFeatures[i], queryFeatures[j]);
            } else {
                val = GaussianSimilarity(queryFeatures[i], queryFeatures[j]);
            }
            currvector.push_back(val);
        }
        kernel.push_back(currvector);
    }
    std::cout << "Done Compute Kernel\n";
}

void QuerySimVideoSummarizer::summarizeBudget(int budget) {
    Set optSet;
    if (summaryFunction == 0) {
        DisparityMin dM(n, kernel);
        int inititem = 1;  // rand()%n;
        optSet.insert(inititem);
        std::cout << "About to start naivegreedy max\n" << std::flush;
        naiveGreedyMaxKnapsack(dM, costList, budget, optSet, 1, false, true);
        std::cout << "Finished naivegreedy max\n" << std::flush;
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 1) {
        MMR m(n, kernel);
        int inititem = rand() % n;
        optSet.insert(inititem);
        naiveGreedyMaxKnapsack(m, costList, budget, optSet, 1, false, true);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 2) {
        FacilityLocation fL(n, kernel);
        lazyGreedyMaxKnapsack(fL, costList, budget, optSet, 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 3) {
        GraphCutFunctions gC(n, kernel, 0.5);
        lazyGreedyMaxKnapsack(gC, costList, budget, optSet, 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 4) {
        SaturateCoverage sC(n, kernel, 0.1);
        lazyGreedyMaxKnapsack(sC, costList, budget, optSet, 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    }
    // cout << "Done with summarization\n" << flush;
}

void QuerySimVideoSummarizer::summarizeStream(double epsilon) {
    Set optSet;
    if (summaryFunction == 0) {
        DisparityMin dM(n, kernel);
        optSet.insert(0);
        std::vector<int> order(n, 1);
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        streamGreedy(dM, epsilon, optSet, order);
        optSet.insert(n - 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 1) {
        MMR m(n, kernel);
        optSet.insert(0);
        std::vector<int> order(n, 1);
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        streamGreedy(m, epsilon, optSet, order);
        optSet.insert(n - 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 2) {
        FacilityLocation fL(n, kernel);
        optSet.insert(0);
        std::vector<int> order(n, 1);
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        streamGreedy(fL, epsilon, optSet, order);
        optSet.insert(n - 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 3) {
        GraphCutFunctions gC(n, kernel, 0.5);
        optSet.insert(0);
        std::vector<int> order(n, 1);
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        streamGreedy(gC, epsilon, optSet, order);
        optSet.insert(n - 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 4) {
        SaturateCoverage sC(n, kernel, 0.1);
        optSet.insert(0);
        std::vector<int> order(n, 1);
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        streamGreedy(sC, epsilon, optSet, order);
        optSet.insert(n - 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    }
}

void QuerySimVideoSummarizer::summarizeCover(double coverage) {
    Set optSet;
    if (summaryFunction == 0) {
        std::cout << "Cover Summarization is not supported for Disparity Min Function\n";
    } else if (summaryFunction == 1) {
        std::cout << "Cover Summarization is not supported for MMR Function\n";
    } else if (summaryFunction == 2) {
        FacilityLocation fL(n, kernel);
        lazyGreedyMaxSC(fL, costList, coverage, optSet, 0);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 3) {
        GraphCutFunctions gC(n, kernel, 0.5);
        lazyGreedyMaxSC(gC, costList, coverage, optSet, 0);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 4) {
        SaturateCoverage sC(n, kernel, 0.1);
        lazyGreedyMaxSC(sC, costList, coverage, optSet, 0);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    }
    // cout << "Done with summarization\n" << flush;
}

void QuerySimVideoSummarizer::playAndSaveSummaryVideo(char* videoFileSave) {
    std::cout << "Cannot play the summary video. Instead please call display\n";
}

void QuerySimVideoSummarizer::displayAndSaveSummaryMontage(char* imageFileSave, int image_size) {
    int summary_x = ceil(sqrt(summarySet.size()));
    int summary_y = ceil(static_cast<double>(summarySet.size() / summary_x));
    std::vector<cv::Mat> summaryimages = std::vector<cv::Mat>();
    for (std::set<int>::iterator it = summarySet.begin(); it != summarySet.end(); it++) {
        summaryimages.push_back(queryVector[*it]);
    }
    cv::Mat collagesummary = cv::Mat::zeros(cv::Size(image_size * summary_x, image_size * summary_y), CV_8UC3);
    tile(summaryimages, collagesummary, summary_x, summary_y, summaryimages.size());
    cv::imshow("Summary Collage", collagesummary);
    if (imageFileSave != "") {
        cv::imwrite(imageFileSave, collagesummary);
    }
    char c = static_cast<char>(cv::waitKey(0));
    if (c == 27) {
        return;
    }
}
