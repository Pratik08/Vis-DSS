/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    Simple Video Summarizer using Color Histogram
 *
 */
#include "DeepCoverVideoSummarizer.h"

static std::string IntToString(int a) {
    std::stringstream ss;
    ss << a;
    std::string str = ss.str();
    return str;
}

DeepCoverVideoSummarizer::DeepCoverVideoSummarizer(char* videoFile, CaffeClassifier& cc, std::string featureLayer,
                                                   int summaryFunction, int FeatureBasedFnType, int segmentType, int snippetLength, bool debugMode) : videoFile(videoFile), cc(cc), featureLayer(featureLayer), summaryFunction(summaryFunction), FeatureBasedFnType(FeatureBasedFnType), segmentType(segmentType), snippetLength(snippetLength), debugMode(debugMode) {
    cv::VideoCapture capture(videoFile);
    frameRate = static_cast<int>(capture.get(CV_CAP_PROP_FPS));
    videoLength = capture.get(CV_CAP_PROP_FRAME_COUNT) / frameRate;
    std::cout << "The video Length is " << videoLength << " and the frameRate is " << frameRate << "\n";
    if (segmentType == 0) {
        for (int i = 0; i < videoLength; i += snippetLength) {
            segmentStartTimes.push_back(i);
        }
    } else {
        segmentStartTimes = shotDetector(capture);
    }
    capture.release();
}

void DeepCoverVideoSummarizer::extractFeatures() {
    cv::VideoCapture capture(videoFile);
    capture.set(CV_CAP_PROP_POS_FRAMES, 0);
    cv::Mat frame;
    std::vector<cv::Mat> CurrVideo = std::vector<cv::Mat> ();
    if (!capture.isOpened()) {
        std::cout << "Error when reading steam" << "\n";
    }
    int frame_count = 0;
    int samplingRate = 1;
    costList = std::vector<double>();
    std::map <std::string, int> labelMap;
    std::vector<std::vector<std::string> > labelSegmentList = std::vector<std::vector<std::string> >();
    for (int i = 0; i < segmentStartTimes.size() - 1; i++) {
        std::vector<float> feat;
        SparseFeature sfeat;
        std::vector<float> p;
        if (segmentStartTimes[i + 1] - segmentStartTimes[i] == 1) {
            cv::MatND hist;
            capture.set(CV_CAP_PROP_POS_FRAMES, segmentStartTimes[i] * frameRate);
            capture >> frame;
            feat = cc.Predict(frame, featureLayer);
            std::cout << "\n";
            sfeat = getSparseFeature(feat);
            p = cc.Predict(frame);
            std::vector<std::pair<std::string, float> > res = cc.Classify(frame);
            std::vector<std::string> currLabels = std::vector<std::string>();
            for (int i = 0; i < res.size(); i++) {
                currLabels.push_back(res[i].first);
                if (labelMap.count(res[i].first) == 0)
                    labelMap[res[i].first] = labelMap.size();
            }
            labelSegmentList.push_back(currLabels);
            if (segmentType == 1) {
                costList.push_back(SmallShotPenalty);
            } else {
                costList.push_back(1);
            }
        } else {
            for (int j = segmentStartTimes[i]; j < segmentStartTimes[i + 1]; j++) {
                capture.set(CV_CAP_PROP_POS_FRAMES, j * frameRate);
                capture >> frame;
                CurrVideo.push_back(frame.clone());
            }
            cv::MatND hist;
            feat = cc.Predict(CurrVideo, featureLayer);
            for (int i = 0; i < feat.size(); i++) {
                feat[i] = abs(feat[i]);
            }
            nFeatures = feat.size();
            sfeat = getSparseFeature(feat);
            p = cc.Predict(CurrVideo);
            nConcepts = p.size();
            std::vector<std::pair<std::string, float> > res = cc.Classify(frame);
            std::vector<std::string> currLabels = std::vector<std::string>();
            for (int i = 0; i < res.size(); i++) {
                currLabels.push_back(res[i].first);
                if (labelMap.count(res[i].first) == 0)
                    labelMap[res[i].first] = labelMap.size();
            }
            labelSegmentList.push_back(currLabels);
            costList.push_back(CurrVideo.size());
            CurrVideo.clear();
        }
        classifierFeatures.push_back(sfeat);
        pVec.push_back(p);
        if (debugMode) {
            std::vector<std::pair<std::string, float> > res = cc.Classify(frame);
            std::string labels = "";
            for (int i = 0; i < res.size() - 1; i++) {
                labels = labels + res[i].first + ", ";
            }
            labels = labels + res[res.size() - 1].first;
            cv::putText(frame, labels, cvPoint(30, 30),
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
            if (frame.data) {
                cv::imshow("Debug Video", frame);
            }
            // Press  ESC on keyboard to exit
            char c = static_cast<char>(cv::waitKey(25));
            if (c == 27) {
                break;
            }
        }
    }
    capture.release();
    n = classifierFeatures.size();
    capture.release();
    coverSet = std::vector<Set>();      // The sets U_i, i \in V
    for (int i = 0; i < labelSegmentList.size(); i++) {
        Set currCoverSet = Set();
        for (int j = 0; j < labelSegmentList[i].size(); j++) {
            currCoverSet.insert(labelMap[labelSegmentList[i][j]]);
        }
        coverSet.push_back(currCoverSet);
    }
}

void DeepCoverVideoSummarizer::summarizeBudget(int budget) {
    Set optSet = Set();
    if (summaryFunction == 0) {
        std::vector<double> featureWeights = std::vector<double>(nFeatures, 1);
        for (int i = 0; i < featureWeights.size(); i++) {
            std::cout << featureWeights[i] << " ";
        }
        FeatureBasedFunctions ff(n, FeatureBasedFnType, classifierFeatures, featureWeights);
        lazyGreedyMaxKnapsack(ff, costList, budget, optSet, 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 1) {
        SetCover sc(n, coverSet);
        lazyGreedyMaxKnapsack(sc, costList, budget, optSet, 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 2) {
        ProbabilisticSetCover psc(n, nConcepts, pVec);
        lazyGreedyMaxKnapsack(psc, costList, budget, optSet, 1);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    }
}

void DeepCoverVideoSummarizer::summarizeCover(double coverage) {
    Set optSet;
    if (summaryFunction == 0) {
        std::vector<double> featureWeights = std::vector<double>(nFeatures, 1);
        FeatureBasedFunctions ff(n, FeatureBasedFnType, classifierFeatures, featureWeights);
        lazyGreedyMaxSC(ff, costList, coverage, optSet, 0);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 1) {
        SetCover sc(n, coverSet);
        lazyGreedyMaxSC(sc, costList, coverage, optSet, 0);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    } else if (summaryFunction == 2) {
        ProbabilisticSetCover psc(n, nConcepts, pVec);
        lazyGreedyMaxSC(psc, costList, coverage, optSet, 0);
        summarySet = std::set<int>();
        for (Set::iterator it = optSet.begin(); it != optSet.end(); it++) {
            summarySet.insert(*it);
        }
    }
}

void DeepCoverVideoSummarizer::playAndSaveSummaryVideo(char* videoFileSave) {
    cv::VideoCapture capture(videoFile);
    cv::Mat frame;
    capture.set(CV_CAP_PROP_POS_FRAMES, 0);
    cv::VideoWriter videoWriter;
    if (videoFileSave != "") {
        videoWriter = cv::VideoWriter(videoFileSave, CV_FOURCC('M', 'J', 'P', 'G'), static_cast<int>(capture.get(CV_CAP_PROP_FPS)), cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
    }
    for (std::set<int>::iterator it = summarySet.begin(); it != summarySet.end(); it++) {
        capture.set(CV_CAP_PROP_POS_FRAMES, segmentStartTimes[*it] * frameRate);
        for (int i = segmentStartTimes[*it]; i < segmentStartTimes[*it + 1]; i++) {
            for (int j = 0; j < frameRate; j++) {
                capture >> frame;
                cv::putText(frame, "Time: " + IntToString(i) + " seconds", cvPoint(30, 30),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
                if (frame.data) {
                    cv::imshow("Summary Video", frame);
                }
                if (videoFileSave != "") {
                    videoWriter.write(frame);
                }
                char c = static_cast<char>(cv::waitKey(25));
                if (c == 27) {
                    break;
                }
            }
        }
    }
    capture.release();
}

void DeepCoverVideoSummarizer::displayAndSaveSummaryMontage(char* imageFileSave, int image_size) {
    int summary_x = ceil(sqrt(summarySet.size()));
    int summary_y = ceil(summarySet.size() / summary_x);
    std::vector<cv::Mat> summaryimages = std::vector<cv::Mat>();
    cv::VideoCapture capture(videoFile);
    cv::Mat frame;
    capture.set(CV_CAP_PROP_POS_FRAMES, 0);
    for (std::set<int>::iterator it = summarySet.begin(); it != summarySet.end(); it++) {
        capture.set(CV_CAP_PROP_POS_FRAMES, segmentStartTimes[*it] * frameRate);
        capture >> frame;
        summaryimages.push_back(frame);
    }
    capture.release();
    cv::Mat collagesummary = cv::Mat(image_size * summary_y, image_size * summary_x, CV_8UC3);
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
