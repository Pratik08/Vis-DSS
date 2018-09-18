#include "UnsupervisedDeepSimSubsetSelection.h"

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
        sim += vec1[i] * vec2[i];
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

UnsupervisedDeepSubsetSelection::UnsupervisedDeepSubsetSelection(std::vector<cv::Mat>& ImageCollection, std::vector<std::string>& ImageCollectionName, CaffeClassifier& cc, std::string featureLayer, int summaryFunction, bool debugMode) : ImageCollection(ImageCollection), ImageCollectionName(ImageCollectionName), cc(cc), featureLayer(featureLayer), summaryFunction(summaryFunction), debugMode(debugMode) {
    n = ImageCollection.size();
    for (int i = 0; i < n; i++) {
        costList.push_back(1);
    }
}

void UnsupervisedDeepSubsetSelection::extractFeatures() {
    classifierFeatures.clear();
    for (int i = 0; i < n; i++) {
        cv::Mat frame = ImageCollection[i].clone();
        std::vector<float> feat = cc.Predict(frame, featureLayer);
        classifierFeatures.push_back(feat);
        if (debugMode) {
            std::vector<std::pair<std::string, float> > res = cc.Classify(frame);
            std::string labels = "";
            for (int i = 0; i < res.size() - 1; i++) {
                labels = labels + res[i].first + ", ";
            }
            labels = labels + res[res.size() - 1].first;
            cv::putText(frame, labels, cvPoint(30, 30),
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
            cv::imshow("Debug Image", frame);
            // Press  ESC on keyboard to exit
            char c = static_cast<char>(cv::waitKey(25));
            if (c == 27) {
                break;
            }
        }
    }
}

void UnsupervisedDeepSubsetSelection::computeKernel(int compare_method) {
    // compare_method is the comparision method for similarity (0: DotProduct, 1:GaussianSimilarity)
    float max = 0;
    for (int i = 0; i < n; i++) {
        std::vector<float> currvector;
        for (int j = 0; j < n; j++) {
            float val;
            if (compare_method == 0) {
                val = DotProduct(classifierFeatures[i], classifierFeatures[j]);
            } else {
                val = GaussianSimilarity(classifierFeatures[i], classifierFeatures[j]);
            }
            currvector.push_back(val);
        }
        kernel.push_back(currvector);
    }
}

void UnsupervisedDeepSubsetSelection::summarizeBudget(int budget) {
    std::cout << "Begin summarization\n" << std::flush;
    Set optSet;
    if (summaryFunction == 0) {
        DisparityMin dM(n, kernel);
        int inititem = 1;  // rand()%n;
        optSet.insert(inititem);
        naiveGreedyMaxKnapsack(dM, costList, budget, optSet, 1, false, true);
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
    std::cout << "Summarization is done with n = " << n << " and budget = " << budget << "\n" << std::flush;
}

void UnsupervisedDeepSubsetSelection::summarizeStream(double epsilon) {
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

void UnsupervisedDeepSubsetSelection::summarizeCover(double coverage) {
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
}

void UnsupervisedDeepSubsetSelection::writeSummarizedImagesInFile(char* subsetFilePath, int budget) {
    std::ofstream outfile;
    std::string subsetDestString(subsetFilePath);
    std::string commandDestPathCreation = "mkdir -p " + subsetDestString + "/" + std::to_string(budget);
    system(commandDestPathCreation.c_str());
    std::string sysCommand = "touch " + subsetDestString + "/" + std::to_string(budget) +  "/train.txt";
    system(sysCommand.c_str());
    outfile.open(subsetDestString + "/" + std::to_string(budget) +  "/train.txt");
    for (std::set<int>::iterator it = summarySet.begin(); it != summarySet.end(); it++) {
      std::cout << ImageCollectionName[*it] << std::endl;
      outfile << ImageCollectionName[*it] << "\n";
    }
}
