#include <iostream>
#include <math.h>
#include <fstream>
#include "../src/utils/json.hpp"
#include "../src/utils/caffeClassifier.h"
#include "../src/activeLearning/ATL.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

std::vector< std::vector<float> > seedFeatureVectors;
jensen::Vector seedIntLabels;
int uncertaintyMode;

bool file_exists (const std::string& name) {
        ifstream f(name.c_str());
        return f.good();
}

void initalizeSeedForClass(std::vector<std::vector<float> > &featureVectorsOfAClass,
                           int budget,
                           int intLabel,
                           std::vector<int> &seededIndicesOfAClass) {
        std::vector<int> possibilities;
        for (int i = 0; i < featureVectorsOfAClass.size(); i++) {
                possibilities.push_back(i);
        }
        //auto engine = std::default_random_engine{};
        std::random_shuffle(possibilities.begin(), possibilities.end());
        for (int i = 0; i < budget; i++) {
                seedFeatureVectors.push_back(featureVectorsOfAClass[possibilities[i]]);
                seedIntLabels.push_back(intLabel);
                seededIndicesOfAClass.push_back(possibilities[i]);
        }
}

void loadData(std::string filePath, nlohmann::json &jsonData) {
        std::ifstream dataFile;
        dataFile.open(filePath);
        dataFile >> jsonData;
}

void saveData(std::string filePath, nlohmann::json jsonData) {
        std::cout << "saving file" << filePath << std::endl;
        std::ofstream dataFile;
        dataFile.open(filePath);
        dataFile << jsonData.dump(4);
        dataFile.close();
}

void extractFeatures(std::string dataJsonPath,
                std::vector<std::vector<float> > &featureVectors,
                jensen::Vector &intLabels,
                std::vector<std::pair<int, std::string> > &stringToIntLabels,
                nlohmann::json &featuresJson,
                std::string featuresJsonSavePath,
                std::string caffeFeatureExtractionLayer,
                CaffeClassifier* caffeClassifier,
                bool storeJson){
        int count = 0;
        nlohmann::json dataJson;
        std::cout << "extracting features from " << dataJsonPath << '\n';
        loadData(dataJsonPath, dataJson);
        std::cout << "json data loaded!" << std::endl;
        for (nlohmann::json::iterator it = dataJson.begin(); it != dataJson.end(); it++) {
		std::cout << it.key() << " " << it.value() << std::endl;
		std::vector<std::string> frames = it.value();
                std::string className = it.key();
                int currentIntLabel;
                if (className == "Unknown_Classes") {
                        currentIntLabel = -2;
                }
                else {
                        currentIntLabel = count;
                        count++;
                }
                stringToIntLabels.push_back(std::make_pair(currentIntLabel, className));
		std::cout << "here" << std::endl;

                for(int i = 0; i < frames.size(); i++) {
                        std::vector<float> currentFeatureVector = caffeClassifier->Predict(cv::imread(frames[i]), caffeFeatureExtractionLayer);
                        if (storeJson) {
                                featuresJson[className].push_back(currentFeatureVector);
                        }
                        featureVectors.push_back(currentFeatureVector);
                        intLabels.push_back(currentIntLabel);
                }
        }
        if(storeJson) {
                std::cout << "Saving features to given json file " << featuresJsonSavePath << std::endl;
                saveData(featuresJsonSavePath, featuresJson);
        }
}

int main(int argc, char *argv[]) {
//feature extractor, beta, full dataset, test dataset json, b
//int totalSubsetFunctionsSize = 8;
      if (argc < 12) {
        std::cout << "usage: ./ActiveTransferLearningExample [trainingDataJsonPath] [trainingJsonSavePath]"
          << " [testingDataJsonPath] [testingJsonSavePath] [betaPercent%] [bPercent%] [initialSeedPercentage%] [upperLimitPercent]"
          << "[uncertaintyMode-optional(0-argMax, 1-marginalSampline,2-Entropy)] [subsetSelectionMode] [saveFolderPath]"
          << std::endl;
        return -1;
      }

        double betaPercent = std::atof(argv[5]);
        double bPercent = std::atof(argv[6]);
        float initialSeedPercentage = std::atof(argv[7]);
        double upperLimitPercent = std::atof(argv[8]);
        int uncertaintyMode = std::atoi(argv[9]);
        int subsetSelectionMode = std::atoi(argv[10]);
        std::string saveFolderPath = argv[11];
        std::string csvFilePath = saveFolderPath + "/" + std::to_string(betaPercent) + "_" + std::to_string(bPercent) + "_" + std::to_string(uncertaintyMode) + "_" + std::to_string(subsetSelectionMode) + ".csv";
        if (upperLimitPercent > 100) {
                upperLimitPercent = 100;
        }
        std::string trainingDataJsonPath = argv[1]; //Json contains key value paits where key is the class name and value is a vector of full training image path(s)
        std::string trainingJsonSavePath = argv[2]; //Path to where the training features are stored.
        nlohmann::json trainingFeaturesJson; //Json containing training key,value pairs of class label and feature.
        std::vector<std::vector<float> > fullTrainingFeatureVectors;
        jensen::Vector fullTrainingIntLabels;
        std::vector<std::pair<int, std::string> > trainingStringToIntLabels;
        std::string caffeNetworkFilePath = "/home/aitoe/bvlc_alexnet/deploy.prototxt";
        std::string caffeTrainedFilePath = "/home/aitoe/bvlc_alexnet/bvlc_reference_caffenet.caffemodel";
        std::string meanFilePath = "/home/aitoe/bvlc_alexnet/imagenet_mean.binaryproto";
        std::string labelFilePath = "/home/aitoe/bvlc_alexnet/labels.txt";
        std::string caffeFeatureExtractionLayer = "fc6";
        if(!file_exists(trainingJsonSavePath)) {
                CaffeClassifier* caffeClassifier = new CaffeClassifier(caffeNetworkFilePath, caffeTrainedFilePath, meanFilePath, labelFilePath);
                extractFeatures(trainingDataJsonPath, fullTrainingFeatureVectors, fullTrainingIntLabels, trainingStringToIntLabels, trainingFeaturesJson, trainingJsonSavePath, caffeFeatureExtractionLayer, caffeClassifier, true);
                fullTrainingFeatureVectors.clear();
                fullTrainingIntLabels.clear();
                trainingStringToIntLabels.clear();
        }
        loadData(trainingJsonSavePath, trainingFeaturesJson);
        int intLabel = 0;
        for (nlohmann::json::iterator theClass = trainingFeaturesJson.begin();
             theClass != trainingFeaturesJson.end();
             theClass++) {
                std::cout << "theClass.key: " << theClass.key() << std::endl;
                trainingStringToIntLabels.push_back(std::make_pair(intLabel,
                                                                   theClass.key().c_str()));
                std::vector<std::vector<float> > featureVectorsOfAClass;
                for (nlohmann::json::iterator storedFeature = theClass.value().begin();
                     storedFeature != theClass.value().end();
                     storedFeature++) {
                        fullTrainingFeatureVectors.push_back(*storedFeature);
                        featureVectorsOfAClass.push_back(*storedFeature);
                        fullTrainingIntLabels.push_back(intLabel);
                }
                int budget = ceil((initialSeedPercentage*featureVectorsOfAClass.size())/100);
                std::vector<int> seededIndicesOfAClass;
                initalizeSeedForClass(featureVectorsOfAClass,
                                      budget,
                                      intLabel,
                                      seededIndicesOfAClass);
                std::sort(seededIndicesOfAClass.begin(), seededIndicesOfAClass.end(), std::greater<int>());
                int removeIndex = fullTrainingFeatureVectors.size()
                                  - featureVectorsOfAClass.size();
                for (int it = 0; it < seededIndicesOfAClass.size(); it++) {
                        int currentRemoveIndex = removeIndex + seededIndicesOfAClass[it];
                        fullTrainingFeatureVectors.erase(fullTrainingFeatureVectors.begin()+currentRemoveIndex);
                        fullTrainingIntLabels.erase(fullTrainingIntLabels.begin()+currentRemoveIndex);
                }
                intLabel++;
        }
        //Extracting/loading testing features
        std::string testingDataJsonPath = argv[3]; //Json contains key value paits where key is the class name and value is a vector of full testing image path(s)
        nlohmann::json testingFeaturesJson; //Json containing testing key,value pairs of class label and feature.
        std::string testingJsonSavePath = argv[4]; //Path to where the testing features are stored.
        std::vector<std::vector<float> > testingFeatureVectors;
        jensen::Vector testingIntLabels;
        std::vector<std::pair<int, std::string> > testingStringToIntLabels;
        if(!file_exists(testingJsonSavePath)) {
                extractFeatures(testingDataJsonPath, testingFeatureVectors, testingIntLabels, testingStringToIntLabels, testingFeaturesJson, testingJsonSavePath, caffeFeatureExtractionLayer, caffeClassifier, true);
                testingStringToIntLabels.clear();
                testingFeatureVectors.clear();
                testingIntLabels.clear();
        }
        loadData(testingJsonSavePath, testingFeaturesJson);
        intLabel = 0;
        for (nlohmann::json::iterator theClass = testingFeaturesJson.begin(); theClass != testingFeaturesJson.end(); theClass++) {
                testingStringToIntLabels.push_back(std::make_pair(intLabel,
                                                                  theClass.key().c_str()));
                for (nlohmann::json::iterator storedFeature = theClass.value().begin();
                     storedFeature != theClass.value().end();
                     storedFeature++) {
                        testingFeatureVectors.push_back(*storedFeature);
                        testingIntLabels.push_back(intLabel);
                }
                intLabel++;
        }
        //Initialising variables for ATL constructor
        int T = floor(upperLimitPercent/bPercent);
        if (T < 1)
                T =1;
        int beta = ceil((betaPercent*fullTrainingFeatureVectors.size())/100);
        int b = ceil((bPercent*fullTrainingFeatureVectors.size())/100);
        if (b > beta) {
                beta = b;
        }
        std::cout << "Debug UpperLimitPercent " << upperLimitPercent <<std::endl << std::flush;
        std::cout << "Debug bPercent " << bPercent <<std::endl << std::flush;
        std::cout << "Debug b " << b <<std::endl << std::flush;
        std::cout << "Debug T " << T <<std::endl << std::flush;
        std::cout << "Debug betaPercent " << betaPercent <<std::endl << std::flush;
        std::cout << "Debug beta " << beta <<std::endl << std::flush;
        //std::string resultsFileName = std::to_string(betaPercent) + "_" + std::to_string(bPercent) + ".csv";
        //preparing output csv file
        std::cout << "Preparing results csv file ..." << std::endl;

        std::ofstream csvFile;
        csvFile.open(csvFilePath);
        csvFile << "Goal 2 - Active Learning with transfer learning\n" << std::flush;
        csvFile << "Training Dataset," << trainingDataJsonPath << std::endl << std::flush;
        csvFile << "Testing Dataset," << testingDataJsonPath << std::endl << std::flush;
        csvFile << "Training data features json path," << trainingJsonSavePath << std::endl << std::flush;
        csvFile << "Testing data features json path," << testingJsonSavePath << std::endl << std::flush;
        csvFile << "Caffe feature extraction layer," << caffeFeatureExtractionLayer << std::endl << std::flush;
        csvFile << "Pre-trained Model ," << caffeTrainedFilePath << std::endl << std::flush;
        csvFile << "Total Training data ,"
            << (seedFeatureVectors.size()+fullTrainingFeatureVectors.size()) << std::endl << std::flush;
        csvFile << "Initial Seed% ," << initialSeedPercentage << std::endl << std::flush;
        csvFile << "Initial Seed ," << seedFeatureVectors.size() << std::endl << std::flush;
        csvFile << "Remaining Unlabelled Data ," << fullTrainingFeatureVectors.size() << std::endl << std::flush;
        csvFile << "Initial Beta% ," << betaPercent << std::endl << std::flush;
        csvFile << "Initial Beta ," << beta << std::endl << std::flush;
        csvFile << "Initial B% ," << bPercent << std::endl << std::flush;
        csvFile << "Initial B ," << b << std::endl << std::flush;
        csvFile << "Uncertainty selection Mode," << uncertaintyMode << std::endl << std::flush;
        csvFile << "DATK Threshold," << 0.1 << std::endl << std::flush;
        csvFile << "\n\n";
        csvFile << "Mode\\t,";
        for (int t = 0; t<=T ; t++) {
            csvFile << t << std::endl << std::flush;
        }
        csvFile << std::endl << std::flush;
        if(subsetSelectionMode == 0){
          csvFile << "FASS with SMF1 = FacilityLocation" << std::endl << std::flush;
        }
        if(subsetSelectionMode == 1){
          csvFile << "FASS with SMF2  = Disparity Min" << std::flush;
        }
        if(subsetSelectionMode == 2){
          csvFile << "Uncertainty Sampling" << std::flush;
        }
        if(subsetSelectionMode == 3){
          csvFile << "Random Selection" << std::flush;
        }
        csvFile.close();
        ATL* atl = new ATL(seedFeatureVectors, seedIntLabels, fullTrainingFeatureVectors, fullTrainingIntLabels, trainingStringToIntLabels, testingFeatureVectors, testingIntLabels, testingStringToIntLabels, beta, b, uncertaintyMode, subsetSelectionMode, csvFilePath);
        atl->run(T);
}
