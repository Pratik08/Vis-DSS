#include <math.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "SupervisedDeepSimSubsetSelection.h"
#include "arguments.h"

char* supersetFilePath;
char* subsetFilePath;
int summaryFunctionSim = 0;
// 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage
int summaryFunctionCover = 0;
// 0: FeatureBasedFunction, 1: Set Cover, 2: Probabilistic Set Cover
int FeatureBasedFnType = 1;
int simcover = 0;
int segmentType;
// 0: Fixed Length Segments, 1: Segments based on Shot Detectors
int summaryAlgo;
// 0: Budgeted Summarization, 1: Stream Summarization, 2: Coverage Summarization
int snippetLength = 2;  // in case of fixed length snippets, the length of the snippetHist
int budget = 10;
double thresh = 0.001;  // threshold for the stream Algorithm
double coverfrac = 0.9;  // coverage fraction for submodular set cover
char* network_file;
char* trained_file;
char* mean_file;
char* label_file;
char* featureLayer;
int summary_grid = 60;
char* help;

Arg Arg::Args[] = {
    Arg("supersetFilePath", Arg::Req, supersetFilePath, "superSetTrainTxt: Image name should be of the format class_imgName.ext", Arg::SINGLE),
    Arg("subsetFilePath", Arg::Req, subsetFilePath, "subsetFileDestination", Arg::SINGLE),
    Arg("summaryModelSim", Arg::Opt, summaryFunctionSim, "Summarization Model -- 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage", Arg::SINGLE),
    Arg("summaryModelCover", Arg::Opt, summaryFunctionCover, "Summarization Model -- 0: FeatureBasedFunction, 1: Set Cover, 2: Probabilistic Set Cover", Arg::SINGLE),
    Arg("simcover", Arg::Req, simcover, "0: Similarity Based Functions, 1: Coverage Based Functions", Arg::SINGLE),
    Arg("summaryAlgo", Arg::Req, summaryAlgo, "Summarization Algorithm: 0: Budgeted Summarization, 1: Stream Summarization, 2: Coverage Summarization", Arg::SINGLE),
    Arg("featureLayer", Arg::Req, featureLayer, "Layer Name for Feature Extraction", Arg::SINGLE),
    Arg("network_file", Arg::Req, network_file, "Input Network File", Arg::SINGLE),
    Arg("trained_file", Arg::Req, trained_file, "Trained Model File", Arg::SINGLE),
    Arg("mean_file", Arg::Req, mean_file, "Mean File", Arg::SINGLE),
    Arg("label_file", Arg::Req, label_file, "Label File", Arg::SINGLE),
    Arg("thredshold", Arg::Opt, thresh, "Threshold for summarization (if summarization algo is 1)", Arg::SINGLE),
    Arg("coverage fraction", Arg::Opt, coverfrac, "coverage fraction for summarization (if summarization algo is 2)", Arg::SINGLE),
    Arg("budget", Arg::Opt, budget, "budget for summarization (if summarization algo is 0)", Arg::SINGLE),
    Arg("help", Arg::Help, help, "Print this message"),
    Arg()
};

int main(int argc, char** argv) {
    bool parse_was_ok = Arg::parse(argc, (char**)argv);
    if (!parse_was_ok) {
        Arg::usage(); exit(-1);
    }

    std::ifstream in(supersetFilePath);

    if (!in) {
      std::cout << "Cannot open input file.\n";
      return 1;
    }


    std::string imageName;
    std::vector<cv::Mat> ImageCollection = std::vector<cv::Mat>();
    std::vector<std::string> ImageCollectionName = std::vector<std::string>();
    std::vector<int> ImageCollectionLabels = std::vector<int>();
    CaffeClassifier cc(network_file, trained_file, mean_file, label_file);
    std::string imageNameWithClass;
    std::string label;
    while (std::getline(in, imageName)) {
      if (imageName.substr(imageName.find_last_of(".") + 1) != "DS_Store") {
        cv::Mat img = cv::imread(imageName);
        imageNameWithClass = imageName.substr(imageName.find_last_of("/") + 1);
        label = imageNameWithClass.substr(0,imageNameWithClass.find("_"));
        if (!img.empty()) {
          ImageCollection.push_back(img);
          ImageCollectionName.push_back(imageName);
          ImageCollectionLabels.push_back(std::stoi(label));
          std::cout << imageName << " " << label <<'\n';
        }
      }
    }

    if (simcover == 0) {
        SupervisedDeepSimSubsetSelection DSS(ImageCollection, ImageCollectionName, ImageCollectionLabels, cc, featureLayer, summaryFunctionSim, false);
        DSS.extractFeatures();
        DSS.computeKernel();
        if (summaryAlgo == 0) {
            DSS.summarizeBudget(budget);
        } else if (summaryAlgo == 1) {
            DSS.summarizeStream(thresh);
        } else if (summaryAlgo == 2) {
            DSS.summarizeCover(coverfrac);
        }
        DSS.writeSummarizedImagesInFile(subsetFilePath, budget);
    } else {
        std::cout << "Only Similarity based summarization available currently" << std::endl;
    }
    return 0;
}
