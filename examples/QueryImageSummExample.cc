#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <dirent.h>
#include "QuerySimImageSummarizer.h"
#include "arguments.h"

char* directory;
char* videoSaveFile;
// Output video file for saving
char* imageSaveFile;
// Video File to analyze
int summaryFunctionSim = 0;
// 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage
int summaryAlgo;
// 0: Budgeted Summarization, 1: Stream Summarization, 2: Coverage Summarization
int budget = 120;  // summary budget in seconds
double thresh = 0.001;  // threshold for the stream Algorithm
double coverfrac = 0.9;  // coverage fraction for submodular set cover
char* queryInput;
char* label_network_file;
char* label_trained_file;
char* label_mean_file;
char* label_label_file;
char* feature_network_file;
char* feature_trained_file;
char* feature_mean_file;
char* feature_label_file;
char* featureLayer;
int summary_grid = 60;
char* help;

Arg Arg::Args[] = {
    Arg("directory", Arg::Req, directory, "Input Image Directory", Arg::SINGLE),
    Arg("imageSaveFile", Arg::Req, imageSaveFile, "Input Image Save File", Arg::SINGLE),
    Arg("queryInput", Arg::Req, queryInput, "Query input class", Arg::SINGLE),
    Arg("summaryModelSim", Arg::Opt, summaryFunctionSim, "Summarization Model -- 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage", Arg::SINGLE),
    Arg("summaryAlgo", Arg::Req, summaryAlgo, "Summarization Algorithm: 0: Budgeted Summarization, 1: Stream Summarization, 2: Coverage Summarization", Arg::SINGLE),
    Arg("featureLayer", Arg::Req, featureLayer, "Layer Name for Feature Extraction", Arg::SINGLE),
    Arg("label_network_file", Arg::Req, label_network_file, "Query Label Network File", Arg::SINGLE),
    Arg("label_trained_file", Arg::Req, label_trained_file, "Query Label Trained Model File", Arg::SINGLE),
    Arg("label_mean_file", Arg::Opt, label_mean_file, "Query Label Mean File", Arg::SINGLE),
    Arg("label_label_file", Arg::Opt, label_label_file, "Query Label - Label File", Arg::SINGLE),
    Arg("feature_network_file", Arg::Req, feature_network_file, "Feature Network File", Arg::SINGLE),
    Arg("feature_trained_file", Arg::Req, feature_trained_file, "Feature Trained Model File", Arg::SINGLE),
    Arg("feature_mean_file", Arg::Opt, feature_mean_file, "Feature Mean File", Arg::SINGLE),
    Arg("feature_label_file", Arg::Opt, feature_label_file, "Feature Label File", Arg::SINGLE),
    Arg("summarygrid", Arg::Opt, summary_grid, "Size of a image in the full grid", Arg::SINGLE),
    Arg("budget", Arg::Opt, budget, "Budget for summarization (if summarization algo is 0)", Arg::SINGLE),
    Arg("threshold", Arg::Opt, thresh, "Threshold for summarization (if summarization algo is 1)", Arg::SINGLE),
    Arg("coverage fraction", Arg::Opt, coverfrac, "coverage fraction for summarization (if summarization algo is 2)", Arg::SINGLE),
    Arg("help", Arg::Help, help, "Print this message"),
    Arg()
};

int main(int argc, char** argv) {
    bool parse_was_ok = Arg::parse(argc, (char**)argv);
    if (!parse_was_ok) {
        Arg::usage(); exit(-1);
    }
    DIR *dir;
    std::string dirName = std::string(directory);
    std::string imgName;
    struct dirent *ent;
    CaffeClassifier ccLabel(label_network_file, label_trained_file, label_mean_file, label_label_file);
    CaffeClassifier ccFeature(feature_network_file, feature_trained_file, feature_mean_file, feature_label_file);
    std::vector<cv::Mat> ImageCollection = std::vector<cv::Mat>();
    std::cout << dirName << "\n";
    dir = opendir(directory);
    if (dir != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            imgName = dirName + "/" + ent->d_name;
            std::cout << imgName << "\n";
            cv::Mat img = cv::imread(imgName);
            if (!img.empty()) {
                ImageCollection.push_back(img);
            }
        }
    }
    QuerySimImageSummarizer IS(ImageCollection, ccLabel, ccFeature, featureLayer, summaryFunctionSim);
    IS.extractFeatures();
    IS.processQuery(queryInput);
    IS.computeKernel();
    if (summaryAlgo == 0) {
        IS.summarizeBudget(budget);
    } else if (summaryAlgo == 1) {
        IS.summarizeStream(thresh);
    } else if (summaryAlgo == 2) {
        IS.summarizeCover(coverfrac);
    }
    IS.displayAndSaveSummaryMontage(imageSaveFile, summary_grid);
    return 0;
}
