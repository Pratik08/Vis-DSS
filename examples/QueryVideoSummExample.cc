#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "QuerySimVideoSummarizer.h"
#include "arguments.h"

char* videoFile;
char* videoSaveFile;
// Video File to analyze
int summaryFunctionSim = 0;
// 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage
int summaryAlgo;
// 0: Budgeted Summarization, 1: Stream Summarization, 2: Coverage Summarization
int budget = 120;  // summary budget in seconds
double thresh = 0.001;  // threshold for the stream Algorithm
double coverfrac = 0.9;  // coverage fraction for submodular set cover
char* queryInput;
char* network_file;
char* trained_file;
char* mean_file;
char* label_file;
char* featureLayer;
char* help;

Arg Arg::Args[] = {
    Arg("videoFile", Arg::Req, videoFile, "Input Video File", Arg::SINGLE),
    Arg("videoSaveFile", Arg::Req, videoSaveFile, "Input Video File", Arg::SINGLE),
    Arg("queryInput", Arg::Req, queryInput, "Query input class", Arg::SINGLE),
    Arg("summaryModelSim", Arg::Opt, summaryFunctionSim, "Summarization Model -- 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage", Arg::SINGLE),
    Arg("summaryAlgo", Arg::Req, summaryAlgo, "Summarization Algorithm: 0: Budgeted Summarization, 1: Stream Summarization, 2: Coverage Summarization", Arg::SINGLE),
    Arg("featureLayer", Arg::Req, featureLayer, "Layer Name for Feature Extraction", Arg::SINGLE),
    Arg("network_file", Arg::Req, network_file, "Input Network File", Arg::SINGLE),
    Arg("trained_file", Arg::Req, trained_file, "Trained Model File", Arg::SINGLE),
    Arg("mean_file", Arg::Req, mean_file, "Mean File", Arg::SINGLE),
    Arg("label_file", Arg::Req, label_file, "Label File", Arg::SINGLE),
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
    CaffeClassifier cc(network_file, trained_file, mean_file, label_file);
    QuerySimVideoSummarizer VS(videoFile, cc, featureLayer, summaryFunctionSim);
    VS.extractFeatures();
    VS.processQuery(queryInput);
    VS.computeKernel();
    if (summaryAlgo == 0) {
        VS.summarizeBudget(budget);
    } else if (summaryAlgo == 1) {
        VS.summarizeStream(thresh);
    } else if (summaryAlgo == 2) {
        VS.summarizeCover(coverfrac);
    }
    VS.playAndSaveSummaryVideo(videoSaveFile);
    // When everything done, release the video capture object
    // Closes all the frames
    return 0;
}
