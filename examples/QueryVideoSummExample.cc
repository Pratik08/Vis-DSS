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
int segmentType;
// 0: Fixed Length Segments, 1: Segments based on Shot Detectors
int summaryAlgo;
// 0: Budgeted Summarization, 1: Stream Summarization, 2: Coverage Summarization
int snippetLength = 2;  // in case of fixed length snippets, the length of the snippetHist
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
char* help;

Arg Arg::Args[] = {
    Arg("videoFile", Arg::Req, videoFile, "Input Video File", Arg::SINGLE),
    Arg("videoSaveFile", Arg::Req, videoSaveFile, "Input Video File", Arg::SINGLE),
    Arg("queryInput", Arg::Req, queryInput, "Query input class", Arg::SINGLE),
    Arg("summaryModelSim", Arg::Opt, summaryFunctionSim, "Summarization Model -- 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage", Arg::SINGLE),
    Arg("segmentType", Arg::Req, segmentType, "Segment Type -- 0: Fixed Length Segments, 1: Segments based on Shot Detectors", Arg::SINGLE),
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
    Arg("snippetLength", Arg::Opt, snippetLength, "Snippet Length", Arg::SINGLE),
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

    CaffeClassifier ccLabel(label_network_file, label_trained_file, label_mean_file, label_label_file);
    CaffeClassifier ccFeature(feature_network_file, feature_trained_file, feature_mean_file, feature_label_file);
    QuerySimVideoSummarizer VS(videoFile, ccLabel, ccFeature, featureLayer, summaryFunctionSim, segmentType);
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
    // VS.displayAndSaveSummaryMontage(videoSaveFile, 60);
    // When everything done, release the video capture object
    // Closes all the frames
    return 0;
}
