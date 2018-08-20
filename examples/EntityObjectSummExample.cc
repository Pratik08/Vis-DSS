#include "EntitySimVideoSummarizer.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "arguments.h"

using namespace std;
using namespace cv;

char* videoFile;
char* imageSaveFile;
// Video File to analyze
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
int snippetLength = 2; // in case of fixed length snippets, the length of the snippetHist
int budget = 120; // summary budget in seconds
double thresh = 0.001; // threshold for the stream Algorithm
double coverfrac = 0.9; // coverage fraction for submodular set cover
char* config_file_object;
char* weights_file_object;
char* label_file_object;
double threshold_object;
char* network_file;
char* trained_file;
char* mean_file;
char* label_file;
char* featureLayer;
char* help;
bool debug = true;
int summary_grid = 60;

Arg Arg::Args[] = {
    Arg("videoFile", Arg::Req, videoFile, "Input Video File", Arg::SINGLE),
    Arg("imageSaveFile", Arg::Req, imageSaveFile, "Input Video File", Arg::SINGLE),
    Arg("summaryModelSim", Arg::Opt, summaryFunctionSim, "Summarization Model -- 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage", Arg::SINGLE),
    Arg("summaryAlgo", Arg::Req, summaryAlgo, "Summarization Algorithm: 0: Budgeted Summarization, 1: Stream Summarization, 2: Coverage Summarization", Arg::SINGLE),
    Arg("summarygrid", Arg::Opt, summary_grid, "Size of a image in the full grid", Arg::SINGLE),
    Arg("featureLayer", Arg::Req, featureLayer, "Layer Name for Feature Extraction", Arg::SINGLE),
    Arg("config_file_object", Arg::Req, config_file_object, "Input Network File", Arg::SINGLE),
    Arg("weights_file_object", Arg::Req, weights_file_object, "Trained Model File", Arg::SINGLE),
    Arg("label_file_object", Arg::Req, label_file_object, "Label File", Arg::SINGLE),
    Arg("threshold_object", Arg::Req, threshold_object, "Threshold for Object Detection", Arg::SINGLE),
    Arg("network_file", Arg::Req, network_file, "Input Network File for Face detection", Arg::SINGLE),
    Arg("trained_file", Arg::Req, trained_file, "Trained Model File for Face detection", Arg::SINGLE),
    Arg("mean_file", Arg::Req, mean_file, "Mean File for Face detection", Arg::SINGLE),
    Arg("label_file", Arg::Req, label_file, "Label File for Face detection", Arg::SINGLE),
    Arg("budget", Arg::Opt, budget, "Budget for summarization (if summarization algo is 0)", Arg::SINGLE),
    Arg("thredshold", Arg::Opt, thresh, "Threshold for summarization (if summarization algo is 1)", Arg::SINGLE),
    Arg("coverage fraction", Arg::Opt, coverfrac, "coverage fraction for summarization (if summarization algo is 2)", Arg::SINGLE),
    Arg("help", Arg::Help, help, "Print this message"),
    Arg()
};

int main(int argc, char** argv) {

    bool parse_was_ok = Arg::parse(argc, (char**)argv);
    if (!parse_was_ok) {
        Arg::usage(); exit(-1);
    }
    DNNClassifier dnnc("OBJECT", "DARKNET_CPU", config_file_object, weights_file_object, label_file_object, threshold_object);
    CaffeClassifier cc(network_file, trained_file, mean_file, label_file);
    EntitySimVideoSummarizer ES(videoFile, cc, dnnc, featureLayer, summaryFunctionSim);
    ES.extractFeatures();
    ES.computeKernel();
    if (summaryAlgo == 0) {
        ES.summarizeBudget(budget);
    }else if (summaryAlgo == 1)  {
        ES.summarizeStream(thresh);
    }else if (summaryAlgo == 2)  {
        ES.summarizeCover(coverfrac);
    }
    ES.displayAndSaveSummaryMontage(imageSaveFile, summary_grid);
}
