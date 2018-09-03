#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "SimpleVideoSummarizer.h"
#include "arguments.h"

char* videoFile;
char* videoSaveFile;
int summaryFunction;  // 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage
int segmentType;  // 0: Fixed Length Segments, 1: Segments based on Shot Detectors
int summaryAlgo;  // 0: Budgeted Summarization, 1: Stream Summarization, 2: Coverage Summarization
int snippetLength = 2;  // in case of fixed length snippets, the length of the snippetHist
int budget = 120;  // summary budget in seconds
double thresh = 0.001;  // threshold for the stream Algorithm
double coverfrac = 0.9;  // coverage fraction for submodular set cover
char* help;

Arg Arg::Args[] = {
    Arg("videoFile", Arg::Req, videoFile, "Input Video File", Arg::SINGLE),
    Arg("videoSaveFile", Arg::Req, videoSaveFile, "Input Video File", Arg::SINGLE),
    Arg("summaryModel", Arg::Req, summaryFunction, "Summarization Model -- 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage", Arg::SINGLE),
    Arg("segmentType", Arg::Req, segmentType, "Segment Type -- 0: Fixed Length Segments, 1: Segments based on Shot Detectors", Arg::SINGLE),
    Arg("summaryAlgo", Arg::Req, summaryAlgo, "Summarization Algorithm: 0: Budgeted Summarization, 1: Stream Summarization, 2: Coverage Summarization", Arg::SINGLE),
    Arg("snippetLength", Arg::Opt, snippetLength, "Snippet Length", Arg::SINGLE),
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

    SimpleVideoSummarizer VS(videoFile, summaryFunction, segmentType, snippetLength);
    VS.extractFeatures();
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
