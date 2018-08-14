#include "DeepSimImageSummarizer.h"
#include "DeepCoverImageSummarizer.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "arguments.h"
#include "math.h"
#include<dirent.h>
using namespace std;
using namespace cv;

char* directory;
char* videoSaveFile;
// Output video file for saving
char* imageSaveFile;
// Output image summary file
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
char* network_file;
char* trained_file;
char* mean_file;
char* label_file;
char* featureLayer;
int summary_grid = 60;
char* help;

Arg Arg::Args[]={
    Arg("directory", Arg::Req, directory, "Input Image Collection Directory",Arg::SINGLE),
    Arg("videoSaveFile", Arg::Req, videoSaveFile, "Input Video File",Arg::SINGLE),
    Arg("imageSaveFile", Arg::Req, videoSaveFile, "Output Summary Image File",Arg::SINGLE),
    Arg("summaryModelSim", Arg::Opt, summaryFunctionSim, "Summarization Model -- 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage",Arg::SINGLE),
    Arg("summaryModelCover", Arg::Opt, summaryFunctionCover, "Summarization Model -- 0: FeatureBasedFunction, 1: Set Cover, 2: Probabilistic Set Cover",Arg::SINGLE),
    Arg("simcover", Arg::Req, simcover, "0: Similarity Based Functions, 1: Coverage Based Functions",Arg::SINGLE),
    Arg("summaryAlgo", Arg::Req, summaryAlgo, "Summarization Algorithm: 0: Budgeted Summarization, 1: Stream Summarization, 2: Coverage Summarization",Arg::SINGLE),
    Arg("summarygrid", Arg::Opt, summary_grid, "Size of a image in the full grid",Arg::SINGLE),
    Arg("featureLayer", Arg::Req, featureLayer, "Layer Name for Feature Extraction",Arg::SINGLE),
    Arg("network_file", Arg::Req, network_file, "Input Network File",Arg::SINGLE),
    Arg("trained_file", Arg::Req, trained_file, "Trained Model File",Arg::SINGLE),
    Arg("mean_file", Arg::Req, mean_file, "Mean File",Arg::SINGLE),
    Arg("label_file", Arg::Req, label_file, "Label File",Arg::SINGLE),
    Arg("budget", Arg::Opt, budget, "Budget for summarization (if summarization algo is 0)", Arg::SINGLE),
    Arg("thredshold", Arg::Opt, thresh, "Threshold for summarization (if summarization algo is 1)", Arg::SINGLE),
    Arg("coverage fraction", Arg::Opt, coverfrac, "coverage fraction for summarization (if summarization algo is 2)", Arg::SINGLE),
    Arg("help", Arg::Help, help, "Print this message"),
    Arg()
};

int main(int argc, char** argv){

  bool parse_was_ok = Arg::parse(argc,(char**)argv);
  if(!parse_was_ok){
      Arg::usage(); exit(-1);
  }
  DIR *dir;
  std::string dirName = string(directory);
  std::string imgName;
  struct dirent *ent;
  std::vector<cv::Mat> ImageCollection = std::vector<cv::Mat>();
  if (dir != NULL) {
      while ((ent = readdir (dir)) != NULL) {
          imgName = dirName + "/" + ent->d_name;
          std::cout<<imgName<<" ";
          cv::Mat img = cv::imread(imgName);
          ImageCollection.push_back(img);
        }
  }
  CaffeClassifier cc(network_file, trained_file, mean_file, label_file);
  if (simcover == 0)
  {
      DeepSimImageSummarizer IS(ImageCollection, cc, featureLayer, summaryFunctionSim);
      IS.extractFeatures();
      IS.computeKernel();
      if (summaryAlgo == 0)
      {
          IS.summarizeBudget(budget);
      }
      else if (summaryAlgo == 1)
      {
          IS.summarizeStream(thresh);
      }
      else if (summaryAlgo == 2)
      {
          IS.summarizeCover(coverfrac);
      }
      IS.displayAndSaveSummaryMontage(imageSaveFile, summary_grid);
    }
    else
    {
      DeepCoverImageSummarizer IS(ImageCollection, cc, featureLayer, summaryFunctionCover,
        FeatureBasedFnType, true);
      IS.extractFeatures();
      if (summaryAlgo == 0)
      {
          IS.summarizeBudget(budget);
      }
      else if (summaryAlgo == 1)
      {
          std::cout << "Streaming algorithms not implemented for coverage functions\n";
      }
      else if (summaryAlgo == 2)
      {
          IS.summarizeCover(coverfrac);
      }
      IS.displayAndSaveSummaryMontage(imageSaveFile, summary_grid);
    }
  // When everything done, release the video capture object
  // Closes all the frames
  return 0;
}
