#include "SimpleImageSummarizer.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "arguments.h"
#include<dirent.h>

using namespace std;
using namespace cv;

char* directory;
char* videoSaveFile;
// Output video file for saving
char* imageSaveFile;
// Output image summary file
int summaryFunction;
// 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage
int summaryAlgo;
// 0: Budgeted Summarization, 1: Stream Summarization, 2: Coverage Summarization
int budget = 20;  // summary budget in seconds
double thresh = 0.001;  // threshold for the stream Algorithm
double coverfrac = 0.9;  // coverage fraction for submodular set cover
int summary_grid = 60;
char* help;

Arg Arg::Args[]={
    Arg("directory", Arg::Req, directory, "Input Image Collection Directory",Arg::SINGLE),
    Arg("videoSaveFile", Arg::Req, videoSaveFile, "Output Summary Video File",Arg::SINGLE),
    Arg("imageSaveFile", Arg::Req, videoSaveFile, "Output Summary Image File",Arg::SINGLE),
    Arg("summarygrid", Arg::Opt, summary_grid, "Size of a image in the full grid",Arg::SINGLE),
    Arg("summaryModel", Arg::Req, summaryFunction, "Summarization Model -- 0: DisparityMin, 1: MMR, 2: FacilityLocation, 3: GraphCut, 4: SaturatedCoverage",Arg::SINGLE),
    Arg("summaryAlgo", Arg::Req, summaryAlgo, "Summarization Algorithm: 0: Budgeted Summarization, 1: Stream Summarization, 2: Coverage Summarization",Arg::SINGLE),
    Arg("budget", Arg::Opt, budget, "Budget for summarization (if summarization algo is 0)", Arg::SINGLE),
    Arg("thredshold", Arg::Opt, thresh, "Threshold for summarization (if summarization algo is 1)", Arg::SINGLE),
    Arg("coverage fraction", Arg::Opt, coverfrac, "coverage fraction for summarization (if summarization algo is 2)", Arg::SINGLE),
    Arg("help", Arg::Help, help, "Print this message"),
    Arg()
};

int main(int argc, char** argv) {

  bool parse_was_ok = Arg::parse(argc,(char**)argv);
  if (!parse_was_ok) {
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
  SimpleImageSummarizer IS(ImageCollection, summaryFunction);
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
  return 0;
}
