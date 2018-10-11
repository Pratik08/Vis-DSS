#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <set>
#include <string>
#include "set.h"
#include "/research/Suraj/WACV_2019/jensen/src/jensen.h"
#include "DeepSimImageSummarizer.h"
#include "DeepCoverImageSummarizer.h"

class ATL{
public:
  ATL(std::vector<std::vector <float>> trainingFeatureVectors, jensen::Vector trainingIntLabels, std::vector<std::vector<float>> unlabeledTrainingFeatureVectors, jensen::Vector unlabeledTrainingIntLabels, std::vector<std::pair<int, std::string>> trainingStringToIntLabels, std::vector<std::vector<float>> testingFeatureVectors, jensen::Vector testingIntLabels, std::vector<std::pair<int, std::string>> testingStringToIntLabels, int beta, int b, int uncertaintyMode, int subsetSelectionMode, std::string csvFilePath);
  ~ATL();
  void sparsifyFeatures(std::vector<float> &featureVector, jensen::SparseFeature &s);
  double getUncertainty(jensen::Vector &predictions, int mode);
  Set getBetaUncertainIndices(std::vector<std::vector<float>> &unlabelledFeatureVectors,int beta);
  std::vector<double> predictAccuracy(std::vector<jensen::SparseFeature>& testFeatures, jensen::Vector& ytest);
  std::vector<int> top5(jensen::Vector x);
  Set getBIndicesFacilityLocation(std::vector<std::vector<float>> &featureVectors, jensen::Vector featureIntLabels, double budget, Set &subsetIndices);
  Set getBIndicesDisparityMin(std::vector<std::vector<float>> &featureVectors, jensen::Vector featureIntLabels, double budget, Set &subsetIndices);
  Set getBIndicesUncertaintySampling(int budget, Set &subsetIndices);
  Set getBIndicesRandom(int n, double budget, Set subsetIndices);
  void train(int *numCorrect = NULL, int *numTotal = NULL, int *top5numCorrect = NULL, int *top5numTotal=NULL);
  void test(int *numCorrect = NULL, int *numTotal = NULL, int *top5numCorrect = NULL, int *top5numTotal = NULL);
  int run(int T);
  std::vector<std::vector <float>> trainingFeatureVectors;
  jensen::Vector trainingIntLabels;
  std::vector<std::vector<float>> testingFeatureVectors;
  jensen::Vector testingIntLabels;
  int beta;
  int b;
  int uncertaintyMode;
  int subsetSelectionMode;
  std::vector<std::pair<int, std::string>> trainingStringToIntLabels;
  std::vector<std::pair<int, std::string>> testingStringToIntLabels;
  jensen::Classifiers<jensen::SparseFeature>* model;
  std::string csvFilePath;

private:
  std::vector<std::vector<float>> unlabeledTrainingFeatureVectors;
  jensen::Vector unlabeledTrainingIntLabels;
  double LRL2RegulatizationParam = 16384;
  int LRL2OptimizationAlgo = 0;
  int LRL2NumOfIterations = 1000;
  double LRL2Tolerance = 0.01;
  std::string labelFilePath = "/research/Suraj/WACV_2019/Datasets/tinyImageNet/labels.txt";
  char* LRL2SaveModelPath = "/research/Suraj/WACV_2019/Datasets/tinyImageNet/weights.datk.json";
  double LRL2PredictionProbThresh = 0.1;
  bool trainConvertLabels = false;
  bool testConvertLabels = false;

};
