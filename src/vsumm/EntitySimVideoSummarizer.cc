/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    Simple Video Summarizer using Color Histogram
 *
 */
#include "EntitySimVideoSummarizer.h"

static std::string IntToString(int a)
{
    stringstream ss;
    ss << a;
    string str = ss.str();
    return str;
}

float DotProduct(std::vector<float> vec1, std::vector<float> vec2)
{
    if (vec1.size() != vec2.size())
    {
        std::cout << "Error: Both vectors need to be of the same size\n";
    }
    float sim = 0;
    float norm1 = 0;
    float norm2 = 0;
    int n = vec1.size();
    for (int i = 0; i < n; i++)
    {
        norm1 += vec1[i]*vec1[i];
        norm2 += vec2[i]*vec2[i];
        for (int j = 0; j < n; j++)
        {
            sim += vec1[i]*vec2[j];
        }
    }
    return sim/(sqrt(norm1)*sqrt(norm2));
}

float GaussianSimilarity(std::vector<float> vec1, std::vector<float> vec2)
{
    if (vec1.size() != vec2.size())
    {
        std::cout << "Error: Both vectors need to be of the same size\n";
    }
    float diff = 0;
    float norm1 = 0;
    float norm2 = 0;
    int n = vec1.size();
    for (int i = 0; i < n; i++)
    {
        norm1 += vec1[i]*vec1[i];
        norm2 += vec2[i]*vec2[i];
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < vec2.size(); j++)
        {
            diff += vec1[i]/sqrt(norm1) - vec2[j]/sqrt(norm2);
        }
    }
    return exp(-diff/2);
}

EntitySimVideoSummarizer::EntitySimVideoSummarizer(char* videoFile, CaffeClassifier& cc, DNNClassifier& dnnc, std::string featureLayer, int summaryFunction, bool debugMode) : videoFile(videoFile), cc(cc), dnnc(dnnc), featureLayer(featureLayer), summaryFunction(summaryFunction), debugMode(debugMode)
{
    cv::VideoCapture capture(videoFile);
    frameRate = (int) capture.get(CV_CAP_PROP_FPS);
    videoLength = capture.get(CV_CAP_PROP_FRAME_COUNT)/frameRate;
    capture.release();
    featMode = 0;
}

EntitySimVideoSummarizer::EntitySimVideoSummarizer(char* videoFile, DlibClassifier& dlibc, DNNClassifier& dnnc, std::string featureLayer, int summaryFunction, bool debugMode) : videoFile(videoFile), dnnc(dnnc), dlibc(dlibc), featureLayer(featureLayer), summaryFunction(summaryFunction), debugMode(debugMode)
{
    cv::VideoCapture capture(videoFile);
    frameRate = (int) capture.get(CV_CAP_PROP_FPS);
    videoLength = capture.get(CV_CAP_PROP_FRAME_COUNT)/frameRate;
    capture.release();
    featMode = 1;
}

void EntitySimVideoSummarizer::extractFeatures()
{
  entityVector = std::vector<cv::Mat>();
  cv::VideoCapture capture(videoFile);
  frameRate = (int) capture.get(CV_CAP_PROP_FPS);
  cv::Mat frame;
  int count = 0;
    if( !capture.isOpened() )
        throw "Error when reading video file\n";
    while(1)
    {
        capture >> frame;
        if(frame.empty())
            break;
        if (count % frameRate == 0)
        {
            std::vector<std::pair<std::string, cv::Rect> > results = std::vector<std::pair<std::string, cv::Rect> >();
            dnnc.Classify(frame, results);
            for (int i = 0; i < results.size(); i++)
            {
                if(results[i].second.x >= 0 && results[i].second.y >= 0 && results[i].second.width + results[i].second.x < frame.cols && results[i].second.height + results[i].second.y < frame.rows)
                {
                    cv::Mat entityIm = frame(results[i].second);
                    entityVector.push_back(entityIm.clone());
                    if (debugMode)
                    {
                      cv::rectangle(frame, results[i].second, cv::Scalar(255, 255, 0), 2);
                    }
                }
            }
            if (debugMode)
            {
                cv::imshow("Debug", frame);
                cv::waitKey(300); // key press to close window
            }
        }
        count++;
    }
    n = entityVector.size();
    entityFeatures = std::vector<std::vector<float> >();
    std::cout << "n = " << n << "\n";
    for (int i = 0; i < n; i++)
    {
      cv::Mat frame = entityVector[i].clone();
      std::vector<float> feat = std::vector<float>();
      if (featMode == 0)
          feat = cc.Predict(frame, featureLayer);
      else if (featMode == 1)
          feat = dlibc.Predict(frame);
      entityFeatures.push_back(feat);
      std::cout << "Processed Entity " << i << "\n";
    }

}

void EntitySimVideoSummarizer::computeKernel(int compare_method)
{
    // compare_method is the comparision method for similarity (0: DotProduct, 1:GaussianSimilarity)
    float max = 0;
    std::cout << "Started Compute Kernel\n";
    kernel = std::vector<std::vector <float>>();
    for (int i = 0; i < n; i++)
    {
        std::vector<float> currvector = std::vector<float>();
        for (int j = 0; j < n; j++)
        {
            float val;
            if (compare_method == 0)
                val = DotProduct(entityFeatures[i], entityFeatures[j]);
            else
                val = GaussianSimilarity(entityFeatures[i], entityFeatures[j]);
            currvector.push_back(val);
        }
        kernel.push_back(currvector);
    }
    std::cout << "Done Compute Kernel\n";
}

void EntitySimVideoSummarizer::summarizeBudget(int budget){
    Set optSet;
    if (summaryFunction == 0)
    {
      DisparityMin dM(n, kernel);
      int inititem = 1; //rand()%n;
      optSet.insert(inititem);
      naiveGreedyMaxKnapsack(dM, costList, budget, optSet, 1, false, true);
      summarySet = std::set<int>();
      for (Set::iterator it = optSet.begin(); it!=optSet.end(); it++)
      {
          summarySet.insert(*it);
      }
    }
    else if (summaryFunction == 1)
    {
      MMR m(n, kernel);
      int inititem = rand()%n;
      optSet.insert(inititem);
      naiveGreedyMaxKnapsack(m, costList, budget, optSet, 1, false, true);
      summarySet = std::set<int>();
      for (Set::iterator it = optSet.begin(); it!=optSet.end(); it++)
      {
          summarySet.insert(*it);
      }
    }
    else if (summaryFunction == 2)
    {
      FacilityLocation fL(n, kernel);
      lazyGreedyMaxKnapsack(fL, costList, budget, optSet, 1);
      summarySet = std::set<int>();
      for (Set::iterator it = optSet.begin(); it!=optSet.end(); it++)
      {
          summarySet.insert(*it);
      }
    }
    else if (summaryFunction == 3)
    {
      GraphCutFunctions gC(n, kernel, 0.5);
      lazyGreedyMaxKnapsack(gC, costList, budget, optSet, 1);
      summarySet = std::set<int>();
      for (Set::iterator it = optSet.begin(); it!=optSet.end(); it++)
      {
          summarySet.insert(*it);
      }
    }
    else if (summaryFunction == 4)
    {
      SaturateCoverage sC(n, kernel, 0.1);
      lazyGreedyMaxKnapsack(sC, costList, budget, optSet, 1);
      summarySet = std::set<int>();
      for (Set::iterator it = optSet.begin(); it!=optSet.end(); it++)
      {
          summarySet.insert(*it);
      }
    }
    // cout << "Done with summarization\n" << flush;
}

void EntitySimVideoSummarizer::summarizeStream(double epsilon){
    Set optSet;
    if (summaryFunction == 0)
    {
      DisparityMin dM(n, kernel);
      optSet.insert(0);
      vector<int> order(n, 1);
      for (int i = 0; i < n; i++)
          order[i] = i;
      streamGreedy(dM, epsilon, optSet, order);
      optSet.insert(n-1);
      summarySet = std::set<int>();
      for (Set::iterator it = optSet.begin(); it!=optSet.end(); it++)
      {
          summarySet.insert(*it);
      }
    }
    else if (summaryFunction == 1)
    {
      MMR m(n, kernel);
      optSet.insert(0);
      vector<int> order(n, 1);
      for (int i = 0; i < n; i++)
          order[i] = i;
      streamGreedy(m, epsilon, optSet, order);
      optSet.insert(n-1);
      summarySet = std::set<int>();
      for (Set::iterator it = optSet.begin(); it!=optSet.end(); it++)
      {
          summarySet.insert(*it);
      }
    }
    else if (summaryFunction == 2)
    {
      FacilityLocation fL(n, kernel);
      optSet.insert(0);
      vector<int> order(n, 1);
      for (int i = 0; i < n; i++)
          order[i] = i;
      streamGreedy(fL, epsilon, optSet, order);
      optSet.insert(n-1);
      summarySet = std::set<int>();
      for (Set::iterator it = optSet.begin(); it!=optSet.end(); it++)
      {
          summarySet.insert(*it);
      }
    }
    else if (summaryFunction == 3)
    {
      GraphCutFunctions gC(n, kernel, 0.5);
      optSet.insert(0);
      vector<int> order(n, 1);
      for (int i = 0; i < n; i++)
          order[i] = i;
      streamGreedy(gC, epsilon, optSet, order);
      optSet.insert(n-1);
      summarySet = std::set<int>();
      for (Set::iterator it = optSet.begin(); it!=optSet.end(); it++)
      {
          summarySet.insert(*it);
      }
    }
    else if (summaryFunction == 4)
    {
      SaturateCoverage sC(n, kernel, 0.1);
      optSet.insert(0);
      vector<int> order(n, 1);
      for (int i = 0; i < n; i++)
          order[i] = i;
      streamGreedy(sC, epsilon, optSet, order);
      optSet.insert(n-1);
      summarySet = std::set<int>();
      for (Set::iterator it = optSet.begin(); it!=optSet.end(); it++)
      {
          summarySet.insert(*it);
      }
    }
}

void EntitySimVideoSummarizer::summarizeCover(double coverage){
    Set optSet;
    if (summaryFunction == 0)
    {
      std::cout << "Cover Summarization is not supported for Disparity Min Function\n";
    }
    else if (summaryFunction == 1)
    {
      std::cout << "Cover Summarization is not supported for MMR Function\n";
    }
    else if (summaryFunction == 2)
    {
      FacilityLocation fL(n, kernel);
      lazyGreedyMaxSC(fL, costList, coverage, optSet, 0);
      summarySet = std::set<int>();
      for (Set::iterator it = optSet.begin(); it!=optSet.end(); it++)
      {
          summarySet.insert(*it);
      }
    }
    else if (summaryFunction == 3)
    {
      GraphCutFunctions gC(n, kernel, 0.5);
      lazyGreedyMaxSC(gC, costList, coverage, optSet, 0);
      summarySet = std::set<int>();
      for (Set::iterator it = optSet.begin(); it!=optSet.end(); it++)
      {
          summarySet.insert(*it);
      }
    }
    else if (summaryFunction == 4)
    {
      SaturateCoverage sC(n, kernel, 0.1);
      lazyGreedyMaxSC(sC, costList, coverage, optSet, 0);
      summarySet = std::set<int>();
      for (Set::iterator it = optSet.begin(); it!=optSet.end(); it++)
      {
          summarySet.insert(*it);
      }
    }
    // cout << "Done with summarization\n" << flush;
}

void EntitySimVideoSummarizer::playAndSaveSummaryVideo(char* videoFileSave)
{
    std::cout << "Cannot play the summary video. Instead please call display\n";
}

void EntitySimVideoSummarizer::displayAndSaveSummaryMontage(char* imageFileSave, int image_size)
{
  int summary_x = ceil(sqrt(summarySet.size()));
  int summary_y = ceil((double)summarySet.size()/summary_x);
  std::vector<cv::Mat> summaryimages = std::vector<cv::Mat>();
  for (set<int>::iterator it = summarySet.begin(); it!=summarySet.end(); it++)
  {
      summaryimages.push_back(entityVector[*it]);
  }
  cv::Mat collagesummary = cv::Mat::zeros(cv::Size(image_size*summary_x,image_size*summary_y),CV_8UC3);
  tile(summaryimages, collagesummary, summary_x, summary_y, summaryimages.size());
  cv::imshow("Summary Collage",collagesummary);
  if (imageFileSave != "")
      cv::imwrite(imageFileSave, collagesummary);
  char c = (char)cv::waitKey(0);
  if(c==27)
      return;
}
