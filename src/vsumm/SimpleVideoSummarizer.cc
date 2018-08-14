/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
    Simple Video Summarizer using Color Histogram
 *
 */
#include "SimpleVideoSummarizer.h"

std::string IntToString(int a)
{
    stringstream ss;
    ss << a;
    string str = ss.str();
    return str;
}
double SmallShotPenalty = 10;

SimpleVideoSummarizer::SimpleVideoSummarizer(char* videoFile, int summaryFunction, int segmentType,
  int snippetLength) : videoFile(videoFile), summaryFunction(summaryFunction), segmentType(segmentType),
  snippetLength(snippetLength)
{
    cv::VideoCapture capture(videoFile);

    frameRate = (int) capture.get(CV_CAP_PROP_FPS);
    videoLength = capture.get(CV_CAP_PROP_FRAME_COUNT)/frameRate;
    std::cout << "The video Length is " << videoLength << " and the frameRate is " << frameRate << "\n";
    if (segmentType == 0) {
      for (int i = 0; i < videoLength; i += snippetLength)
        segmentStartTimes.push_back(i);
    }
    else {
        segmentStartTimes = shotDetector(capture);
    }
    capture.release();
}

void SimpleVideoSummarizer::extractFeatures(double resizeParam){
    cv::VideoCapture capture(videoFile);
    capture.set(CV_CAP_PROP_POS_FRAMES, 0);
    cv::Mat frame;
    std::vector<cv::Mat> CurrVideo = std::vector<cv::Mat> ();
    if( !capture.isOpened() )
            std::cout << "Error when reading steam" << "\n";
    int frame_count = 0;
    int samplingRate = 1;
    costList = std::vector<double>();
    for (int i = 0; i < segmentStartTimes.size()-1; i++)
    {
      if (segmentStartTimes[i+1] - segmentStartTimes[i] == 1) {
          cv::MatND hist;
          capture.set(CV_CAP_PROP_POS_FRAMES, segmentStartTimes[i]*frameRate);
          capture >> frame;
          cv::resize(frame, frame, cv::Size(frame.cols*resizeParam, frame.rows*resizeParam));
          colorHist(frame, hist);
          snippetHist.push_back(hist);
          if (segmentType == 1)
              costList.push_back(SmallShotPenalty);
          else
              costList.push_back(1);
      }
      else {
          for (int j = segmentStartTimes[i]; j < segmentStartTimes[i+1]; j++)
          {
              capture.set(CV_CAP_PROP_POS_FRAMES, j*frameRate);
              capture >> frame;
              cv::resize(frame, frame, cv::Size(frame.cols*resizeParam, frame.rows*resizeParam));
              CurrVideo.push_back(frame.clone());
          }
          cv::MatND hist;
          colorHist(CurrVideo, 0, CurrVideo.size() - 1, hist);
          snippetHist.push_back(hist);
          costList.push_back(CurrVideo.size());
          CurrVideo.clear();
      }
    }
    capture.release();
    n = snippetHist.size();
    capture.release();
}

void SimpleVideoSummarizer::computeKernel(int compare_method, double gamma)
{
    // compare_method is the comparision method for histogram similarity
    // gamma is a power of the similarity function: s_{ij} = sim(H_i, H_j)^{\gamma}
    float max = 0;
    for (int i = 0; i < n; i++)
    {
        std::vector<float> currvector;
        for (int j = 0; j < n; j++)
        {
            float val = cv::compareHist(snippetHist[i], snippetHist[j], compare_method);
            if (max < val)
                max = val;
            currvector.push_back(val);
        }
        kernel.push_back(currvector);
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if ((compare_method == 0) || (compare_method == 2) )
                kernel[i][j] = kernel[i][j]/max;
            else if ((compare_method == 1) || (compare_method == 3) )
                kernel[i][j] = (max - kernel[i][j])/max;
            if (compare_method == 0)
                kernel[i][j] = (kernel[i][j] + 1)/2;
            kernel[i][j] = pow(kernel[i][j], gamma);
        }
    }
}

void SimpleVideoSummarizer::summarizeBudget(int budget){
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

void SimpleVideoSummarizer::summarizeStream(double epsilon){
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

void SimpleVideoSummarizer::summarizeCover(double coverage){
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

void SimpleVideoSummarizer::playAndSaveSummaryVideo(char* videoFileSave)
{
    cv::VideoCapture capture(videoFile);
    cv::Mat frame;
    capture.set(CV_CAP_PROP_POS_FRAMES, 0);
    cv::VideoWriter videoWriter;
    if (videoFileSave != "")
      videoWriter = cv::VideoWriter(videoFileSave, CV_FOURCC('M','J','P','G'), (int) capture.get(CV_CAP_PROP_FPS),
        cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH),capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
    for (std::set<int>::iterator it = summarySet.begin(); it != summarySet.end(); it++)
    {
        capture.set(CV_CAP_PROP_POS_FRAMES, segmentStartTimes[*it]*frameRate);
        for (int i = segmentStartTimes[*it]; i < segmentStartTimes[*it+1]; i++)
        {
              for (int j = 0; j < frameRate; j++)
              {
                  capture >> frame;
                  cv::putText(frame, "Time: " + IntToString(i) + " seconds", cvPoint(30,30),
                      cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
                  if (frame.data)
                      cv::imshow("Sumamry Video", frame);
                  if (videoFileSave != "")
                    videoWriter.write(frame);
                  // Press  ESC on keyboard to exit
                  char c = (char)cv::waitKey(25);
                  if(c==27)
                      break;
              }
        }
    }
    capture.release();
}

void SimpleVideoSummarizer::displayAndSaveSummaryMontage(char* imageFileSave, int image_size)
{
    int summary_x = ceil(sqrt(summarySet.size()));
    int summary_y = ceil(summarySet.size()/summary_x);
    std::vector<cv::Mat> summaryimages = std::vector<cv::Mat>();
    cv::VideoCapture capture(videoFile);
    cv::Mat frame;
    capture.set(CV_CAP_PROP_POS_FRAMES, 0);
    for (std::set<int>::iterator it = summarySet.begin(); it != summarySet.end(); it++)
    {
        capture.set(CV_CAP_PROP_POS_FRAMES, segmentStartTimes[*it]*frameRate);
        capture >> frame;
        summaryimages.push_back(frame);
    }
    capture.release();
    cv::Mat collagesummary = cv::Mat(image_size*summary_y,image_size*summary_x,CV_8UC3);
    tile(summaryimages, collagesummary, summary_x, summary_y);
    cv::imshow("Summary Collage",collagesummary);
    if (imageFileSave != "")
        cv::imwrite(imageFileSave, collagesummary);
    char c = (char)cv::waitKey(0);
    if(c==27)
        return;
}
