/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#ifndef SUMMARIZER
#define SUMMARIZER

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <iostream>
#include "naiveGreedyMax.h"
#include "streamgreedy.h"
#include "lazyGreedyMax.h"


using namespace std;
using namespace cv;

int h_bins = 10;
int s_bins = 10;
int similarity_type = 0;

// Compute the average color histogram for frames between startframe and endframe
void colorHist(vector<Mat>& frames, int startframe, int endframe, MatND& hist)
{
    int histSize[] = { h_bins, s_bins };
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };
    MatND histcurr;
    /// Calculate the histograms for the HSV images
    calcHist( &frames[startframe], 1, channels, Mat(), hist, 2, histSize, ranges, true, false );
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat() );
    for (int j = startframe + 1; j <= endframe; j++){
        calcHist( &frames[j], 1, channels, Mat(), histcurr, 2, histSize, ranges, true, false );
        normalize(histcurr, histcurr, 0, 1, NORM_MINMAX, -1, Mat() );
        hist += histcurr;
    }
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat() );
}

void colorHist(Mat image, MatND& hist)
{
    int h_bins = 10;
    int s_bins = 10;
    int histSize[] = { h_bins, s_bins };
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };
    /// Calculate the histograms for the HSV images
    calcHist( &image, 1, channels, Mat(), hist, 2, histSize, ranges, true, false );
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat() );
}

vector<vector<float> > computeKernel(vector<MatND>& hist, int n, int compare_method = 0, double gamma = 1)
{
    // compare_method is the comparision method for histogram similarity
    vector<vector<float> > kernel;
    float max = 0;
    for (int i = 0; i < n; i++)
    {
        vector<float> currvector;
        for (int j = 0; j < n; j++)
        {
            float val = compareHist(hist[i], hist[j], compare_method);
            if (max < val)
                max = val;
            currvector.push_back(val);
        }
        kernel.push_back(currvector);
        emit player->progressBar(0.2*i + 1.6*n, 2*n);
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
        emit player->progressBar(1.8*n + 0.2*i, 2*n);
    }
    return kernel;
}

void cleanKernel(vector<vector<float> >& kernel, vector<MatND>& hist, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (kernel[i][j] == 1)
            {
                if ( (compareHist(hist[i], hist[j], 1) == 0) && (compareHist(hist[i], hist[j], 3) == 0) )
                {
                    kernel[i][j] = 1;
                }
                else
                    kernel[i][j] = 0.5;
            }
        }
    }
    return;
}

void summarize(vector<Mat>& snipetHist, int n, int budget, set<int>& summarySet){
    Set optSet;
    vector<vector<float> > kernel = computeKernel(player, snipetHist, n, similarity_type);
    cleanKernel(kernel, snipetHist, n);
    DisparityMin dM(n, kernel);
    int inititem = rand()%n;
    optSet.insert(inititem);
    naiveGreedyMax(dM, budget-1, optSet, 0, false, true);
    summarySet = set<int>();
    for (Set::iterator it = optSet.begin(); it!=optSet.end(); it++)
    {
        summarySet.insert(*it);
    }
    // cout << "Done with summarization\n" << flush;
}

void summarize(vector<Mat>& snipetHist, int n, double epsilon, set<int>& summarySet){
    Set optSet;
    vector<vector<float> > kernel = computeKernel(player, snipetHist, n, similarity_type);
    cleanKernel(kernel, snipetHist, n);
    MMR m(n, kernel);
    optSet.insert(0);
    vector<int> order(n, 1);
    for (int i = 0; i < n; i++)
        order[i] = i;
    streamGreedy(m, epsilon, optSet, order);
    optSet.insert(n-1);
    summarySet = set<int>();
    for (Set::iterator it = optSet.begin(); it!=optSet.end(); it++)
    {
        summarySet.insert(*it);
    }
    // cout << "Done with summarization\n" << flush;
}

void Player::extractFeaturesOne(){
    VideoCapture capture;
    capture.open(filenamep);
    Mat frame;
    if( !capture.isOpened() )
            return;
    int frame_count = 0;
    while (!stop)
    {
        MatND hist;
        capture.set(CV_CAP_PROP_POS_FRAMES, frame_count*frameRate);
        capture >> frame;
        if( (frame.empty()) || (frame_count*frameRate > capture.get(CV_CAP_PROP_FRAME_COUNT))){
            break;
        }
        resize(frame, frame, Size(frame.cols/4, frame.rows/4));
        colorHist(frame, hist);
        snippetHist.push_back(hist);
        emit progressBar(1.6*frame_count*frameRate, 2*capture.get(CV_CAP_PROP_FRAME_COUNT));
        frame_count++;
    }
    capture.release();
    n = snippetHist.size();
}

void Player::extractFeatures(){
    VideoCapture capture;
    capture.open(filenamep);
    Mat frame;
    vector<Mat> CurrVideo = vector<Mat> ();
    if( !capture.isOpened() )
            cout << "Error when reading steam" << "\n";
    int frame_count = 0;
    int samplingRate = 1;
    while (!stop)
    {
        MatND hist;
        if (frame_count % frameRate/samplingRate == 0){
            capture.set(CV_CAP_PROP_POS_FRAMES, frame_count);
            capture >> frame;
            if( (frame.empty()) || (frame_count > capture.get(CV_CAP_PROP_FRAME_COUNT))){
                break;
            }
            resize(frame, frame, Size(frame.cols/4, frame.rows/4));
            // imshow("Frame", frame);
            // cvWaitKey(20);
            // std::cout << frame_count << " " << capture.get(CV_CAP_PROP_FRAME_COUNT) << "\n" << flush;
            emit progressBar(1.6*frame_count, 2*capture.get(CV_CAP_PROP_FRAME_COUNT));
            CurrVideo.push_back(frame.clone());
        }
        if (frame_count % (SNP_SIZE*frameRate) == 0){
            colorHist(CurrVideo, 0, CurrVideo.size() - 1, hist);
            snippetHist.push_back(hist);
            CurrVideo.clear();
        }
        frame_count++;
    }
    capture.release();
    n = snippetHist.size();
}

#endif // SUMMARIZER
