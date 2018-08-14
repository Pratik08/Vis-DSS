# IVSUMM: Image and Video Summarization Library

## License
IVSUMM is Licensed under the GNU GENERAL PUBLIC LICENSE. See LICENSE for more details.

## Features and Functionalities
1) Video Summarization
- `Simple Video Summarization` (using Color Histogram features)
- `Deep Video Summarization` using Features from a Deep Model and Similarity based functions
- `Deep Video Summarization` using Features from a Deep Model and Coverage Based Functions

2) Image Collection Summarization
- `Simple Video Summarization` (using Color Histogram features)
- `Deep Video Summarization` using Features from a Deep Model and Similarity based functions
- `Deep Video Summarization` using Features from a Deep Model and Coverage Based Functions

3) Entity Based Summarization
- Summarize all the Entities (Faces, Objects) in a Video. We use OpenCV DNN to extract Faces (using the ResNet model) and YOLO for Object detection.

## Summarization Models (-summaryModel)
- `Facility Location Functions` (Representation Models)
- `Disparity Min` and `Disparity Sum` (Diversity Models)
- `Set Cover` and `Probabilistic Set Cover` (Coverage Models)
- `Feature Based Functions`
- `Graph Cut` and `Saturated Coverage Functions`

## Summarization Algorithms (-summaryAlgo)
- `Budgeted Greedy Algorithm` (Lazy or naive greedy algorithm under a budget, say, 60 seconds)
- `Stream Greedy Algorithm` (Provide a threshold for summarization, say, 0.001)
- `Coverage Greedy Algorithm` (Provide a coverage fraction, say, 0.9 fraction of the video)

## Segment Type (-segmentType)
In the case of video summarization, we support two kinds of segmentation algorithms
- Fixed Lengh Snippets
- Shot Detection based Snippets

## Dependencies
- If you just want to compile and build `SimpleVideoSummarizer` and `SimpleImageSummarizer` examples, you only need OpenCV 3 (https://github.com/opencv/opencv)
- For running Deep Video Summarizer examples (with Caffe models), you will need to install Caffe (https://github.com/BVLC/caffe). You just need the CPU version of Caffe
- For running the Entity based Summarizers you might also need Dlib if you are using the feature extractor algorithms from dlib.

## Building and Compiling
- Modify the CMakeLists.txt to point to yout OpenCV and Caffe locations
- `mkdir build`
- `cd build`
- `cmake ..`
- `make`

## Example commands to run the executables:

SimpleVideoSummarizer: DisparityMin with Budgeted Summarization 
`./SimpleVSExample -videoFile <videoFileName> -videoSaveFile <videoSummaryFileName> -summaryModel 0 -segmentType 0 -summaryAlgo 0 -budget 30`

SimpleVideoSummarizer: Facility Location with Budgeted Summarization
`./SimpleVSExample -videoFile <videoFileName> -videoSaveFile <videoSummaryFileName> -summaryModel 2 -segmentType 0 -summaryAlgo 0 -budget 30`

DeepVideoSummarizer DisparityMin with Budgeted Summarization (Using GoogleNet Scene Model)
`./DeepVSExample -videoFile <videoFileName> -videoSaveFile <videoSummaryFileName> -summaryModelSim 0 -simcover 0 -segmentType 0 -summaryAlgo 0 -featureLayer loss3/classifier -network_file ../../Models/googlenet_places205/deploy_places205.protxt -trained_file ../../Models/googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel -mean_file ../../Models/hybridCNN/hybridCNN_mean.binaryproto -label_file ../../Models/googlenet_places205/categoryIndex_places205.csv -budget 30`

DeepVideoSummarizer: Facility Location with Budgeted Summarization (Using GoogleNet Scene Model)
`./DeepVSExample -videoFile <videoFileName> -videoSaveFile <videoSummaryFileName> -summaryModelSim 2 -simcover 0 -segmentType 0 -summaryAlgo 0 -featureLayer loss3/classifier -network_file ../../Models/googlenet_places205/deploy_places205.protxt -trained_file ../../Models/googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel -mean_file ../../Models/hybridCNN/hybridCNN_mean.binaryproto -label_file ../../Models/googlenet_places205/categoryIndex_places205.csv -budget 30`

DeepVideoSummarizer: SetCover with Budgeted Summarization (Using GoogleNet Scene Model)
`./DeepVSExample -videoFile <videoFileName> -videoSaveFile <videoSummaryFileName> -summaryModelSim 0 -simcover 1 -segmentType 0 -summaryAlgo 0 -featureLayer loss3/classifier -network_file ../../Models/googlenet_places205/deploy_places205.protxt -trained_file ../../Models/googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel -mean_file ../../Models/hybridCNN/hybridCNN_mean.binaryproto -label_file ../../Models/googlenet_places205/categoryIndex_places205.csv -budget 30`
