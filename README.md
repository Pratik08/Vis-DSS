# IVSUMM: Image and Video Summarization Library

## License
IVSUMM is Licensed under the GNU GENERAL PUBLIC LICENSE. See LICENSE for more details.

## Features and Functionalities
1) Video Summarization
- Simple Video Summarization (using Color Histogram features)
- Deep Video Summarization using Features from a Deep Model and Similarity based functions
- Deep Video Summarization using Features from a Deep Model and Coverage Based Functions

2) Image Collection Summarization
- Simple Video Summarization (using Color Histogram features)
- Deep Video Summarization using Features from a Deep Model and Similarity based functions
- Deep Video Summarization using Features from a Deep Model and Coverage Based Functions

3) Entity Based Summarization
- Summarize all the Entities (Faces, Objects) in a Video

## Summarization Models
- Facility Location Functions (Representation Models)
- Disparity Min and Disparity Sum (Diversity Models)
- Set Cover and Probabilistic Set Cover (Coverage Models)
- Feature Based Functions
- Graph Cut and Saturated Coverage Functions

## Example commands to run the executables:

DisparityMin with Budgeted Summarization (SimpleVideoSummarizer: Using Color Histogram)
./SimpleVSExample -videoFile ../videos/friends.mp4 -videoSaveFile ../videos/friends_summary_DM30.avi -summaryModel 0 -segmentType 0 -summaryAlgo 0 -budget 30

Facility Location with Budgeted Summarization: (SimpleVideoSummarizer: Using Color Histogram)
./SimpleVSExample -videoFile ../videos/friends.mp4 -videoSaveFile ../videos/friends_summary_FL30.avi -summaryModel 2 -segmentType 0 -summaryAlgo 0 -budget 30

DisparityMin with Budgeted Summarization (DeepVideoSummarizer: Using GoogleNet Scene Model)
./DeepVSExample -videoFile ../videos/friends.mp4 -videoSaveFile ../videos/friends_summary_DMDeep.avi -summaryModelSim 0 -simcover 0 -segmentType 0 -summaryAlgo 0 -featureLayer loss3/classifier -network_file ../../Models/googlenet_places205/deploy_places205.protxt -trained_file ../../Models/googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel -mean_file ../../Models/hybridCNN/hybridCNN_mean.binaryproto -label_file ../../Models/googlenet_places205/categoryIndex_places205.csv -budget 30

Facility Location with Budgeted Summarization (DeepVideoSummarizer: Using GoogleNet Scene Model)
./DeepVSExample -videoFile ../videos/friends.mp4 -videoSaveFile ../videos/friends_summary_DMDeep.avi -summaryModelSim 2 -simcover 0 -segmentType 0 -summaryAlgo 0 -featureLayer loss3/classifier -network_file ../../Models/googlenet_places205/deploy_places205.protxt -trained_file ../../Models/googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel -mean_file ../../Models/hybridCNN/hybridCNN_mean.binaryproto -label_file ../../Models/googlenet_places205/categoryIndex_places205.csv -budget 30
