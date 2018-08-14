Example commands to run the executables:

DisparityMin with Budgeted Summarization (SimpleVideoSummarizer: Using Color Histogram)
./SimpleVSExample -videoFile ../videos/friends.mp4 -videoSaveFile ../videos/friends_summary_DM30.avi -summaryModel 0 -segmentType 0 -summaryAlgo 0 -budget 30

Facility Location with Budgeted Summarization: (SimpleVideoSummarizer: Using Color Histogram)
./SimpleVSExample -videoFile ../videos/friends.mp4 -videoSaveFile ../videos/friends_summary_FL30.avi -summaryModel 2 -segmentType 0 -summaryAlgo 0 -budget 30

DisparityMin with Budgeted Summarization (DeepVideoSummarizer: Using GoogleNet Scene Model)
./DeepVSExample -videoFile ../videos/friends.mp4 -videoSaveFile ../videos/friends_summary_DMDeep.avi -summaryModelSim 0 -simcover 0 -segmentType 0 -summaryAlgo 0 -featureLayer loss3/classifier -network_file ../../Models/googlenet_places205/deploy_places205.protxt -trained_file ../../Models/googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel -mean_file ../../Models/hybridCNN/hybridCNN_mean.binaryproto -label_file ../../Models/googlenet_places205/categoryIndex_places205.csv -budget 30

Facility Location with Budgeted Summarization (DeepVideoSummarizer: Using GoogleNet Scene Model)
./DeepVSExample -videoFile ../videos/friends.mp4 -videoSaveFile ../videos/friends_summary_DMDeep.avi -summaryModelSim 2 -simcover 0 -segmentType 0 -summaryAlgo 0 -featureLayer loss3/classifier -network_file ../../Models/googlenet_places205/deploy_places205.protxt -trained_file ../../Models/googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel -mean_file ../../Models/hybridCNN/hybridCNN_mean.binaryproto -label_file ../../Models/googlenet_places205/categoryIndex_places205.csv -budget 30
