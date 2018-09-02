#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "caffeClassifier.h"
#include "arguments.h"

char* imageFile;
char* network_file;
char* trained_file;
char* mean_file;
char* label_file;
char* help;

Arg Arg::Args[] = {
    Arg("imageFile", Arg::Req, imageFile, "Input Image File", Arg::SINGLE),
    Arg("network_file", Arg::Req, network_file, "Input Network File", Arg::SINGLE),
    Arg("trained_file", Arg::Req, trained_file, "Trained Model File", Arg::SINGLE),
    Arg("mean_file", Arg::Req, mean_file, "Mean File", Arg::SINGLE),
    Arg("label_file", Arg::Req, label_file, "Label File", Arg::SINGLE),
    Arg("help", Arg::Help, help, "Print this message"),
    Arg()
};

int main(int argc, char** argv) {
    bool parse_was_ok = Arg::parse(argc, (char**)argv);
    if (!parse_was_ok) {
        Arg::usage(); exit(-1);
    }
    cv::Mat image;
    image = cv::imread(imageFile, CV_LOAD_IMAGE_COLOR);  // Read the file
    if (!image.data) {                            // Check for invalid input
        std::cout <<  "Could not open or find the image" << std::endl;
        return -1;
    }
    cv::namedWindow("Display window", WINDOW_AUTOSIZE);  // Create a window for display.
    CaffeClassifier c(network_file, trained_file, mean_file, label_file);
    std::vector<std::pair<std::string, float> > res = c.Classify(image);
    for (int i = 0; i < res.size(); i++) {
        std::cout << res[i].first << ": " << res[i].second << "\n";
    }
    cv::imshow("Display window", image);                  // Show our image inside it.
    cv::waitKey(0);                                         // Wait for a keystroke in the window
    return 0;
}
