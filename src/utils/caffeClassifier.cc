#include "caffeClassifier.h"

static bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i) {
        pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
    }
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i) {
        result.push_back(pairs[i].second);
    }
    return result;
}

CaffeClassifier::CaffeClassifier() {
    this->empty = true;
}

CaffeClassifier::CaffeClassifier(const std::string& network_file,
                                 const std::string& trained_file,
                                 const std::string& mean_file,
                                 const std::string& label_file) {
#ifdef CPU_ONLY
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

    this->net_.reset(new caffe::Net<float>(network_file, caffe::TEST));
    this->net_->CopyTrainedLayersFrom(trained_file);

    caffe::Blob<float>* input_layer = this->net_->input_blobs()[0];
    this->num_channels_ = input_layer->channels();
    this->input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    std::cout << "initialized caffe classifier" << std::endl;

    if (!mean_file.empty()) {
        this->SetMean(mean_file);
    }

    if (!label_file.empty()) {
        /* Load labels. */
        std::ifstream labels(label_file.c_str());
        CHECK(labels) << "Unable to open labels file " << label_file;
        std::string line;
        while (std::getline(labels, line)) {
            labels_.push_back(std::string(line));
        }

        caffe::Blob<float>* output_layer = this->net_->output_blobs()[0];
        CHECK_EQ(labels_.size(), output_layer->channels())
            << "Number of labels is different from the output layer dimension.";
    }
}

std::vector<std::pair<std::string, float> > CaffeClassifier::Classify(const cv::Mat& img,
                                                                      int N) {
    std::vector<float> output = this->Predict(img);

    N = std::min<int>(labels_.size(), N);
    std::vector<int> maxN = Argmax(output, N);
    std::vector<std::pair<std::string, float> > predictions;
    for (int i = 0; i < N; ++i) {
        int idx = maxN[i];
        std::cout << "CLASS: " << labels_[idx] << " PROB: " << output[idx] << std::endl << std::flush;
        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }
    return predictions;
}

std::vector<std::pair<std::string, float> > CaffeClassifier::Classify(const std::vector<cv::Mat> &imgvec,
                                                                      int N) {
    std::vector<float> output = this->Predict(imgvec);
    N = std::min<int>(labels_.size(), N);
    std::vector<int> maxN = Argmax(output, N);
    std::vector<std::pair<std::string, float> > predictions = std::vector<std::pair<std::string, float> >();
    for (int i = 0; i < N; ++i) {
        int idx = maxN[i];
         // std::cout << "CLASS: " << labels_[idx] << " PROB: " << output[idx] << std::endl << std::flush;
        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }
    return predictions;
}

/* Load the mean file in binaryproto format. */
void CaffeClassifier::SetMean(const std::string& mean_file) {
    caffe::BlobProto blob_proto;
    caffe::ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    caffe::Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
        << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; i++) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    this->mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> CaffeClassifier::getLayerOutput(std::string layer) {
    caffe::shared_ptr<caffe::Blob<float> > output_layer = this->net_->blob_by_name(layer);
    const float *begin = output_layer->cpu_data();
    const float *end = begin + output_layer->channels();

    return std::vector<float>(begin, end);;
}

std::vector<float> CaffeClassifier::Predict(const std::vector<cv::Mat>& imgvec, std::string layer, const bool &verbose) {
    std::vector<float> feat = Predict(imgvec[0], layer, verbose);
    for (int i = 1; i < imgvec.size(); i++) {
        std::vector<float> currfeat = Predict(imgvec[i], layer, verbose);
        for (int j = 0; j < feat.size(); j++) {
            feat[j] += currfeat[j];
        }
    }
    for (int j = 0; j < feat.size(); j++) {
        feat[j] = feat[j] / imgvec.size();
    }
    return feat;
}

std::vector<float> CaffeClassifier::Predict(const cv::Mat &img, std::string layer, const bool &verbose) {
#ifdef CPU_ONLY
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
    caffe::Blob<float>* input_layer = this->net_->input_blobs()[0];
    input_layer->Reshape(1, this->num_channels_,
                         this->input_geometry_.height,
                         this->input_geometry_.width);
    this->net_->Reshape();
    std::vector<cv::Mat> input_channels;
    this->WrapInputLayer(&input_channels);
    this->Preprocess(img, &input_channels);   // convert img to caffe input
     // std::cout << "Preprocessed" << std::endl;
    caffe::Timer tmr;
    tmr.Start();
    this->net_->Forward();   // Caffe functions
    tmr.Stop();
    if (verbose == true) {
        std::cout << "Feature extracton time = " << tmr.Seconds() << std::endl;
    }
    if (!layer.empty()) {
        return this->getLayerOutput(layer);
    }
    caffe::Blob<float> *output_layer = net_->output_blobs()[0];
    const float *begin = output_layer->cpu_data();
    const float *end = begin + output_layer->channels();
    return std::vector<float>(begin, end);
}

void CaffeClassifier::Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels) {
    cv::Mat sample;
    if (img.channels() == 3 && this->num_channels_ == 1) {
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4 && this->num_channels_ == 1) {
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    } else if (img.channels() == 4 && this->num_channels_ == 3) {
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    } else if (img.channels() == 1 && this->num_channels_ == 3) {
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    } else {
        sample = img;
    }
    cv::Mat sample_resized;
    if (sample.size() != this->input_geometry_) {
        cv::resize(sample, sample_resized, this->input_geometry_);
    } else {
        sample_resized = sample;
    }
    cv::Mat sample_float;
    if (this->num_channels_ == 3) {
        sample_resized.convertTo(sample_float, CV_32FC3);
    } else {
        sample_resized.convertTo(sample_float, CV_32FC1);
    }
    cv::split(sample_float, *input_channels);
    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == this->net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network. ";
}

void CaffeClassifier::WrapInputLayer(std::vector<cv::Mat> *input_channels) {
    caffe::Blob<float>* input_layer = this->net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

bool CaffeClassifier::isEmpty() {
    return this->empty;
}
