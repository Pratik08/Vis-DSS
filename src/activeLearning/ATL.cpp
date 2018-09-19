#include "ATL.h"

bool sortByUncertainty(const std::pair<int,double> &a, const std::pair<int,double> &b) {
    return a.second > b.second;
}

ATL::ATL(std::vector<std::vector <float>> trainingFeatureVectors, jensen::Vector trainingIntLabels, std::vector<std::vector<float>> unlabeledTrainingFeatureVectors, jensen::Vector unlabeledTrainingIntLabels, std::vector<std::pair<int, std::string>> trainingStringToIntLabels, std::vector<std::vector<float>> testingFeatureVectors, jensen::Vector testingIntLabels, std::vector<std::pair<int, std::string>> testingStringToIntLabels, int beta, int b, int uncertaintyMode, int subsetSelectionMode, std::string csvFilePath){
  this->trainingFeatureVectors = trainingFeatureVectors;
  this->trainingIntLabels = trainingIntLabels;
  this->unlabeledTrainingFeatureVectors = unlabeledTrainingFeatureVectors;
  this->unlabeledTrainingIntLabels = unlabeledTrainingIntLabels;
  this->trainingStringToIntLabels = trainingStringToIntLabels;
  this->testingFeatureVectors = testingFeatureVectors;
  this->testingIntLabels = testingIntLabels;
  this->testingStringToIntLabels = testingStringToIntLabels;
  this->beta = beta;
  this->b = b;
  this->uncertaintyMode = uncertaintyMode;
  this->subsetSelectionMode = subsetSelectionMode;
  this->csvFilePath = csvFilePath;
}

ATL::~ATL(){}

void ATL::sparsifyFeatures(std::vector<float> &featureVector, jensen::SparseFeature &s) {
    s.featureIndex = std::vector<int>();
    s.featureVec = std::vector<double>();
    s.index = 0;
    for (int i = 0; i < featureVector.size(); i++) {
        if (featureVector[i] != 0) {
            s.featureIndex.push_back(i);
            s.featureVec.push_back(featureVector[i]);
        }
    }
    s.numFeatures = featureVector.size();
    s.numUniqueFeatures = s.featureVec.size();
}

double ATL::getUncertainty(jensen::Vector &predictions, int mode) {
    double uncertainty;
    switch(mode) {
        case 0: //Do arg max
        {
            int index = jensen::argMax(predictions);
            uncertainty = 1 - predictions[index];
            break;
        }
        case 1:
        {
            jensen::Vector predCopy = predictions;
            int maxIndex = jensen::argMax(predCopy);
            double maxProbability = predCopy[maxIndex];
            predCopy.erase(predCopy.begin() + maxIndex);
            double secondMax = predCopy[jensen::argMax(predCopy)];
            double difference = maxProbability - secondMax;
            uncertainty = 1 - difference;
            break;
        }
        case 2:
        {
            // Use entropy
            // std::cout << "Using Entropy" << std::endl;
            uncertainty = 0;
            // std::cout << "Iterating Entropy" << std::endl;
            for (jensen::Vector::const_iterator it = predictions.begin(); it != predictions.end(); it++) {
                if (*it != 0) {
                    uncertainty = uncertainty
                        + (*it * log2(*it));
                    // std::cout << "Using Entropy calc" << uncertainty << std::endl;
                    // std::cout << "Probability " << *it <<
                    //     "changes entropy by " << (*it * log2(*it)) << std::endl;
                }
            }
            // std::cout << "Finished iterating Entropy" << std::endl;
            uncertainty = 0 - uncertainty;
        }
        break;
    }
    return uncertainty;
}
Set ATL::getBetaUncertainIndices(std::vector<std::vector<float>> &unlabelledFeatureVectors,int beta) {
    std::vector<std::pair<int, double>> predictionUncertainties;
    int index = 0;
    for (std::vector<std::vector<float>>::iterator it = unlabelledFeatureVectors.begin();
        it < unlabelledFeatureVectors.end();
        it++, index++) {
        jensen::SparseFeature s;
        sparsifyFeatures(*it, s);
        jensen::Vector predictions;
        this->model->predictProbability(s, predictions);
        double uncertainty = getUncertainty(predictions, this->uncertaintyMode);
        std::cout << "Uncertainty is " << uncertainty << std::endl;
        predictionUncertainties.push_back(std::make_pair(index,
            uncertainty));
    }
    std::sort(predictionUncertainties.begin(), predictionUncertainties.end(), sortByUncertainty);
    Set uncertainIndices;
    for (int i = 0; i < beta; i++) {
        uncertainIndices.insert(predictionUncertainties[i].first);
    }
    float lastUncertainty = predictionUncertainties[beta - 1].second;
    for (int i = beta; i < predictionUncertainties.size(); i++) {
        if (predictionUncertainties[i].second < lastUncertainty) {
            break;
        }
        uncertainIndices.insert(predictionUncertainties[i].first);
    }
    std::cout << "Asked for "
        << beta
        << " uncertain samples but gave "
        << uncertainIndices.size()
        << std::endl;
    return uncertainIndices;
}

std::vector<double> ATL::predictAccuracy(
    std::vector<jensen::SparseFeature>& testFeatures,
    jensen::Vector& ytest) {
    // assert(testFeatures.size() == ytest.size());
    std::vector<double> accuracies;
    double accuracy = 0;
    double top5Accuracy = 0;
    for (int i = 0; i < testFeatures.size(); i++) {
        jensen::Vector predictions;
        this->model->predictProbability(testFeatures[i], predictions);
        // int prediction = this->model->predict(s);
        // if (prediction == -1)
        //     prediction = 0;

        std::cout << "--all predictions--\n";
        for(int j=0; j < predictions.size(); j++) {
            std::cout << j << " : " << predictions[j] << std::endl << std::flush;
        }
        std::cout << "---------\n";

        int predictionMax = jensen::argMax(predictions);
        std::vector<int> top5Predictions = top5(predictions);
        if (predictions.size() == 2) {
            if (predictionMax == 0) {
                predictionMax = -1;
            }
        }
        // std::cout << "Predicted " << ytest[i] << " as " << predictionMax
        //     << " with probability " << predictionMax << std::endl << std::flush;

        // Calculating top 1 accuracy
        if (predictions[predictionMax] >= this->LRL2PredictionProbThresh) {
            if (predictionMax == ytest[i]) {
                accuracy++;
            }
        }
        else if (ytest[i] == -2) {
            accuracy++;
        }

        // Calculating top 5 accuracy
        for (int k = 0; k < 5; k++) {
          if (predictions[top5Predictions[k]] >= this->LRL2PredictionProbThresh) {
              if (top5Predictions[k] == ytest[i]) {
                  top5Accuracy++;
                  break;
              }
          }
          else if (ytest[i] == -2) {
              top5Accuracy++;
              break;
          }
        }
    }
    accuracies.push_back(accuracy);
    accuracies.push_back(top5Accuracy);
    return accuracies;
}

std::vector<int> ATL::top5(jensen::Vector x){
    std::vector<int> maxIndices;
    for (int i = 0; i < 5; i++) {
      double maxVal = 0.0;
      int maxIndex = -1;
      for (int i = 0; i < x.size(); i++) {
        if (maxVal < x[i]) {
          maxVal = x[i];
          maxIndex = i;
        }
      }
      maxIndices.push_back(maxIndex);
      x[maxIndex] = 0;
    }
    return maxIndices;
}
Set ATL::getBIndicesFacilityLocation(std::vector<std::vector<float>> &featureVectors, jensen::Vector featureIntLabels,
    double budget,
    Set &subsetIndices) {
    //for each feature vector i (0 to n), compute its dot product with feature vector j (0 to n), update min, update max, store at k(i,j)
    Set indices;
    double val = 0;
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();
    std::map<int,int> indexMapping;
    int counter = 0;
    std::vector<std::vector<float>> kernel;
    for (int i=0; i < featureVectors.size(); i++) {
        if(!subsetIndices.contains(i)) {
            continue;
        }
        indexMapping[counter++] = i;
        std::vector<float> currvector;
        for (int j=0; j<featureVectors.size(); j++) {
            if(!subsetIndices.contains(j)) {
                continue;
            }
            std::vector<double> A(featureVectors.at(i).begin(),featureVectors.at(i).end());
            std::vector<double> B(featureVectors.at(j).begin(),featureVectors.at(j).end());
            if(featureIntLabels[i] == featureIntLabels[j]){
              val = jensen::innerProduct(A,B);
            }
            if(val < min) min = val;
            if(val > max) max = val;
            currvector.push_back(val);
        }
        kernel.push_back(currvector);
    }
    //0-1 normalize using min and max - this becomes the kernel
    double range = max-min;
    for (int i=0; i < kernel.size(); i++) {
        for (int j=0; j < kernel.size(); j++) {
            kernel[i][j] = (kernel[i][j]-min)/range;
        }
    }
    //instantiate facility location using kernel
    FacilityLocation facLoc(kernel.size(), kernel);
    //call lazygreedymax with this facility location fuction, B, null output set
    lazyGreedyMax(facLoc, budget, indices);

    Set scaledIndices;
    for (Set::iterator it = indices.begin();
        it != indices.end();
        it++) {
        scaledIndices.insert(indexMapping[*it]);
    }
    return scaledIndices;
}

Set ATL::getBIndicesDisparityMin(std::vector<std::vector<float>> &featureVectors, jensen::Vector featureIntLabels,
    double budget,
    Set &subsetIndices) {
    //for each feature vector i (0 to n), compute its dot product with feature vector j (0 to n), update min, update max, store at k(i,j)
    Set indices;
    double val = 0;
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();
    std::vector<std::vector<float>> kernel;
    std::map<int,int> indexMapping;
    int counter = 0;
    for (int i=0; i < featureVectors.size(); i++) {
        if(!subsetIndices.contains(i)) {
            continue;
        }
        indexMapping[counter++] = i;
        std::vector<float> currvector;
        for (int j=0; j<featureVectors.size(); j++) {
            if(!subsetIndices.contains(j)) {
                continue;
            }
            std::vector<double> A(featureVectors.at(i).begin(),featureVectors.at(i).end());
            std::vector<double> B(featureVectors.at(j).begin(),featureVectors.at(j).end());
            if(featureIntLabels[i] == featureIntLabels[j]){
              val = jensen::innerProduct(A,B);
            }
            if(val < min) min = val;
            if(val > max) max = val;
            currvector.push_back(val);
        }
        kernel.push_back(currvector);
    }
    //0-1 normalize using min and max - this becomes the kernel
    double range = max-min;
    for (int i=0; i < kernel.size(); i++) {
        for (int j=0; j < kernel.size(); j++) {
            kernel[i][j] = (kernel[i][j]-min)/range;
        }
    }
    //instantiate facility location using kernel
    DisparityMin dispMin(kernel.size(), kernel);
    //call lazygreedymax with this facility location fuction, B, null output set
    naiveGreedyMax(dispMin, budget, indices, 0, false, true);
    Set scaledIndices;
    for (Set::iterator it = indices.begin();
        it != indices.end();
        it++) {
        scaledIndices.insert(indexMapping[*it]);
    }
    return scaledIndices;
}

Set ATL::getBIndicesUncertaintySampling(int budget, Set &subsetIndices) {
    Set indices;
    Set::iterator it;
    int i =0;
    for (it = subsetIndices.begin(); it != subsetIndices.end(); it++) {
        if (i == budget)
            break;
        indices.insert(*it);
        i++;
    }
    return indices;
}

Set ATL::getBIndicesRandom(int n,
    double budget,
    Set subsetIndices) {
    //for each feature vector i (0 to n), identify random featureVectors and populate indices
    Set indices;
    std::vector<int> possibilities;
    for (int i = 0; i < n; i++) {
        if (subsetIndices.contains(i)) {
            possibilities.push_back(i);
        }
    }
    //auto engine = std::default_random_engine{};
    std::random_shuffle(possibilities.begin(), possibilities.end());
    for (int i = 0; i < budget; i++) {
        indices.insert(possibilities[i]);
    }
    return indices;
}

void ATL::train(int *numCorrect,
    int *numTotal, int *top5numCorrect, int *top5numTotal){
  std::cout << "creating sparse feature vector" << std::endl;
  std::vector<jensen::SparseFeature> sparseFeatures = std::vector<jensen::SparseFeature>();
  for (int i = 0; i < this->trainingFeatureVectors.size(); i++) {
      std::cout << "feature: " << this->trainingFeatureVectors[i].size() << " - label: "
          << this->trainingIntLabels[i] << std::endl;
      jensen::SparseFeature s;
      s.featureIndex = std::vector<int>();
      s.featureVec = std::vector<double>();
      s.index = i;
      for (int j = 0; j < this->trainingFeatureVectors[i].size(); j++) {
          if (this->trainingFeatureVectors[i][j] != 0) {
              s.featureIndex.push_back(j);
              s.featureVec.push_back(this->trainingFeatureVectors[i][j]);
          }
      }
      s.numFeatures = this->trainingFeatureVectors[i].size();
      s.numUniqueFeatures = s.featureVec.size();
      sparseFeatures.push_back(s);
  }
  std::cout << "sparseFeatures.size()" << sparseFeatures.size() << std::endl << std::flush;

  if (this->trainingStringToIntLabels.size() == 2 && !(this->trainConvertLabels)) {
      for (int i = 0; i < this->trainingIntLabels.size(); i++) {
          this->trainingIntLabels[i] = 2 * this->trainingIntLabels[i] - 1;
      }
      this->trainConvertLabels = true;
  }
  this->model = new jensen::L2LogisticRegression<jensen::SparseFeature>(sparseFeatures,
      this->trainingIntLabels,
      this->trainingFeatureVectors[0].size(),
      this->trainingFeatureVectors.size(),
      this->trainingStringToIntLabels.size(),
      this->LRL2RegulatizationParam,
      this->LRL2OptimizationAlgo,
      this->LRL2NumOfIterations,
      this->LRL2Tolerance);
      std::cout << "ATL: calling train" << std::endl;
      this->model->train();
      std::cout << "ATL: save" << std::endl;
      this->model->saveModel(LRL2SaveModelPath);

      std::cout << "ATL: writing labels to labels.txt" << std::endl;
      std::ofstream labelFile;
      labelFile.open(labelFilePath,
          std::ofstream::out | std::ofstream::trunc);

      for (int i = 0; i < this->trainingStringToIntLabels.size(); i++) {
          labelFile << this->trainingStringToIntLabels[i].second << "\n";
      }

      labelFile.close();
      std::cout << "ATL: Calling predictAccuracy" << std::endl;
      std::vector<double> accuracies = this->predictAccuracy(sparseFeatures, this->trainingIntLabels);
      double accuracy = accuracies[0];
      double accuracy_percentage = accuracy/this->trainingIntLabels.size();
      if (numCorrect != NULL) {
          *numCorrect = accuracy;
      }
      if (numTotal != NULL) {
          *numTotal = this->trainingIntLabels.size();
      }
      std::cout << "The Top 1 train acuracy of the classifier is " << accuracy_percentage << "("
          << accuracy << "/"<< this->trainingIntLabels.size() << ")" << std::endl;

      double top5Accuracy = accuracies[1];
      double top5Accuracy_percentage = top5Accuracy/this->trainingIntLabels.size();
      if (top5numCorrect != NULL) {
          *top5numCorrect = top5Accuracy;
      }
      if (top5numTotal != NULL) {
          *top5numTotal = this->trainingIntLabels.size();
      }
      std::cout << "The Top 5 train acuracy of the classifier is " << top5Accuracy_percentage << "("
          << top5Accuracy << "/"<< this->trainingIntLabels.size() << ")" << std::endl;

}

void ATL::test(int *numCorrect,
    int *numTotal, int *top5numCorrect, int *top5numTotal) {
      std::cout << "creating sparse feature vector" << std::endl;
      std::vector<jensen::SparseFeature> sparseFeatures = std::vector<jensen::SparseFeature>();
      for (int i = 0; i < this->testingFeatureVectors.size(); i++) {
          std::cout << "feature: " << this->testingFeatureVectors[i].size() << " - label: "
              << this->testingIntLabels[i] << std::endl;
          jensen::SparseFeature s;
          s.featureIndex = std::vector<int>();
          s.featureVec = std::vector<double>();
          s.index = i;
          for (int j = 0; j < this->testingFeatureVectors[i].size(); j++) {
              if (this->testingFeatureVectors[i][j] != 0) {
                  s.featureIndex.push_back(j);
                  s.featureVec.push_back(this->testingFeatureVectors[i][j]);
              }
          }
          s.numFeatures = this->testingFeatureVectors[i].size();
          s.numUniqueFeatures = s.featureVec.size();
          sparseFeatures.push_back(s);
      }

      if (this->testingStringToIntLabels.size() == 2 && !(this->testConvertLabels)) {
          for (int i = 0; i < this->testingIntLabels.size(); i++) {
              this->testingIntLabels[i] = 2 * this->testingIntLabels[i] - 1;
          }
          this->testConvertLabels = true;
      }
      std::vector<double> accuracies = predictAccuracy(sparseFeatures, this->testingIntLabels);
      double accuracy = accuracies[0]; // Top 1 test accuracy
      double accuracy_percentage = accuracy/this->testingIntLabels.size();
      if (numCorrect != NULL) {
          *numCorrect = accuracy;
      }
      if (numTotal != NULL) {
          *numTotal = this->testingIntLabels.size();
      }
      cout << "The Top 1 test acuracy of the classifier is " << accuracy_percentage << "(" << accuracy
          << "/" << this->testingIntLabels.size() << ")" << std::endl;

      double top5accuracy = accuracies[1]; // Top 5 test accurac
      double top5accuracy_percentage = top5accuracy/this->testingIntLabels.size();
      if (top5numCorrect != NULL) {
          *top5numCorrect = top5accuracy;
      }
      if (top5numTotal != NULL) {
          *top5numTotal = this->testingIntLabels.size();
      }
      cout << "The Top 5 test acuracy of the classifier is " << top5accuracy_percentage << "(" << top5accuracy
          << "/" << this->testingIntLabels.size() << ")" << std::endl;
}


 int ATL::run(int T){
  std::ofstream csvFile;
  csvFile.open(this->csvFilePath, std::ios_base::app);
  for (int t = 1; t <= T; t++) {
    std::cout << "___________________________________________________________________" << std::endl;
    std::cout << "DEBUG Trainingfeatvector.size() -> seed vectors " << this->trainingFeatureVectors.size() << std::endl;
    std::cout << "DEBUG TrainingIntLabels.size() " << this->trainingIntLabels.size() << std::endl;
    std::cout << "DEBUG trainingStringToIntLabels.size() " << this->trainingStringToIntLabels.size() << std::endl;
    std::cout << "DEBUG unlabeledTrainingFeatureVectors.size() -> unlabelled vectors, i.e full - seed " << this->unlabeledTrainingFeatureVectors.size() << std::endl;
    int *numCorrect = (int *) malloc(sizeof(int));
    int *numTotal = (int *) malloc(sizeof(int));
    int *top5numCorrect = (int *) malloc(sizeof(int));
    int *top5numTotal = (int *) malloc(sizeof(int));
    train();
    test(numCorrect, numTotal, top5numCorrect, top5numTotal);
    // Reporting Top 1 test accuracy at epoch T
    csvFile << ",=" << *numCorrect << "/" << *numTotal << std::flush;
    // Reporting Top 5 test accuracy at epoch T
    csvFile << ",=" << *top5numCorrect << "/" << *top5numTotal << std::endl << std::flush;
    std::cout << t << ": numCorrect = " << *numCorrect << " numTotal = " << *numTotal << std::endl;
    std::cout << t << ": top5numCorrect = " << *top5numCorrect << " top5numTotal = " << *top5numTotal << std::endl;
    if (this->unlabeledTrainingFeatureVectors.size() == 0) {
        break;
    }
    int maxBeta = beta;
    if (beta > this->unlabeledTrainingFeatureVectors.size() ) {
        maxBeta = this->unlabeledTrainingFeatureVectors.size();
    }
    int maxB = b;
    if (b > this->unlabeledTrainingFeatureVectors.size() ) {
        maxB = this->unlabeledTrainingFeatureVectors.size();
    }
    Set uncertainIndices = getBetaUncertainIndices(this->unlabeledTrainingFeatureVectors, maxBeta);
    Set indicesToTrain;
    switch(this->subsetSelectionMode) {
      case 0:{
          std::cout << "Using facility location" << std::endl;
          indicesToTrain = getBIndicesFacilityLocation(
              this->unlabeledTrainingFeatureVectors,
              this->unlabeledTrainingIntLabels,
              maxB,
              uncertainIndices);
          break;
      }
      case 1:{
          std::cout << "Using Disparity Min" << std::endl;
          indicesToTrain = getBIndicesDisparityMin(
              this->unlabeledTrainingFeatureVectors,
              this->unlabeledTrainingIntLabels,
              maxB,
              uncertainIndices);
          break;
      }
      case 2:{
          std::cout
              << "Using Uncertainty Sampling"
              << std::endl;
          indicesToTrain = getBIndicesUncertaintySampling(
              maxB,
              uncertainIndices);
          break;
      }
      case 3:{
          std::cout << "Using random mode" << std::endl;
          Set allIndices;
          for (int i = 0; i < unlabeledTrainingFeatureVectors.size(); i++) {
              allIndices.insert(i);
          }
          indicesToTrain = getBIndicesRandom(
              unlabeledTrainingFeatureVectors.size(),
              maxB,
              allIndices);
          break;
      }
      default:
          std::cout << "no submodular function found" << std::endl;
          return -1;
    }
    Set::iterator it;
    std::vector<int> indiciesToDelete;
    for (it = indicesToTrain.begin(); it != indicesToTrain.end(); ++it) {
      if (this->trainingStringToIntLabels.size() == 2){
        this->trainingIntLabels.push_back(2 * this->unlabeledTrainingIntLabels[*it] - 1);
      }
      else{
        this->trainingIntLabels.push_back(this->unlabeledTrainingIntLabels[*it]);
      }
        this->trainingFeatureVectors.push_back(this->unlabeledTrainingFeatureVectors[*it]);
        indiciesToDelete.push_back(*it);
    }
    std::sort(indiciesToDelete.begin(), indiciesToDelete.end(), std::greater<int>());
    for(int i = 0; i < indiciesToDelete.size(); i++) {
        this->unlabeledTrainingFeatureVectors.erase(this->unlabeledTrainingFeatureVectors.begin()+indiciesToDelete[i]);
        this->unlabeledTrainingIntLabels.erase(this->unlabeledTrainingIntLabels.begin()+indiciesToDelete[i]);
    }
    std::cout << "t b budget mode " << t << " " << b << " " << beta << " "  << std::endl;
    std::cout << "___________________________________________________________________" << std::endl;
  }
  int *numCorrectFinal = (int *) malloc(sizeof(int));
  int *numTotalFinal = (int *) malloc(sizeof(int));
  int *top5numCorrectFinal = (int *) malloc(sizeof(int));
  int *top5numTotalFinal = (int *) malloc(sizeof(int));
  train();
  test(numCorrectFinal, numTotalFinal, top5numCorrectFinal, top5numTotalFinal);

  // Reporting final Top 1 accuracy
  std::cout << ": numCorrectFinal = " << *numCorrectFinal << " numTotalFinal = " << *numTotalFinal << std::endl;
  csvFile << ",=" << *numCorrectFinal << "/" << *numTotalFinal << std::flush;

  // Reporting final Top 5 accuracy
  std::cout << ": top5numCorrectFinal = " << *top5numCorrectFinal << " top5numTotalFinal = " << *top5numTotalFinal << std::endl;
  csvFile << ",=" << *top5numCorrectFinal << "/" << *top5numTotalFinal << std::endl << std::flush;

  csvFile.close();
  return 0;
}
