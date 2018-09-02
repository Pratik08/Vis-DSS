/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#ifndef SRC_ENGINE_SPARSEFEATURE_H_
#define SRC_ENGINE_SPARSEFEATURE_H_

#include <vector>

struct SparseFeature {   // Stores the feature vector for each item in the groundset
    long int index;   // index of the item
    int numUniqueFeatures;   // number of non-zero enteries in the feature vector
    std::vector<int> featureIndex;   // Indices which are non-zero (generally sparse)
    std::vector<double> featureVec;   // score of the features present.
    int numFeatures;
};

SparseFeature getSparseFeature(std::vector<float>& vec);

#endif  // SRC_ENGINE_SPARSEFEATURE_H_
