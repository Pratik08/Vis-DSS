/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#include "SparseFeature.h"

SparseFeature getSparseFeature(std::vector<float>& vec) {
    SparseFeature s;

    s.featureIndex = std::vector<int>();
    s.featureVec = std::vector<double>();
    s.index = 0;
    for (int j = 0; j < vec.size(); j++) {
        if (vec[j] != 0) {
            s.featureIndex.push_back(j);
            s.featureVec.push_back(vec[j]);
        }
    }
    s.numFeatures = vec.size();
    s.numUniqueFeatures = s.featureVec.size();
    return s;
}
