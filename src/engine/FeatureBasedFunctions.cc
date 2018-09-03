/*
    Copyright (C) Rishabh Iyer
 *	Implementation of the feature based functions (essentially sums of concave over modular functions): f(X) = \sum_{f \in F} w_f g(m_f(X)), where g is a concave function, {m_f}_{f \in F} are a set of feature scores, and f \in F are features.
    Author: Rishabh Iyer.
 *
 */
#include "FeatureBasedFunctions.h"

FeatureBasedFunctions::FeatureBasedFunctions(int n, int type, std::vector<struct SparseFeature>& feats, std::vector<double>& featureWeights, double thresh) :
    SetFunctions(n), type(type), thresh(thresh), feats(feats), featureWeights(featureWeights), nFeatures(featureWeights.size()), sizepreCompute(featureWeights.size()) {
    for (std::vector<double>::const_iterator it = featureWeights.begin(); it != featureWeights.end(); it++) {
        if (*it < 0) {
            std::cout << "********************************************************************" << std::endl;
            std::cout << "Warning: the input feature weights has some entry being negative, please double check" << std::endl;
            std::cout << "********************************************************************" << std::endl;
        }
    }
    preCompute.resize(sizepreCompute);
}


FeatureBasedFunctions::FeatureBasedFunctions(const FeatureBasedFunctions & f) :
    SetFunctions(f.n), feats(f.feats), featureWeights(f.featureWeights), nFeatures(f.nFeatures), type(f.type), thresh(f.thresh), sizepreCompute(f.sizepreCompute) {
    preCompute = f.preCompute;
    preComputeSet = f.preComputeSet;
    // std::cout << "Called copy constructor " << std::endl;
}


FeatureBasedFunctions::~FeatureBasedFunctions() {
    // std::cout << "Call destructor of feature based function!" << std::endl;
}


FeatureBasedFunctions* FeatureBasedFunctions::clone() const {
    // std::cout << "called feature based function clone" << std::endl;
    return new FeatureBasedFunctions(*this);
}


double FeatureBasedFunctions::concaveFunction(double K) const {
    switch (type) {
    case 1:
        return sqrt(K);
    case 2:
        return (1 - 1 / (K + 1));
    case 3:
        return log(1 + K);
    case 4:
        return std::min(K, thresh);
    default:
        return sqrt(K);
    }
}


double FeatureBasedFunctions::eval(const Set& set) const {
    // Evaluation of function valuation.
    double sum = 0;

    std::vector<double> featurescoresset(nFeatures, 0);
    Set::const_iterator it;
    for (it = set.begin(); it != set.end(); ++it) {
        for (int j = 0; j < feats[*it].numUniqueFeatures; j++) {
            featurescoresset[feats[*it].featureIndex[j]] += feats[*it].featureVec[j];
        }
    }
    for (int ii = 0; ii < nFeatures; ii++) {
        // std::cout << this->featureWeights[ii] << std::endl;
        // std::cout << concaveFunction(featurescoresset[ii]) << std::endl;
        sum = sum + featureWeights[ii] * concaveFunction(featurescoresset[ii]);
    }
    return sum;
}


double FeatureBasedFunctions::evalFast(const Set& set) const {
    // Evaluation of function valuation.
    double sum = 0;
    for (int ii = 0; ii < nFeatures; ii++) {
        sum = sum + featureWeights[ii] * concaveFunction(preCompute[ii]);
    }
    return sum;
}


double FeatureBasedFunctions::evalGainsadd(const Set& set, int item) const {
    // Evaluation of gains.
    if (set.contains(item)) {
        std::cout << "Error in using evalGainsadd: the provided item already belongs to the subset\n";
        return 0;
    }
    double gains = 0;
    double temp;
    double diff;
    std::vector<double> featurescoresset(nFeatures, 0);
    Set::const_iterator it;
    for (it = set.begin(); it != set.end(); ++it) {
        for (int j = 0; j < feats[*it].numUniqueFeatures; j++) {
            featurescoresset[feats[*it].featureIndex[j]] += feats[*it].featureVec[j];
        }
    }
    for (int i = 0; i < feats[item].numUniqueFeatures; i++) {
        temp = featurescoresset[feats[item].featureIndex[i]];
        diff = concaveFunction(temp + feats[item].featureVec[i]) - concaveFunction(temp);
        gains += featureWeights[feats[item].featureIndex[i]] * diff;
    }
    return gains;
}


double FeatureBasedFunctions::evalGainsremove(const Set& set, int item) const {
    // Evaluation of function valuation.
    if (!set.contains(item)) {
        std::cout << "Error in using evalGainsremove: the provided item does not belong to the subset\n";
        return 0;
    }
    double sum = 0;
    double sumd = 0;
    std::vector<double> featurescoresset(nFeatures, 0);
    std::vector<double> featurescoressetd(nFeatures, 0);
    Set::const_iterator it;
    for (it = set.begin(); it != set.end(); ++it) {
        for (int j = 0; j < feats[*it].numUniqueFeatures; j++) {
            featurescoresset[feats[*it].featureIndex[j]] += feats[*it].featureVec[j];
            if (*it != item) {
                featurescoressetd[feats[*it].featureIndex[j]] += feats[*it].featureVec[j];
            }
        }
    }
    for (int ii = 0; ii < nFeatures; ii++) {
        sum = sum + featureWeights[ii] * concaveFunction(featurescoresset[ii]);
        sumd = sumd + featureWeights[ii] * concaveFunction(featurescoressetd[ii]);
    }
    return (sum - sumd);
}


double FeatureBasedFunctions::evalGainsaddFast(const Set& set, int item) const {
    // Fast evaluation of Adding gains using the precomputed statistics. This is used, for example, in the implementation of the forward greedy algorithm.
    double gains = 0;
    double temp;
    double diff;
    int num_wrds = feats[item].numUniqueFeatures;
    gains = 0;
    for (int i = 0; i < num_wrds; i++) {
        temp = preCompute[feats[item].featureIndex[i]];
        diff = concaveFunction(temp + feats[item].featureVec[i]) - concaveFunction(temp);
        gains += featureWeights[feats[item].featureIndex[i]] * diff;
    }
    return gains;
}


double FeatureBasedFunctions::evalGainsremoveFast(const Set& set, int item) const {
    // Fast evaluation of Removing gains using the precomputed statistics. This is used, for example, in the implementation of the reverse greedy algorithm.
    double gains = 0;
    double temp;
    double diff;
    int num_wrds = feats[item].numUniqueFeatures;
    gains = 0;
    for (int i = 0; i < num_wrds; i++) {
        temp = preCompute[feats[item].featureIndex[i]] - feats[item].featureVec[i];
        diff = concaveFunction(preCompute[feats[item].featureIndex[i]]) - concaveFunction(temp);
        gains += featureWeights[feats[item].featureIndex[i]] * diff;
    }
    return gains;
}


void FeatureBasedFunctions::updateStatisticsAdd(const Set& set, int item) const {
    // Update statistics for algorithms sequentially adding elements (for example, the greedy algorithm).
    for (int i = 0; i < feats[item].numUniqueFeatures; i++) {
        preCompute[feats[item].featureIndex[i]] += feats[item].featureVec[i];
    }
    preComputeSet.insert(item);
}


void FeatureBasedFunctions::updateStatisticsRemove(const Set& sset, int item) const {
    // Update statistics for algorithms sequentially removing elements (for example, the reverse greedy algorithm).
    for (int i = 0; i < feats[item].numUniqueFeatures; i++) {
        preCompute[feats[item].featureIndex[i]] -= feats[item].featureVec[i];
    }
    preComputeSet.remove(item);
}


void FeatureBasedFunctions::clearpreCompute() const {
    for (int i = 0; i < sizepreCompute; i++) {
        preCompute[i] = 0;
    }
    preComputeSet.clear();
}


void FeatureBasedFunctions::setpreCompute(const Set& set) const {
    clearpreCompute();
    Set::const_iterator it;
    for (it = set.begin(); it != set.end(); ++it) {
        updateStatisticsAdd(set, *it);
    }
    preComputeSet = set;
}
