/*
    Copyright (C) 2015 Rishabh Iyer
 *	Header file for defining the saturated coverage function: f(X) = \sum_{i \in V} \min\{\sum_{j \in X} s_{ij}, \alpha \sum_{j \in V} s_{ij}\}.
    Author: Rishabh Iyer.
 *
 */

#include "SaturateCoverage.h"

SaturateCoverage::SaturateCoverage(int n, const std::vector<std::vector <float> >& kernel, const double alpha) : SetFunctions(n), kernel(kernel), alpha(alpha), sizepreCompute(n) {
    preCompute.assign(sizepreCompute, 0);   // precomputed statistics for speeding up greedy
    if (alpha <= 0) {
        std::cout << "Cannot input alpha = " << alpha << " for the saturated coverage function to be smaller or equal to 0\n";
    }
    for (int i = 0; i < sizepreCompute; i++) {
        preCompute[i] = 0;
    }
    modularthresh.resize(n);
    for (int i = 0; i < n; i++) {
        modularthresh[i] = 0;
        for (int j = 0; j < n; j++) {
            modularthresh[i] += kernel[i][j];
        }
    }
}


SaturateCoverage::SaturateCoverage(const SaturateCoverage& f) : SetFunctions(f.n), kernel(f.kernel), alpha(f.alpha), sizepreCompute(f.sizepreCompute) {
    preCompute = f.preCompute;
    modularthresh = f.modularthresh;
    preComputeSet = f.preComputeSet;
}


SaturateCoverage::~SaturateCoverage() {
}


SaturateCoverage* SaturateCoverage::clone() const {
    return new SaturateCoverage(*this);
}


double SaturateCoverage::eval(const Set& sset) const {
    double sum = 0;
    double sumvali;
    Set::const_iterator it;
    for (int i = 0; i < n; i++) {
        sumvali = 0;
        for (it = sset.begin(); it != sset.end(); ++it) {
            sumvali += kernel[i][*it];
        }
        if (sumvali < alpha * modularthresh[i]) {
            sum += sumvali;
        } else {
            sum += alpha * modularthresh[i];
        }
    }
    return sum;
}


double SaturateCoverage::evalFast(const Set& sset) const {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        if (preCompute[i] < alpha * modularthresh[i]) {  // the ith component is not saturated?
            sum += preCompute[i];
        } else {
            sum += alpha * modularthresh[i];
        }
    }
    return sum;
}


double SaturateCoverage::evalGainsadd(const Set& sset, int item) const {
    if (sset.contains(item)) {
        std::cout << "Error in using evalGainsadd: the provided item already belongs to the subset\n";
        return 0;
    }
    double gains = 0;
    double sumvali;
    Set::const_iterator it;
    for (int i = 0; i < n; i++) {
        sumvali = 0;
        for (it = sset.begin(); it != sset.end(); ++it) {
            sumvali += kernel[i][*it];
        }
        if (sumvali + kernel[i][item] < alpha * modularthresh[i]) {
            gains += kernel[i][item];   // there is a bug here!!
        } else if (sumvali >= alpha * modularthresh[i]) {
            gains += 0;
        } else {
            gains += (alpha * modularthresh[i] - sumvali);
        }
    }
    return gains;
}


double SaturateCoverage::evalGainsremove(const Set& sset, int item) const {
    if (!sset.contains(item)) {
        std::cout << "Error in using evalGainsremove: the provided item does not belong to the subset\n";
        return 0;
    }
    double sum = 0;
    double sumd = 0;
    double sumvali;
    double sumvald;
    Set::const_iterator it;
    for (int i = 0; i < n; i++) {
        sumvali = 0;
        sumvald = 0;
        for (it = sset.begin(); it != sset.end(); ++it) {
            sumvali += kernel[i][*it];
            if (*it != item) {
                sumvald += kernel[i][*it];
            }
        }
        if (sumvali < alpha * modularthresh[i]) {
            sum += sumvali;
        } else {
            sum += alpha * modularthresh[i];
        }
        if (sumvald < alpha * modularthresh[i]) {
            sumd += sumvald;
        } else {
            sumd += alpha * modularthresh[i];
        }
    }
    return (sum - sumd);
}


double SaturateCoverage::evalGainsaddFast(const Set& sset, int item) const {
    // Fast evaluation of Adding gains using the precomputed statistics. This is used, for example, in the implementation of the forward greedy algorithm.
    double gains = 0;
    for (int i = 0; i < n; i++) {
        if ((preCompute[i] < alpha * modularthresh[i]) && (preCompute[i] + kernel[item][i] > alpha * modularthresh[i])) {   // adding item just saturates component i
            gains += (-preCompute[i] + alpha * modularthresh[i]);
        } else if (preCompute[i] + kernel[item][i] <= alpha * modularthresh[i]) {   // the ith component is not saturated?
            gains += kernel[item][i];
        }
    }
    return gains;
}


double SaturateCoverage::evalGainsremoveFast(const Set& sset, int item) const {
    // Fast evaluation of Removing gains using the precomputed statistics. This is used, for example, in the implementation of the reverse greedy algorithm.
    double gains = 0;
    for (int i = 0; i < n; i++) {
        if ((preCompute[i] >= alpha * modularthresh[i]) && (preCompute[i] - kernel[item][i] < alpha * modularthresh[i])) {  // removing item just unsaturates component i
            gains += kernel[item][i] - preCompute[i] + alpha * modularthresh[i];
        } else if (preCompute[i] < alpha * modularthresh[i]) {  // Component i is already unsaturated
            gains += kernel[item][i];
        }
    }
    return gains;
}


void SaturateCoverage::updateStatisticsAdd(const Set& sset, int item) const {
    // Update statistics for algorithms sequentially adding elements (for example, the greedy algorithm).
    for (int i = 0; i < n; i++) {
        preCompute[i] += kernel[i][item];
    }
    preComputeSet.insert(item);
}


void SaturateCoverage::updateStatisticsRemove(const Set& sset, int item) const {
    // Update statistics for algorithms sequentially removing elements (for example, the reverse greedy algorithm).
    for (int i = 0; i < n; i++) {
        preCompute[i] -= kernel[i][item];
    }
    preComputeSet.remove(item);
}


void SaturateCoverage::clearpreCompute() const {
    for (int i = 0; i < sizepreCompute; i++) {
        preCompute[i] = 0;
    }
    preComputeSet.clear();
}


void SaturateCoverage::setpreCompute(const Set& sset) const {
    clearpreCompute();
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); ++it) {
        updateStatisticsAdd(sset, *it);
    }
    preComputeSet = sset;
}
