/*
    Copyright (C) Rishabh Iyer
 *	Header file for defining a disparity sum set function: f(X) = \sum_{i, j \in X} (1 - s_{ij})
    Assumes that the kernel S is symmetric.
    Author: Rishabh Iyer
 *
 */

#include "DisparitySum.h"

DisparitySum::DisparitySum(int n, std::vector<std::vector<float> >& kernel) : SetFunctions(n), kernel(kernel) {
    sizepreCompute = n;
    preComputeVal = 0;
}


DisparitySum::DisparitySum(const DisparitySum& f) : SetFunctions(f.n), kernel(f.kernel) {
    sizepreCompute = n;
    preComputeVal = 0;
}


DisparitySum & DisparitySum::operator=(const DisparitySum & f) {
    return *this;
}


DisparitySum * DisparitySum::clone() const {
    return new DisparitySum(*this);
}


DisparitySum::~DisparitySum() {
}


double DisparitySum::eval(const Set& sset) const {
// Evaluation of function valuation.
    double sum = 0;
    for (Set::const_iterator it = sset.begin(); it != sset.end(); it++) {
        for (Set::const_iterator it2 = sset.begin(); it2 != sset.end(); it2++) {
            sum += 1 - kernel[*it][*it2];
        }
    }
    return sum;
}


double DisparitySum::evalFast(const Set& sset) const {
    return preComputeVal;
}


double DisparitySum::evalGainsadd(const Set& sset, int item) const {
    if (sset.contains(item)) {
        std::cout << "Warning: the provided item belongs to the subset\n";
        return 0;
    }
    double gains = 0;
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); it++) {
        gains += (2 - kernel[item][*it] - kernel[*it][item]);
    }
    gains += (1 - kernel[item][item]);
    return gains;
}


double DisparitySum::evalGainsremove(const Set& sset, int item) const {
    if (!sset.contains(item)) {
        std::cout << "Warning: the provided item does not belong to the subset\n";
        return 0;
    }
    double gains = 0;
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); it++) {
        if (*it != item)
            gains += (2 - kernel[item][*it] - kernel[*it][item]);
    }
    gains += (1 - kernel[item][item]);
    return gains;
}


double DisparitySum::evalGainsaddFast(const Set& sset, int item) const {
// Fast evaluation of Adding gains using the precomputed statistics. This is used, for example, in the implementation of the forward greedy algorithm.
// For the sake of speed, we do not check if item does not belong to the subset. You should check this before calling this function.
    double gains = 0;
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); it++) {
        gains += (2 - kernel[item][*it] - kernel[*it][item]);
    }
    gains += (1 - kernel[item][item]);
    return gains;
}


double DisparitySum::evalGainsremoveFast(const Set& sset, int item) const {
// Fast evaluation of Removing gains using the precomputed statistics. This is used, for example, in the implementation of the reverse greedy algorithm.
// For the sake of speed, we do not check if item belong to the subset. You should check this before calling this function.
    double gains = 0;
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); it++) {
        if (*it != item)
            gains += (2 - kernel[item][*it] - kernel[*it][item]);
    }
    gains += (1 - kernel[item][item]);
    return gains;
}


void DisparitySum::updateStatisticsAdd(const Set& sset, int item) const {
    preComputeVal += evalGainsaddFast(sset, item);
}


void DisparitySum::updateStatisticsRemove(const Set& sset, int item) const {
    preComputeVal -= evalGainsaddFast(sset, item);
}


void DisparitySum::setpreCompute(const Set& sset) const {
    preComputeVal = eval(sset);
}


void DisparitySum::clearpreCompute() const {
    preComputeVal = 0;
}


void DisparitySum::resetData(std::vector<int> Rset) {
}
