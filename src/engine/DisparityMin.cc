/*
    Copyright (C) Rishabh Iyer
 *	Header file for defining a family of graph cut based functions: f(X) = \min_{i, j \in X} (1 - s_{ij})
    Assumes that the kernel S is symmetric.
    Author: Rishabh Iyer
 *
 */

#include "DisparityMin.h"

#define INF 1e30

DisparityMin::DisparityMin(int n, std::vector<std::vector<float> >& kernel) : SetFunctions(n), kernel(kernel) {
    sizepreCompute = n;
    preComputeVal = 0;
}

DisparityMin::DisparityMin(const DisparityMin& f) : SetFunctions(f.n), kernel(f.kernel) {
    sizepreCompute = n;
    preComputeVal = 0;
}

DisparityMin & DisparityMin::operator=(const DisparityMin & f) {
    return *this;
}

DisparityMin * DisparityMin::clone() const {
    return new DisparityMin(*this);
}

DisparityMin::~DisparityMin() {
}

double DisparityMin::eval(const Set& sset) const {
// Evaluation of function valuation.
    double minval = 1;
    for (Set::const_iterator it = sset.begin(); it != sset.end(); it++) {
        for (Set::const_iterator it2 = sset.begin(); it2 != sset.end(); it2++) {
            if ((1 - kernel[*it][*it2] < minval) && (*it != *it2))
                minval = 1 - kernel[*it][*it2];
        }
    }
    return minval;
}

double DisparityMin::evalFast(const Set& sset) const {
    return preComputeVal;
}

double DisparityMin::evalGainsadd(const Set& sset, int item) const {
    if (sset.contains(item)) {
        // std::cout << "Warning: the provided item belongs to the subset\n";
        return 0;
    }
    double minval;
    if (sset.size() == 1) {
        minval = 1;
    } else {
        minval = preComputeVal;
    }
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); it++) {
        if ((1 - kernel[*it][item] < minval) && (*it != item)) {
            minval = 1 - kernel[*it][item];
        }
    }
    return (minval - preComputeVal);
}

double DisparityMin::evalGainsremove(const Set& sset, int item) const {
    if (!sset.contains(item)) {
        // std::cout << "Warning: the provided item does not belong to the subset\n";
        return 0;
    }
    Set rsset(sset);
    rsset.remove(item);
    return eval(sset) - eval(rsset);
}

double DisparityMin::evalGainsaddFast(const Set& sset, int item) const {
// Fast evaluation of Adding gains using the precomputed statistics. This is used, for example, in the implementation of the forward greedy algorithm.
// For the sake of speed, we do not check if item does not belong to the subset. You should check this before calling this function.
    double minval;
    if (sset.size() == 1) {
        minval = 1;
    } else {
        minval = preComputeVal;
    }
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); it++) {
        if ((1 - kernel[*it][item] < minval) && (*it != item)) {
            minval = 1 - kernel[*it][item];
        }
    }
    // std::cout << minval - preComputeVal << "\n" << std::flush;
    return (minval - preComputeVal);
}

double DisparityMin::evalGainsremoveFast(const Set& sset, int item) const {
// Fast evaluation of Removing gains using the precomputed statistics. This is used, for example, in the implementation of the reverse greedy algorithm.
// For the sake of speed, we do not check if item belong to the subset. You should check this before calling this function.
    Set rsset(sset);
    rsset.remove(item);
    return (eval(sset) - eval(rsset));
}

void DisparityMin::updateStatisticsAdd(const Set& sset, int item) const {
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); it++) {
        if ((1 - kernel[*it][item] < preComputeVal) && (*it != item)) {
            preComputeVal = 1 - kernel[*it][item];
        }
    }
}

void DisparityMin::updateStatisticsRemove(const Set& sset, int item) const {
}

void DisparityMin::setpreCompute(const Set& sset) const {
    preComputeVal = eval(sset);
}

void DisparityMin::clearpreCompute() const {
    preComputeVal = 0;
}

int DisparityMin::size() const {
    return n;
}
