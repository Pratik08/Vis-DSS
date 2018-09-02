/*
    Copyright (C) Rishabh Iyer 2015
 *	Header file for defining a family of graph cut based functions: f(X) = \sum_{i \in X} \sum_{j \in V} s_{ij} - \lambda \sum_{i, j \in X} s_{ij}
    Assumes that the kernel S is symmetric.
    Author: Rishabh Iyer
 *
 */

#include "GraphCutFunctions.h"

GraphCutFunctions::GraphCutFunctions(int n, const std::vector<std::vector <float> >& kernel, const double lambda) : SetFunctions(n), kernel(kernel), lambda(lambda), sizepreCompute(n) {
    preCompute.resize(sizepreCompute);   // precomputed statistics for speeding up greedy
    if (lambda < 0) {
        std::cout << "*********************************************************************************" << std::endl;
        std::cout << "Warning: The input lambda = " << lambda << " is negative: the set function to be instantiated may not be submodular any more" << std::endl;
        std::cout << "*********************************************************************************" << std::endl;
    }
    for (int i = 0; i < sizepreCompute; i++) {
        preCompute[i] = 0;
    }
    modularscores.resize(n);
    for (int i = 0; i < n; i++) {
        modularscores[i] = 0;
        for (int j = 0; j < n; j++) {
            modularscores[i] += kernel[i][j];
        }
    }
}


GraphCutFunctions::GraphCutFunctions(const GraphCutFunctions& f) : SetFunctions(f.n), kernel(f.kernel), lambda(f.lambda), sizepreCompute(f.n) {
    preCompute = f.preCompute;
    modularscores = f.modularscores;
    // std::cout << "Calling copy constructor" << std::endl;
}

GraphCutFunctions::~GraphCutFunctions() {
}


GraphCutFunctions* GraphCutFunctions::clone() const {
    return new GraphCutFunctions(*this);
}


double GraphCutFunctions::eval(const Set& sset) const {
// Evaluation of function valuation.
    double sum = 0;
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); ++it) {
        sum += modularscores[*it];
        Set::const_iterator it2;
        for (it2 = sset.begin(); it2 != sset.end(); ++it2) {
            sum -= lambda * kernel[*it][*it2];
        }
    }
    return sum;
}


double GraphCutFunctions::evalFast(const Set& sset) const {
    double sum = 0;
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); ++it) {
        sum += modularscores[*it] - lambda * preCompute[*it];
    }
    return sum;
}


double GraphCutFunctions::evalGainsadd(const Set& sset, int item) const {
    if (sset.contains(item)) {
        std::cout << "Warning: the provided item belongs to the subset\n";
        return 0;
    }
    double gains = modularscores[item];
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); ++it) {
        gains -= lambda * (kernel[item][*it] + kernel[*it][item]);
    }
    gains -= lambda * kernel[item][item];
    return gains;
}


double GraphCutFunctions::evalGainsremove(const Set& sset, int item) const {
    if (!sset.contains(item)) {
        std::cout << "Warning: the provided item does not belong to the subset\n";
        return 0;
    }
    double gains = modularscores[item];
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); ++it) {
        if (*it != item) {
            gains -= lambda * (kernel[item][*it] + kernel[*it][item]);
        }
    }
    gains -= lambda * kernel[item][item];
    return gains;
}


double GraphCutFunctions::evalGainsaddFast(const Set& sset, int item) const {
// Fast evaluation of Adding gains using the precomputed statistics. This is used, for example, in the implementation of the forward greedy algorithm.
// For the sake of speed, we do not check if item does not belong to the subset. You should check this before calling this function.
    return (modularscores[item] - (2 * lambda * preCompute[item]) + (lambda * kernel[item][item]));
}


double GraphCutFunctions::evalGainsremoveFast(const Set& sset, int item) const {
// Fast evaluation of Removing gains using the precomputed statistics. This is used, for example, in the implementation of the reverse greedy algorithm.
// For the sake of speed, we do not check if item belong to the subset. You should check this before calling this function.

    return (modularscores[item] - (2 * lambda * preCompute[item]) + (lambda * kernel[item][item]));
}


void GraphCutFunctions::updateStatisticsAdd(const Set& sset, int item) const {
    for (int i = 0; i < n; i++) {
        preCompute[i] += kernel[i][item];
    }
}


void GraphCutFunctions::updateStatisticsRemove(const Set& sset, int item) const {
    for (int i = 0; i < n; i++) {
        preCompute[i] -= kernel[i][item];
    }
}


void GraphCutFunctions::setpreCompute(const Set& sset) const {
    clearpreCompute();
    Set::const_iterator it;
    /*
       for (it = sset.begin(); it != sset.end(); ++it) {
        updateStatisticsAdd(sset, *it);
       }*/
    for (int i = 0; i < n; i++) {
        for (it = sset.begin(); it != sset.end(); ++it) {
            preCompute[i] += kernel[i][*it];
        }
    }
}


void GraphCutFunctions::clearpreCompute() const {
    for (int i = 0; i < sizepreCompute; i++) {
        preCompute[i] = 0;
    }
}
