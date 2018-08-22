/*
    Copyright (C) 2015 Rishabh Iyer
 *	Header file for defining the saturated coverage function: f(X) = \sum_{i \in V} \min\{\sum_{j \in X} s_{ij}, \alpha \sum_{j \in V} s_{ij}\}.
    Author: Rishabh Iyer.
 *
 */
#ifndef SRC_ENGINE_SATURATECOVERAGE_H_
#define SRC_ENGINE_SATURATECOVERAGE_H_

#include <vector>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include "set.h"
#include "SetFunctions.h"

class SaturateCoverage : public SetFunctions {
    const std::vector<std::vector <float> >& kernel;  // similarity kernel (s_{ij})_{i, j \in V} (assumed to be symmetric).
    const double alpha;  // the threshold alpha
    std::vector<double> modularthresh;  // a Precomputed quantity: modularthresh[i] = sum_{i \in V} s_{ij}.
    mutable std::vector<double> preCompute;  // Precomputed statistics: Stores (for given X) p_X(i) = \sum_{j \in X} s_{ij}.
    const int sizepreCompute;  // size of the precompute statistics (in this case, n)
    mutable Set preComputeSet;  // This points to the Set X for which the statistics p_X is calculated.
 public:
    // Functions
    SaturateCoverage(int n, const std::vector<std::vector <float> >& kernel, const double alpha);
    SaturateCoverage(const SaturateCoverage& f);
    ~SaturateCoverage();
    SaturateCoverage* clone() const;
    double eval(const Set& sset) const;
    double evalFast(const Set& sset) const;
    double evalGainsadd(const Set& sset, int item) const;
    double evalGainsremove(const Set& sset, int item) const;
    double evalGainsaddFast(const Set& sset, int item) const;
    double evalGainsremoveFast(const Set& sset, int item) const;
    void updateStatisticsAdd(const Set& sset, int item) const;
    void updateStatisticsRemove(const Set& sset, int item) const;
    void setpreCompute(const Set& sset) const;
    void clearpreCompute() const;
};

#endif  // SRC_ENGINE_SATURATECOVERAGE_H_
