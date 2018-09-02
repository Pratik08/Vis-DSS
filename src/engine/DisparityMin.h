/*
    Copyright (C) Rishabh Iyer
 *	Header file for defining a disparity sum set function: f(X) = \min_{i, j \in X} (1 - s_{ij})
    Assumes that the kernel S is symmetric.
    Note: This function is *not* submodular!
    Author: Rishabh Iyer
 *
 */
#ifndef SRC_ENGINE_DISPARITYMIN_H_
#define SRC_ENGINE_DISPARITYMIN_H_

#include <vector>
#include <algorithm>
#include <iostream>
#include "set.h"
#include "SetFunctions.h"

class DisparityMin : public SetFunctions {
 public:
    int n;
    std::vector<std::vector <float> >& kernel;  // The matrix s_{ij}_{i \in V, j \in V} of similarity (the assumption is it is normalized between 0 and 1 -- max similarity is 1 and min similarity is 0)
    mutable int sizepreCompute;
    mutable double preComputeVal;
    // Functions
    DisparityMin(int n, std::vector<std::vector <float> >& kernel);
    DisparityMin(const DisparityMin& f);
    DisparityMin & operator=(const DisparityMin & f);
    DisparityMin * clone() const;
    ~DisparityMin();
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
    int size() const;
};

#endif  // SRC_ENGINE_DISPARITYMIN_H_
