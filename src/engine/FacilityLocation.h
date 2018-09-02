/*
    Copyright (C) Rishabh Iyer
 *	Header file for defining the facility location function: f(X) = \sum_{i \in V} \max_{j \in X} s_{ij}.
    Assumes that the kernel S is symmetric.
    Author: Rishabh Iyer.
 *
 */

#ifndef SRC_ENGINE_FACILITYLOCATION_H_
#define SRC_ENGINE_FACILITYLOCATION_H_

#include <vector>
#include <iostream>
#include "set.h"
#include "SetFunctions.h"

class FacilityLocation : public SetFunctions {
 protected:
    const std::vector<std::vector <float> >& kernel;   // The matrix s_{ij}_{i \in V, j \in V}
    mutable std::vector<double> preCompute;   // Precomputed statistics of length 2*n. For a given set X, preCompute[i] = max_{j \in X} s_{ij} and preCompute[n+i] = 2max_{j \in X} s_{ij}, where 2max stands for the second max. This preComputed statistics is used in several algorithms for speed ups.
    const int sizepreCompute;   // size of the precompute statistics (in this case, 2*n).
    mutable Set preComputeSet;   // This points to the preComputed Set for which the statistics p_X is calculated.
 public:
    // Functions
    FacilityLocation(int n, const std::vector<std::vector <float> >& kernel);
    FacilityLocation(const FacilityLocation & f);
    ~FacilityLocation();
    FacilityLocation* clone() const;
    double eval(const Set& sset) const;
    double evalFast(const Set& sset) const;
    double evalGainsadd(const Set& sset, int item) const;
    double evalGainsremove(const Set& sset, int item) const;
    double evalGainsaddFast(const Set& sset, int item) const;
    double evalGainsremoveFast(const Set& sset, int item) const;
    void updateStatisticsAdd(const Set& sset, int item) const;
    void updateStatisticsRemove(const Set& sset, int item) const;
    void setpreCompute(const Set& ssett) const;
    void clearpreCompute() const;
};
#endif  // SRC_ENGINE_FACILITYLOCATION_H_
