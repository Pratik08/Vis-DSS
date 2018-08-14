/*
    Copyright (C) Rishabh Iyer 2015
 *	Header file for defining a family of graph cut based functions: f(X) = \sum_{i \in X} \sum_{j \in V} s_{ij} - \lambda \sum_{i, j \in X} s_{ij}
    Assumes that the kernel S is symmetric.
    Author: Rishabh Iyer
 *
 */
#ifndef SMTK_GRAPHCUT_SF
#define SMTK_GRAPHCUT_SF

#include <vector>
#include "set.h"
#include "SetFunctions.h"

class GraphCutFunctions : public SetFunctions {
 public:
    const std::vector<std::vector <float> >& kernel;  // The matrix s_{ij}_{i \in V, j \in V}
    mutable std::vector<double> preCompute;  // stores p_X(j) = \sum_{i \in X} s_{ij}, for a given X.
    mutable std::vector<double> modularscores;  // a Precomputed quantity: modularscores[i] = sum_{i \in V} s_{ij}.
    const double lambda;
    const int sizepreCompute;
    // Functions
    GraphCutFunctions(int n, const std::vector<std::vector <float> >& kernel, const double lambdaval);
    GraphCutFunctions(const GraphCutFunctions& f);
    ~GraphCutFunctions();
    GraphCutFunctions* clone() const;
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

#endif
