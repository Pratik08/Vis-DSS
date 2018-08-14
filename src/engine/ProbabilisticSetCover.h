/*
    Copyright (C) 2015 Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#ifndef PROBABILISTIC_SET_COVER
#define PROBABILISTIC_SET_COVER
#include <vector>
#include "set.h"
#include "SetFunctions.h"

class ProbabilisticSetCover : public SetFunctions {
 protected:
    const std::vector<std::vector<float> > & p;  // The sets U_i, i \in V
    mutable std::vector<float> preCompute;  // Precomputed statistics \cup_{i \in X} U_i
    int nConcepts;
 public:
    // Functions
    ProbabilisticSetCover(int n, int nConcepts, const std::vector<std::vector<float> >& p);
    ProbabilisticSetCover(const ProbabilisticSetCover & f);
    ~ProbabilisticSetCover();
    ProbabilisticSetCover* clone() const;
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

#endif
