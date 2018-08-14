/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#ifndef SET_COVER
#define SET_COVER
#include <vector>
#include "set.h"
#include "SetFunctions.h"

class SetCover : public SetFunctions {
 protected:
    const std::vector<Set> & coverSet;  // The sets U_i, i \in V
    mutable Set preCompute;  // Precomputed statistics \cup_{i \in X} U_i
 public:
    // Functions
    SetCover(int n, const std::vector<Set> & coverSet);
    SetCover(const SetCover & f);
    ~SetCover();
    SetCover* clone() const;
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
