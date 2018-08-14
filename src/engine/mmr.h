#ifndef MMR_H
#define MMR_H

#include "set.h"
#include "SetFunctions.h"
#include <vector>

class MMR : public SetFunctions {
    public:
    int n;
    std::vector<std::vector <float> >& kernel; // The matrix s_{ij}_{i \in V, j \in V} of similarity (the assumption is it is normalized between 0 and 1 -- max similarity is 1 and min similarity is 0)
    mutable double preComputeVal;
    // Functions
    MMR(int n, std::vector<std::vector <float> >& kernel);
    MMR(const MMR& f);
    MMR & operator=(const MMR & f);
    MMR * clone() const;
    ~MMR();
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

#endif // MMR_H
