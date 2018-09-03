/*
    Copyright (C) Rishabh Iyer
 *	Defining the facility location function: f(X) = \sum_{i \in V} \max_{j \in X} s_{ij}.
    Assumes that the kernel S is symmetric.
    Author: Rishabh Iyer.
 *
 */
#include "FacilityLocation.h"

FacilityLocation::FacilityLocation(int n, const std::vector<std::vector <float> >& kernel) : SetFunctions(n), kernel(kernel), sizepreCompute(2 * n) {
    // std::cout << sizeof(FacilityLocation) << " " << sizeof(kernel) << endl;
    preCompute.assign(sizepreCompute, 0);   // precomputed statistics for speeding up greedy
}


FacilityLocation::FacilityLocation(const FacilityLocation & f) : SetFunctions(f.n), kernel(f.kernel), sizepreCompute(f.sizepreCompute) {
    preCompute = f.preCompute;
    preComputeSet = f.preComputeSet;
    // std::cout << preCompute.size() << endl;
    // for (std::vector<double>::iterator it = preCompute.begin(); it != preCompute.end(); it++) {
    // std::cout << *it << endl;
    // }
}


FacilityLocation::~FacilityLocation() {
    preComputeSet.clear();
    preCompute.clear();
}


FacilityLocation* FacilityLocation::clone() const {
    return new FacilityLocation(*this);
}


double FacilityLocation::eval(const Set& sset) const {
    double sum = 0;
    double maxvali;
    Set::const_iterator it;
    for (int i = 0; i < n; i++) {
        maxvali = 0;
        for (it = sset.begin(); it != sset.end(); ++it) {
            if (kernel[i][*it] > maxvali) {
                maxvali = kernel[i][*it];
            }
        }
        sum += maxvali;
    }
    return sum;
}


double FacilityLocation::evalFast(const Set& sset) const {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += preCompute[i];
    }
    return sum;
}


double FacilityLocation::evalGainsadd(const Set& sset, int item) const {
    if (sset.contains(item)) {
        std::cout << "Error in using evalGainsadd: the provided item already belongs to the subset\n";
        return 0;
    }
    double gains = 0;
    Set::const_iterator it;
    for (int i = 0; i < n; i++) {
        double maxvali = 0;
        for (it = sset.begin(); it != sset.end(); ++it) {
            if (kernel[i][*it] > maxvali) {
                maxvali = kernel[i][*it];
            }
        }
        if (maxvali < kernel[i][item]) {
            gains += (kernel[i][item] - maxvali);
        }
    }
    return gains;
}


double FacilityLocation::evalGainsremove(const Set& sset, int item) const {
    if (!sset.contains(item)) {
        std::cout << "Error in using evalGainsremove: the provided item does not belong to the subset\n";
        return 0;
    }
    double sum = 0;
    double sumd = 0;
    double maxvali;
    double maxvald;
    Set::const_iterator it;
    for (int i = 0; i < n; i++) {
        maxvali = 0;
        maxvald = 0;
        for (it = sset.begin(); it != sset.end(); ++it) {
            if (kernel[i][*it] > maxvali) {
                maxvali = kernel[i][*it];
            }
            if ((*it != item) && (kernel[i][*it] > maxvald)) {
                maxvald = kernel[i][*it];
            }
        }
        sum += maxvali;
        sumd += maxvald;
    }
    return (sum - sumd);
}


double FacilityLocation::evalGainsaddFast(const Set& sset, int item) const {
// Fast evaluation of Adding gains using the precomputed statistics. This is used, for example, in the implementation of the forward greedy algorithm.
    double gains = 0;
    for (int i = 0; i < n; i++) {
        if (preCompute[i] < kernel[item][i]) {
            gains += (kernel[item][i] - preCompute[i]);
        }
    }
    return gains;
}


double FacilityLocation::evalGainsremoveFast(const Set& sset, int item) const {
// Fast evaluation of Removing gains using the precomputed statistics. This is used, for example, in the implementation of the reverse greedy algorithm.
    double gains = 0;
    for (int i = 0; i < n; i++) {
        if (preCompute[i] == kernel[item][i]) {
            gains += (preCompute[i] - preCompute[n + i]);
        }
    }
    return gains;
}


void FacilityLocation::updateStatisticsAdd(const Set& sset, int item) const {
// Update statistics for algorithms sequentially adding elements (for example, the greedy algorithm).
    for (int i = 0; i < n; i++) {
        if (kernel[i][item] > preCompute[i]) {
            preCompute[n + i] = preCompute[i];
            preCompute[i] = kernel[i][item];
        } else if (kernel[i][item] > preCompute[n + i]) {
            preCompute[n + i] = kernel[i][item];
        }
    }
    preComputeSet.insert(item);
}


void FacilityLocation::updateStatisticsRemove(const Set& sset, int item) const {
// Update statistics for algorithms sequentially removing elements (for example, the reverse greedy algorithm).
    for (int i = 0; i < n; i++) {
        if ((kernel[i][item] == preCompute[i]) || (kernel[i][item] == preCompute[n + i])) {   // We obtained the largest or the second largest value, we need to recompute the statistics. Else, it remains the same.
            preCompute[i] = 0; preCompute[n + i] = 0;
            Set::const_iterator it;
            for (it = sset.begin(); it != sset.end(); ++it) {
                if (*it != item) {
                    if (kernel[i][*it] > preCompute[n + i]) {
                        if (kernel[i][*it] <= preCompute[i]) {
                            preCompute[n + i] = kernel[i][*it];
                        } else {
                            preCompute[n + i] = preCompute[i];
                            preCompute[i] = kernel[i][*it];
                        }
                    }
                }
            }
        }
    }
    preComputeSet.remove(item);
}


void FacilityLocation::clearpreCompute() const {
    for (int i = 0; i < sizepreCompute; i++) {
        preCompute[i] = 0; preComputeSet.clear();
    }
}


void FacilityLocation::setpreCompute(const Set& sset) const {
    clearpreCompute();
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); ++it) {
        updateStatisticsAdd(sset, *it);
    }
    preComputeSet = sset;
}
