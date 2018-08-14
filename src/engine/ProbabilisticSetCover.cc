/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */
#include <iostream>
using namespace std;

#include "ProbabilisticSetCover.h"

ProbabilisticSetCover::ProbabilisticSetCover(int n, int nConcepts, const std::vector<std::vector<float> >& p) : SetFunctions(n), nConcepts(nConcepts), p(p) {
    preCompute = std::vector<float> (nConcepts);  // precomputed statistics for speeding up greedy
}


ProbabilisticSetCover::ProbabilisticSetCover(const ProbabilisticSetCover & f) : SetFunctions(f), nConcepts(f.nConcepts), p(f.p) {
    preCompute = f.preCompute;
}


ProbabilisticSetCover::~ProbabilisticSetCover() {
    preCompute = std::vector<float>(nConcepts);
}


ProbabilisticSetCover* ProbabilisticSetCover::clone() const {
    return new ProbabilisticSetCover(*this);
}


double ProbabilisticSetCover::eval(const Set& sset) const {
    Set cset = Set();

    Set::const_iterator it;
    double sum = 0;
    for (int i = 0; i < nConcepts; i++) {
        double prod = 1;
        for ( it = sset.begin(); it != sset.end(); ++it) {
            prod = prod * (1 - p[*it][i]);
        }
        sum = sum + (1 - prod);
    }
    return sum;
}


double ProbabilisticSetCover::evalFast(const Set& sset) const {
    double sum = 0;

    for (int i = 0; i < nConcepts; i++) {
        sum = sum + (1 - preCompute[i]);
    }
    return sum;
}


double ProbabilisticSetCover::evalGainsadd(const Set& sset, int item) const {
    if (sset.contains(item)) {
        cout << "Error in using evalGainsadd: the provided item already belongs to the subset\n";
        return 0;
    }
    Set aset = sset;
    aset.insert(item);
    return eval(aset) - eval(sset);
}


double ProbabilisticSetCover::evalGainsremove(const Set& sset, int item) const {
    if (!sset.contains(item)) {
        cout << "Error in using evalGainsremove: the provided item does not belong to the subset\n";
        return 0;
    }
    Set rset = sset;
    rset.remove(item);
    return eval(sset) - eval(rset);
}


double ProbabilisticSetCover::evalGainsaddFast(const Set& sset, int item) const {
// Fast evaluation of Adding gains using the precomputed statistics. This is used, for example, in the implementation of the forward greedy algorithm.
    double gains = 0;

    for (int i = 0; i < nConcepts; i++) {
        gains = gains + (preCompute[i] - preCompute[i] * (1 - p[item][i]));
    }
    return gains;
}


double ProbabilisticSetCover::evalGainsremoveFast(const Set& sset, int item) const {
// Fast evaluation of Removing gains using the precomputed statistics. This is used, for example, in the implementation of the reverse greedy algorithm.
    double gains = 0;

    for (int i = 0; i < nConcepts; i++) {
        gains = gains + (preCompute[i] / (1 - p[item][i]) - preCompute[i]);
    }
    return gains;
}


void ProbabilisticSetCover::updateStatisticsAdd(const Set& sset, int item) const {
// Update statistics for algorithms sequentially adding elements (for example, the greedy algorithm).
    for (int i = 0; i < nConcepts; i++) {
        preCompute[i] = preCompute[i] * (1 - p[item][i]);
    }
}


void ProbabilisticSetCover::updateStatisticsRemove(const Set& sset, int item) const {
// Update statistics for algorithms sequentially removing elements (for example, the reverse greedy algorithm).
    for (int i = 0; i < nConcepts; i++) {
        preCompute[i] = preCompute[i] / (1 - p[item][i]);
    }
}


void ProbabilisticSetCover::clearpreCompute() const {
    preCompute = std::vector<float>(nConcepts);
}


void ProbabilisticSetCover::setpreCompute(const Set& sset) const {
    clearpreCompute();
    Set::const_iterator it;
    for (int i = 0; i < nConcepts; i++) {
        preCompute[i] = 1;
        for ( it = sset.begin(); it != sset.end(); ++it) {
            preCompute[i] = preCompute[i] * (1 - p[*it][i]);
        }
    }
}
