/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#include "SetCover.h"

SetCover::SetCover(int n, const std::vector<Set> & coverSet) : SetFunctions(n), coverSet(coverSet) {
    preCompute = Set();   // precomputed statistics for speeding up greedy
}


SetCover::SetCover(const SetCover & f) : SetFunctions(f.n), coverSet(f.coverSet) {
    preCompute = f.preCompute;
    // cout << preCompute.size() << endl;
    // for (std::vector<double>::iterator it = preCompute.begin(); it != preCompute.end(); it++) {
    // cout << *it << endl;
    // }
}


SetCover::~SetCover() {
    preCompute = Set();
    // kernel = NULL;
}


SetCover* SetCover::clone() const {
    return new SetCover(*this);
}


double SetCover::eval(const Set& sset) const {
    Set cset = Set();
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); ++it) {
        Set::const_iterator it2;
        for (it2 = coverSet[*it].begin(); it2 != coverSet[*it].end(); ++it2) {
            cset.insert(*it2);
        }
    }
    return cset.size();
}


double SetCover::evalFast(const Set& sset) const {
    return preCompute.size();
}


double SetCover::evalGainsadd(const Set& sset, int item) const {
    if (sset.contains(item)) {
        std::cout << "Error in using evalGainsadd: the provided item already belongs to the subset\n";
        return 0;
    }
    Set aset = sset;
    aset.insert(item);
    return (eval(aset) - eval(sset));
}


double SetCover::evalGainsremove(const Set& sset, int item) const {
    if (!sset.contains(item)) {
        std::cout << "Error in using evalGainsremove: the provided item does not belong to the subset\n";
        return 0;
    }
    Set rset = sset;
    rset.remove(item);
    return (eval(sset) - eval(rset));
}


double SetCover::evalGainsaddFast(const Set& sset, int item) const {
// Fast evaluation of Adding gains using the precomputed statistics. This is used, for example, in the implementation of the forward greedy algorithm.
    Set aset = preCompute;
    Set::const_iterator it2;
    for (it2 = coverSet[item].begin(); it2 != coverSet[item].end(); ++it2) {
        aset.insert(*it2);
    }
    return (aset.size() - preCompute.size());
}


double SetCover::evalGainsremoveFast(const Set& sset, int item) const {
// Fast evaluation of Removing gains using the precomputed statistics. This is used, for example, in the implementation of the reverse greedy algorithm.
    Set rset = preCompute;
    Set::const_iterator it2;
    for (it2 = coverSet[item].begin(); it2 != coverSet[item].end(); ++it2) {
        rset.remove(*it2);
    }
    return (preCompute.size() - rset.size());
}


void SetCover::updateStatisticsAdd(const Set& sset, int item) const {
// Update statistics for algorithms sequentially adding elements (for example, the greedy algorithm).
    Set::const_iterator it2;
    for (it2 = coverSet[item].begin(); it2 != coverSet[item].end(); ++it2) {
        preCompute.insert(*it2);
    }
}


void SetCover::updateStatisticsRemove(const Set& sset, int item) const {
// Update statistics for algorithms sequentially removing elements (for example, the reverse greedy algorithm).
    Set::const_iterator it2;
    for (it2 = coverSet[item].begin(); it2 != coverSet[item].end(); ++it2) {
        preCompute.remove(*it2);
    }
}


void SetCover::clearpreCompute() const {
    preCompute = Set();
}


void SetCover::setpreCompute(const Set& sset) const {
    clearpreCompute();
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); ++it) {
        Set::const_iterator it2;
        for (it2 = coverSet[*it].begin(); it2 != coverSet[*it].end(); ++it2) {
            preCompute.insert(*it2);
        }
    }
}
