#include "mmr.h"
#include <iostream>
using namespace std;

MMR::MMR(int n, std::vector<std::vector <float> >& kernel) : SetFunctions(n), kernel(kernel){
    preComputeVal = 0;
}

MMR::MMR(const MMR& f): SetFunctions(f.n), kernel(f.kernel){
    preComputeVal = 0;
}
MMR & MMR::operator=(const MMR & f){
    return *this;
}

MMR * MMR::clone() const{
    return new MMR(*this);
}

MMR::~MMR(){}

double MMR::eval(const Set& sset) const{
// Evaluation of function valuation.
    return preComputeVal;
}

double MMR::evalFast(const Set& sset) const{
    return preComputeVal;
}

double MMR::evalGainsadd(const Set& sset, int item) const{
    double minval = 1e30;
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); it++){
        if (1 - kernel[*it][item] < minval)// && (kernel[*it][item] != 1) )
            minval = 1 - kernel[*it][item];
    }
    return minval;
}

double MMR::evalGainsremove(const Set& sset, int item) const{
    double minval = 1e30;
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); it++){
        if ((1 - kernel[*it][item] < minval) && (item != *it))
            minval = 1 - kernel[*it][item];
    }
    return minval;
}

double MMR::evalGainsaddFast(const Set& sset, int item) const{
// Fast evaluation of Adding gains using the precomputed statistics. This is used, for example, in the implementation of the forward greedy algorithm.
// For the sake of speed, we do not check if item does not belong to the subset. You should check this before calling this function.
    return evalGainsadd(sset, item);
}

double MMR::evalGainsremoveFast(const Set& sset, int item) const{
// Fast evaluation of Removing gains using the precomputed statistics. This is used, for example, in the implementation of the reverse greedy algorithm.
// For the sake of speed, we do not check if item belong to the subset. You should check this before calling this function.
    return evalGainsremove(sset, item);
0;
}

void MMR::updateStatisticsAdd(const Set& sset, int item) const{
    double minval = 1e30;
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); it++){
        if ((1 - kernel[*it][item] < minval))
            minval = 1 - kernel[*it][item];
    }
    preComputeVal += (1 - minval);
}

void MMR::updateStatisticsRemove(const Set& sset, int item) const{
    double minval = 1e30;
    Set::const_iterator it;
    for (it = sset.begin(); it != sset.end(); it++){
        if ((1 - kernel[*it][item] < minval) && (item != *it))
            minval = 1 - kernel[*it][item];
    }
    preComputeVal -= (1 - minval);
}

void MMR::setpreCompute(const Set& sset) const{}

void MMR::clearpreCompute() const{
    preComputeVal = 0;
}

int MMR::size() const{
    return n;
}
