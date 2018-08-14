/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#ifndef NAIVEGREEDYMAX
#define NAIVEGREEDYMAX

#include "set.h"
#include "SetFunctions.h"
#include <iostream>
using namespace std;

void naiveGreedyMax(SetFunctions& f, double budget, Set& greedySet, int verbosity, bool precomputeSet,
bool isEquality, Set groundSet = Set(), bool grSetn = true);

void naiveGreedyMaxKnapsack(SetFunctions& f, std::vector<double>& costList, double budget, Set& greedySet, int verbosity, bool precomputeSet,
bool isEquality, Set groundSet = Set(), bool grSetn = true);

void naiveGreedyMaxSC(SetFunctions& f, std::vector<double>& costList, double coverfrac, Set& greedySet, int verbosity, bool precomputeSet,
bool isEquality, Set groundSet = Set(), bool grSetn = true);
#endif // NAIVEGREEDYMAX
