/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#ifndef SRC_ENGINE_NAIVEGREEDYMAX_H_
#define SRC_ENGINE_NAIVEGREEDYMAX_H_

#include <iostream>
#include <vector>
#include "set.h"
#include "SetFunctions.h"

void naiveGreedyMax(SetFunctions& f, double budget, Set& greedySet, int verbosity, bool precomputeSet, bool isEquality, Set groundSet = Set(), bool grSetn = true);

void naiveGreedyMaxKnapsack(SetFunctions& f, std::vector<double>& costList, double budget, Set& greedySet, int verbosity, bool precomputeSet, bool isEquality, Set groundSet = Set(), bool grSetn = true);

void naiveGreedyMaxSC(SetFunctions& f, std::vector<double>& costList, double coverfrac, Set& greedySet, int verbosity, bool precomputeSet, bool isEquality, Set groundSet = Set(), bool grSetn = true);

#endif  // SRC_ENGINE_NAIVEGREEDYMAX_H_
