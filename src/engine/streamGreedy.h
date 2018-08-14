/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#ifndef STREAMGREEDY
#define STREAMGREEDY

#include "SetFunctions.h"
#include "set.h"

void streamGreedy(const SetFunctions& f, double epsilon, Set& greedySet, std::vector<int>& stream, int verbosity = 0);

#endif // STREAMGREEDY
