/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#ifndef SRC_ENGINE_STREAMGREEDY_H_
#define SRC_ENGINE_STREAMGREEDY_H_

#include <vector>
#include "SetFunctions.h"
#include "set.h"
#include "assert.h"

void streamGreedy(const SetFunctions& f, double epsilon, Set& greedySet, std::vector<int>& stream, int verbosity = 0);

#endif  // SRC_ENGINE_STREAMGREEDY_H_
