/*
    Copyright (C) Rishabh Iyer

 *	The algorithm implemented here is the forward greedy algorithm, and has a 1 - 1/e guarantee for monotone submodular maximization!
    It greedily adds elements untill it violates the budget (and the gain is non-negative). This is an accelerated version with priority queue (Minoux 1976).
    Solves the problem \max_{|X| \leq B} f(X), where f is a submodular function.
    Anthor: Rishabh Iyer (adapted from an original implementation by Hui Lin)

    Input: Submodular Function: f
        budget: b
        GroundSet (in case it is not [n])
    Output: GreedySet: greedySet

    Suggested choice of the Set datastructure: VectorSet or Set

 *
 */
#ifndef SRC_ENGINE_LAZYGREEDYMAX_H_
#define SRC_ENGINE_LAZYGREEDYMAX_H_

#include <vector>
#include "set.h"
#include "SetFunctions.h"

void lazyGreedyMax(const SetFunctions& f, double budget, Set& greedySet, int verbosity = 1, bool precomputeSet = false,
                   bool equalityConstraint = false, Set groundSet = Set(), bool grSetn = true);

void lazyGreedyMaxKnapsack(SetFunctions& f, std::vector<double>& costList, double budget, Set& greedySet,
                           int verbosity = 1, double alpha = 1, Set groundSet = Set(), bool grSetn = true);

void lazyGreedyMaxSC(SetFunctions& f, std::vector<double>& costList, double cover, Set& greedySet, int verbosity = 1,
                     double alpha = 1, Set groundSet = Set(), bool grSetn = true);

#endif  // SRC_ENGINE_LAZYGREEDYMAX_H_
