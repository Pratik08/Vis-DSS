/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#include "naiveGreedyMax.h"

void naiveGreedyMax(SetFunctions& f, double budget, Set& greedySet, int verbosity, bool precomputeSet,
                    bool isEquality, Set groundSet, bool grSetn) {
    if (grSetn) {
        groundSet = Set(f.size(), true);  // GroundSet constructor
    }
    if (verbosity > 0) {
        std::cout << "Starting the naive greedy algorithm\n" << std::flush;
    }
    double currvalbest;
    int curridbest;
    double currentCost = greedySet.size();
    int iter = 0;
    if (!precomputeSet) {
        // f.setpreCompute(greedySet); // clear the precomputed statistics, in case it is already not cleared, and set it to the greedySet.
    }
    // naive greedy algorithm implementation
    Set::iterator it;
    while (1) {
        iter++;
        currvalbest = -1;
        for (it = groundSet.begin(); it != groundSet.end(); ++it) {
            if (!greedySet.contains(*it) && (f.evalGainsaddFast(greedySet, *it) > currvalbest)) {
                currvalbest = f.evalGainsaddFast(greedySet, *it);
                curridbest = *it;
            }
        }
        if (verbosity > 0) {
            std::cout << "Current best value is " << currvalbest << "\n" << std::flush;
        }
        if (currentCost <= budget) {
            if ((currvalbest < 0) && (!isEquality)) {
                break;
            }
            f.updateStatisticsAdd(greedySet, curridbest);
            currentCost += 1;
            greedySet.insert(curridbest);
            if (verbosity > 0) {
                std::cout << "Adding element " << curridbest << " in iteration " << iter << " and the objective value is " << f.eval(greedySet) << "\n" << std::flush;
            }
        } else {
            break;
        }
    }
}

void naiveGreedyMaxKnapsack(SetFunctions& f, std::vector<double>& costList, double budget, Set& greedySet, int verbosity, bool precomputeSet, bool isEquality, Set groundSet, bool grSetn) {
    if (grSetn) {
        groundSet = Set(f.size(), true);  // GroundSet constructor
    }
    if (verbosity > 0) {
        std::cout << "Starting the naive greedy algorithm\n" << std::flush;
    }
    double currvalbest;
    int curridbest;
    double currentCost = greedySet.size();
    int iter = 0;
    if (!precomputeSet) {
        f.setpreCompute(greedySet);  // clear the precomputed statistics, in case it is already not cleared, and set it to the greedySet.
    }
    // naive greedy algorithm implementation
    Set::iterator it;
    while (1) {
        iter++;
        currvalbest = -10000000;
        for (it = groundSet.begin(); it != groundSet.end(); ++it) {
            if (!greedySet.contains(*it) && (f.evalGainsaddFast(greedySet, *it) / costList[*it] > currvalbest)) {
                currvalbest = f.evalGainsaddFast(greedySet, *it) / costList[*it];
                curridbest = *it;
            }
        }
        if (verbosity > 0) {
            std::cout << "Current best value is " << currvalbest << " and id is " << curridbest << "\n" << std::flush;
        }
        if (currentCost + costList[curridbest] <= budget) {
            if ((currvalbest < 0) && (!isEquality)) {
                break;
            }
            f.updateStatisticsAdd(greedySet, curridbest);
            currentCost += costList[curridbest];
            greedySet.insert(curridbest);
            if (verbosity > 0) {
                std::cout << "Adding element " << curridbest << " in iteration " << iter << " and the objective value is " << f.eval(greedySet) << "\n" << std::flush;
            }
        } else {
            break;
        }
    }
}

void naiveGreedyMaxSC(SetFunctions& f, std::vector<double>& costList, double coverfrac, Set& greedySet, int verbosity, bool precomputeSet, bool isEquality, Set groundSet, bool grSetn) {
    if (grSetn) {
        groundSet = Set(f.size(), true);  // GroundSet constructor
    }
    if (verbosity > 0) {
        std::cout << "Starting the naive greedy algorithm\n" << std::flush;
    }
    double currvalbest;
    int curridbest;
    double currentCost;
    int iter = 0;
    if (!precomputeSet) {
        f.setpreCompute(greedySet);  // clear the precomputed statistics, in case it is already not cleared, and set it to the greedySet.
    }
    // naive greedy algorithm implementation
    Set::iterator it;
    double fV = f.eval(groundSet);
    while (1) {
        iter++;
        currvalbest = -1;
        for (it = groundSet.begin(); it != groundSet.end(); ++it) {
            if (!greedySet.contains(*it) && (f.evalGainsaddFast(greedySet, *it) / costList[*it] > currvalbest)) {
                currvalbest = f.evalGainsaddFast(greedySet, *it);
                curridbest = *it;
            }
        }
        if (verbosity > 0) {
            std::cout << "Current best value is " << currvalbest << "\n" << std::flush;
        }
        if (f.evalFast(greedySet) + f.evalGainsaddFast(greedySet, curridbest) < fV * coverfrac) {
            if ((currvalbest < 0) && (!isEquality)) {
                break;
            }
            f.updateStatisticsAdd(greedySet, curridbest);
            currentCost += costList[*it];
            greedySet.insert(curridbest);
            if (verbosity > 0) {
                std::cout << "Adding element " << curridbest << " in iteration " << iter << " and the objective value is " << f.eval(greedySet) << "\n" << std::flush;
            }
        } else {
            break;
        }
    }
}
