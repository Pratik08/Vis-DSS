/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#include "streamGreedy.h"

void streamGreedy(const SetFunctions& f, double epsilon, Set& greedySet, std::vector<int>& stream, int verbosity) {
    f.setpreCompute(greedySet);  // clear the precomputed statistics, in case it is already not cleared, and set it to the greedySet.
    // accelerated greedy algorithm implementation
    // make sure that the length of the stream is the ground set size
    assert(stream.size() == f.size());
    for (int i = 0; i < stream.size(); i++) {
        double val = f.evalGainsaddFast(greedySet, stream[i]);
        if (val > epsilon) {
            greedySet.insert(stream[i]);
            f.updateStatisticsAdd(greedySet, stream[i]);
            if (verbosity > 0) {
                printf("Added item %d, gains = %f, epsilon = %f\n", stream[i], val, epsilon);
            }
        }
    }
}
