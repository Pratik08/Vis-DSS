/*
 *	Abstract base class header for implementing (not necessarily submodular) set functions. This implements all oracle functions of general set functions.

    All set functions are defined on a ground set V = [n] = {1, 2, ..., n}. If the groundSet is not [n],
    one will need to use a transformation function, which transforms a groundSet U, to V = [|U|], through a mapping.
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer
 *
 */
#ifndef SRC_ENGINE_SETFUNCTIONS_H_
#define SRC_ENGINE_SETFUNCTIONS_H_

#include <vector>
#include "set.h"

class SetFunctions {
    // protected:
 protected:
    const int n;   // Ground set size.

 public:
    // Constructors
    SetFunctions();
    SetFunctions(int n);
    SetFunctions(const SetFunctions& f);   // copy constructor
    virtual ~SetFunctions();

    virtual SetFunctions * clone() const = 0;
    // Evaluate the function at sset, i.e f(sset)
    virtual double eval(const Set& set) const = 0;
    // Evaluate the function at sset, i.e f(sset) using preComputed stats
    virtual double evalFast(const Set& set) const = 0;
    // evaulate the gain of adding item to sset, i.e f(item | sset)
    virtual double evalGainsadd(const Set& set, int item) const;
    // evaluate the gain of removing item from sset, i.e f(item | sset/item)
    virtual double evalGainsremove(const Set& set, int item) const;
    // fast evaluation of gain of adding item to sset, using preComputed statistics.
    virtual double evalGainsaddFast(const Set& set, int item) const;
    // fast evaluation of gain of adding item to sset, using preComputed statistics.
    virtual double evalGainsremoveFast(const Set& set, int item) const;

    // Functions managing preComputed statistics.
    // updating 'statistics' in sequential algorithms for adding elements. This is really useful for any algorithm which creates a sequence os sets X_0                 \subset X_1 \subset .. X_n such that |X_i| = |X_{i-1}| + 1.
    virtual void updateStatisticsAdd(const Set& set, int item) const = 0;
    // updating 'statistics' in sequential algorithms for removing elements. This is really useful for any algorithm which creates a sequence os sets X_0               \subset X_1 \subset .. X_n such that |X_i| = |X_{i-1}| + 1.
    virtual void updateStatisticsRemove(const Set& set, int item) const = 0;
    // Clears the precompute statistics.
    virtual void clearpreCompute() const = 0;
    // Compute the precomputed statistics for a given set.
    virtual void setpreCompute(const Set& set) const = 0;

    // Non-virtual functions
    int size() const;
    double operator()(const Set& set) const;
    double operator()(const Set& set, int item) const;
};

#endif  // SRC_ENGINE_SETFUNCTIONS_H_
