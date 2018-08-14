/*
    Copyright (C) Rishabh Iyer
 *	Abstract base class for implementing (not necessarily submodular) set functions. This implements all oracle functions of general set functions.
    Author: Rishabh Iyer
 *
 */
#include "SetFunctions.h"
#include <iostream>
#include "set.h"
using namespace std;

SetFunctions::SetFunctions() : n(0) {
}


SetFunctions::SetFunctions(int n) : n(n) {
}


SetFunctions::SetFunctions(const SetFunctions& f) : n(f.n) {
}


SetFunctions::~SetFunctions() {
}


int SetFunctions::size() const {
    return n;
}


double SetFunctions::evalGainsadd(const Set& set, int item) const {
    // std::cout << "Call set function gain add " << std::endl;
    /*HashSet vset(set.size()+1); // there is possibly a bug with the VectorSet's insert method
       Set::const_iterator it(set);
       for (it = set.begin(); it != set.end(); ++it){
        if (*it == item){
            error("item already in the set to be added\n");
        }
        vset.insert(*it);
       }*/
    Set aset(set);

    aset.insert(item);
    return eval(aset) - eval(set);
}


double SetFunctions::evalGainsremove(const Set& set, int item) const {
    Set rset;  // probably, we need to implement the copy constructor for Set

    Set::const_iterator it;
    for (it = set.begin(); it != set.end(); ++it) {
        rset.insert(*it);
    }
    rset.remove(item);
    return eval(set) - eval(rset);
}


double SetFunctions::evalGainsaddFast(const Set& set, int item) const {
    return evalGainsadd(set, item);
}


double SetFunctions::evalGainsremoveFast(const Set& set, int item) const {
    return evalGainsremove(set, item);
}


double SetFunctions::operator()(const Set& set) const {
    return eval(set);
}


double SetFunctions::operator()(const Set& set, int item) const {
    return evalGainsadd(set, item);
}
