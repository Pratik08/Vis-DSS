/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#include "set.h"

Set::Set() {
    uset = std::unordered_set<int>();
}

Set::Set(int max_elements) {
    uset = std::unordered_set<int>();
}

Set::Set(int max_elements, bool) {
    uset = std::unordered_set<int>();
    for (int i = 0; i < max_elements; i++)
        uset.insert(i);
}

Set::Set(const Set& other) : uset(other.uset) {
}                                                    // deep copy

Set& Set::operator=(const Set& other) {      // deep copy
    Set::const_iterator it;
    for (it = other.begin(); it != other.end(); it++) {
        uset.insert(*it);
    }
    return *this;
}

void Set::insert(int i) {
    uset.insert(i);
}

void Set::clear() {
    uset.clear();
}

void Set::remove(int i) {
    uset.erase(i);
}

bool Set::contains(int i) const {
    if (uset.count(i) >= 1) {
        return true;
    } else {
        return false;
    }
}

int Set::size() const {
    return uset.size();
}

Set::iterator Set::begin() {
    return uset.begin();
}

Set::iterator Set::end() {
    return uset.end();
}

Set::const_iterator Set::begin() const {
    return uset.begin();
}

Set::const_iterator Set::end() const {
    return uset.end();
}
