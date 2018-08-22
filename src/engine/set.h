/*
    Copyright (C) Rishabh Iyer
    Author: Rishabh Iyer.
 *
 */

#ifndef SRC_ENGINE_SET_H_
#define SRC_ENGINE_SET_H_

#include <unordered_set>

class Set {
 protected:
    std::unordered_set<int> uset;
 public:
    Set();
    Set(int max_elements);
    Set(int max_elements, bool);
    Set(const Set& other);
    Set& operator=(const Set& other);
    void insert(int i);
    void remove(int i);
    bool contains(int i) const;
    void clear();
    int size() const;
    typedef std::unordered_set<int>::iterator iterator;
    typedef std::unordered_set<int>::const_iterator const_iterator;

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
};


#endif  // SRC_ENGINE_SET_H_
