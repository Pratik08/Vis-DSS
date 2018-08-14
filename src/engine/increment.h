/*
    Copyright (C) Rishabh Iyer
 *	A class containing the increments, used in lazy greedy algorithms
    Anthor: Rishabh Iyer
 *
 */
#ifndef __INCREMENT__
#define __INCREMENT__

class Increment {
 public:
    Increment();
    Increment(double x, int i);
    bool operator<(const Increment& right) const;
    int get_index() const;
    double get_value() const;
 private:
    int index;
    double value;
};

#endif
