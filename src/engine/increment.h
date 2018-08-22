/*
    Copyright (C) Rishabh Iyer
 *	A class containing the increments, used in lazy greedy algorithms
    Anthor: Rishabh Iyer
 *
 */
#ifndef SRC_ENGINE_INCREMENT_H_
#define SRC_ENGINE_INCREMENT_H_

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

#endif  // SRC_ENGINE_INCREMENT_H_
