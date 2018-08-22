/*
    Copyright (C) Rishabh Iyer
 *	A class containing the increments, used in lazy greedy algorithms
    Anthor: Rishabh Iyer
 *
 */
#include "stdio.h"
#include "increment.h"

Increment::Increment() {
}

Increment::Increment(double x, int i) {
    this->value = x;
    this->index = i;
}

int Increment::get_index() const {
    return index;
}

double Increment::get_value() const {
    return value;
}

bool Increment::operator<(const Increment& right) const {
    return value < right.value;
}
