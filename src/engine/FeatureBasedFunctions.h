/*
 *  Copyright (C) Rishabh Iyer
 *	Header file for defining feature based functions (essentially sums of concave over modular functions): f(X) = \sum_{f \in F} w_f g(m_f(X)), where g is a concave function, {m_f}_{f \in F} are a set of feature scores, and f \in F are features.
    Author: Rishabh Iyer.
 *
 */
#ifndef FEATUREBASED_FUNCTION
#define FEATUREBASED_FUNCTION

#include <vector>
#include "SetFunctions.h"
#include "set.h"
#include "SparseFeature.h"

class FeatureBasedFunctions : public SetFunctions {
 protected:
    const int nFeatures;  // Number of Features |F|
    const int type;  // type: type of concave function, 1: sqrt over modular, 2: inverse function, 3: Log function, 4: Whatever else you want, Default: sqrt over modular
    const double thresh;
    const std::vector<struct SparseFeature>& feats;  // structure of the feature vectors for items. Size = n (groundset size)
    const std::vector<double>& featureWeights;  // Feature Weights (w_f)
    mutable std::vector<double> preCompute;  // Precomputed statistics. For a set X, p_X(f) = m_f(X).
    mutable Set preComputeSet;  // This points to the preComputed Set for which the statistics p_X is calculated.
    const int sizepreCompute;  // size of the precompute statistics (in this case, |F|).
    double concaveFunction(double K) const;

 public:
    // Functions
    FeatureBasedFunctions(int n, int type, std::vector<struct SparseFeature>& feats, std::vector<double>& featureWeights, double thresh = 1.0);
    FeatureBasedFunctions(const FeatureBasedFunctions & f);
    ~FeatureBasedFunctions();

    FeatureBasedFunctions* clone() const;

    double eval(const Set& sset) const;
    double evalFast(const Set& sset) const;
    double evalGainsadd(const Set& sset, int item) const;
    double evalGainsremove(const Set& sset, int item) const;
    double evalGainsaddFast(const Set& sset, int item) const;
    double evalGainsremoveFast(const Set& sset, int item) const;
    void updateStatisticsAdd(const Set& sset, int item) const;
    void updateStatisticsRemove(const Set& sset, int item) const;
    void setpreCompute(const Set& sset) const;
    void clearpreCompute() const;
};
#endif
