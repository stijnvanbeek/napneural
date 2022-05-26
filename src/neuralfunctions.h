#pragma once

#include <neuraltypes.h>
#include <vector>

namespace neural
{

    Value sigmoid(Value x);
    Value sigmoidDerivative(Value x);
    Value random(Value low, Value high);

    class Sigmoid
    {
    public:
        Sigmoid(Value min, Value max, int n);

        Value get(Value x);
        Value getDerivative(Value x);

    private:
        std::vector<Value> mValues;
        Value mMin = 0.f;
        Value mMax = 1.f;
        Value mInverseRange = 1.f;
    };

}