#include "neuralfunctions.h"

#include <cstdlib>
#include <cmath>
#include <cassert>

namespace neural
{

    Value sigmoid(Value x)
    {
        return 1.0 / (1.0 + exp(-x));
    }


    Value sigmoidDerivative(Value x)
    {
        return sigmoid(x) * (1 - sigmoid(x));
    }


    Value random(Value low, Value high)
    {
        Value result = low + static_cast <Value>(rand()) / static_cast <Value>(RAND_MAX / (high-low));
        assert(result >= low && result <= high);
        return result;
    }


    Sigmoid::Sigmoid(Value min, Value max, int n) : mMin(min), mMax(max)
    {
        Value inc = (max - min) / Value(n);
        Value x = min;
        for (auto i = 0; i < n; ++i)
        {
            mValues.emplace_back(sigmoid(x));
            x += inc;
        }
        mInverseRange = 1.f / (mMax - mMin);
    }


    Value Sigmoid::get(Value x)
    {
        assert(x >= mMin && x <= mMax);
        float proportion = (x - mMin) * mInverseRange;
        float index = proportion * (mValues.size() - 1);
        int left = int(index);
        int right = left + 1;
        float lerpValue = index - left;
        return mValues[left] * (1.f - lerpValue) + lerpValue * mValues[right];
    }


    Value Sigmoid::getDerivative(Value x)
    {
        return get(x) * (1 - get(x));
    }



}
