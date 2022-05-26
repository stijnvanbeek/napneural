#include "neuralfunctions.h"

#include <cstdlib>
#include <cmath>

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
        return low + static_cast <Value>(rand()) / static_cast <Value>(RAND_MAX / (high-low));
    }


}
