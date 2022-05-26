#pragma once

#include <neuraltypes.h>

namespace neural
{

    Value sigmoid(Value x);
    Value sigmoidDerivative(Value x);
    Value random(Value low, Value high);

}