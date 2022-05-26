#include "link.h"

#include <neuron.h>

namespace neural
{

    Value Link::getValue() const
    {
        return mWeight * mSource->getValue();
    }

}
