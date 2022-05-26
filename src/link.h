#pragma once

#include <neuraltypes.h>

namespace neural
{

    // Forward declarations
    class Neuron;

    class Link
    {
    public:
        Link(Neuron& source, Neuron& dest) : mSource(&source), mDest(&dest) { }
        virtual ~Link() = default;

        void setWeight(Value value) { mWeight = value; }
        Value getWeight() const { return mWeight; }

		void setWeightBackPropagationMemory(Value value) { mWeightBackPropagationMemory = value; }
		Value getWeightBackPropagationMemory() const { return mWeightBackPropagationMemory; }

		Value getValue() const;

		Neuron& getSource() { return *mSource; }
		Neuron& getDestination()  { return *mDest; }

    private:
        Value mWeight = 1.f;
		Value mWeightBackPropagationMemory = 0.f;
        Neuron* mSource = nullptr;
        Neuron* mDest = nullptr;
    };


}


