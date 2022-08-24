#pragma once

#include <neuraltypes.h>
#include <neuralfunctions.h>
#include <link.h>
#include <vector>

namespace neural
{

    class Neuron
    {
    public:
        struct Delta {
            std::vector<Value> mWeights;
            Value mBias;
        };

    public:
        Neuron(size_t neuronId, Sigmoid& sigmoid) : mId(neuronId), mSigmoid(sigmoid) { }
        virtual ~Neuron() = default;

        void addOutput(Neuron& output);

		void setValue(Value value) { mValue = value; }
        Value getValue() const { return mValue; }
		Value getDerivativeValue() const { return mDerivativeValue; }

        void setBias(Value value) { mBias = value; }
        Value getBias() const { return mBias; }

		void setBiasBackPropagationMemory(Value value) { mBiasBackPropagationMemory = value; }
		Value getBiasBackPropagationMemory() const { return mBiasBackPropagationMemory; }

		void calculateValue();
		Delta getBackPropagationDeltas(const std::vector<Value>& targetOutputValues);

		size_t getInputCount() const { return mInputs.size(); }
		Link& getInput(size_t index) { return *mInputs[index]; }

    private:
        std::vector<Link*> mInputs;
        std::vector<std::unique_ptr<Link>> mOutputs;
        Value mBias = 0.f;
		Value mBiasBackPropagationMemory = 0.f;
        Value mValue = 0.f;
		Value mDerivativeValue = 0.f;
		size_t mId = 0;
		Sigmoid& mSigmoid;
    };

}