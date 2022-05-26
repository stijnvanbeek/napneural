#include "neuron.h"

namespace neural
{

    void Neuron::addOutput(Neuron &output)
    {
        auto link = std::make_unique<Link>(*this, output);
        output.mInputs.emplace_back(link.get());
        mOutputs.emplace_back(std::move(link));
    }


    void Neuron::calculateValue()
    {
        Value rawValue = 0.f;
        for (auto& input : mInputs)
			rawValue += input->getValue();
		rawValue += mBias;
        mValue = mSigmoid.get(rawValue);
		mDerivativeValue = mSigmoid.getDerivative(rawValue);
//        mValue = sigmoid(rawValue);
//        mDerivativeValue = sigmoidDerivative(rawValue);
    }


	Neuron::Delta Neuron::getBackPropagationDeltas(const std::vector<Value>& targetOutputValues)
	{
		Delta result;
		result.mWeights.resize(mInputs.size(), 0.f);

		// Is this an output layer neuron?
		if (mOutputs.empty())
		{
			Value d = (getValue() - targetOutputValues[mId]) * getDerivativeValue();
			for (auto i = 0; i < mInputs.size(); ++i)
			{
				result.mWeights[i] = -d * mInputs[i]->getSource().getValue();
				mInputs[i]->setWeightBackPropagationMemory(d);
			}
			result.mBias = -d * mBias;
//			setBiasBackPropagationMemory(d1 * d2);
		}

		// this is a hidden layer neuron
		else {
			float d = 0;
			for (auto i = 0; i < mOutputs.size(); ++i)
				d += mOutputs[i]->getWeightBackPropagationMemory() * mOutputs[i]->getWeight();
			d *= getDerivativeValue();
			for (auto i = 0; i < mInputs.size(); i++)
			{
				result.mWeights[i] = -d * mInputs[i]->getSource().getValue();
				mInputs[i]->setWeightBackPropagationMemory(d);
			}
			result.mBias = -d * mBias;
			setBiasBackPropagationMemory(d);
		}

		return result;
	}

}
