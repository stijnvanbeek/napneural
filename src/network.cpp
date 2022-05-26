#include "network.h"

#include <neuralfunctions.h>
#include <cassert>
#include <string>
#include <iostream>
#include <cmath>

namespace neural
{

    void Network::addLayer(std::size_t size)
    {
        Layer* previousLayer = nullptr;

        mLayers.emplace_back();
		if (mLayers.size() > 1)
			previousLayer = &mLayers[mLayers.size() - 2];
        auto& newLayer = mLayers.back();

        for (auto i = 0; i < size; ++i)
        {
            auto newNeuron = std::make_unique<Neuron>(i, mSigmoid);
            if (previousLayer)
            {
                for (auto& previousNeuron : *previousLayer)
                    previousNeuron->addOutput(*newNeuron);
            }
            newLayer.emplace_back(std::move(newNeuron));
        }
    }


    void Network::process(const std::vector<Value>& inputValues)
    {
        assert(mLayers.size() > 1);
        assert(inputValues.size() == mLayers.begin()->size());

        auto& inputLayer = *mLayers.begin();
        for (auto i = 0; i < inputValues.size(); ++i)
            inputLayer[i]->setValue(inputValues[i]);

        for (auto layerIndex = 1; layerIndex < mLayers.size(); ++layerIndex) {
            for (auto &neuron : mLayers[layerIndex])
                neuron->calculateValue();
        }
    }


	std::vector<Value> Network::getOutputValues() const
	{
		std::vector<Value> outputValues;
		auto& outputLayer = mLayers.back();
		outputValues.resize(outputLayer.size());
		for (auto i = 0; i < outputLayer.size(); ++i)
			outputValues[i] = outputLayer[i]->getValue();
		return outputValues;
	}


	void Network::randomize()
	{
		for (auto& layer : mLayers)
			for (auto& neuron : layer)
			{
				for (auto i = 0; i < neuron->getInputCount(); ++i)
					neuron->getInput(i).setWeight(neural::random(0.f, 1.f));
			}
	}


    void Network::train(const std::vector<Result>& data, int epochs, int miniBatchSize, Value learningRate, bool log)
    {
        std::vector<const Result*> shuffledData;
        for (auto& result : data)
            shuffledData.emplace_back(&result);

        // Loop through epochs
        for (auto epoch = 0; epoch < epochs; ++epoch)
        {
            // Shuffle the test data
            for (auto i = 0; i < shuffledData.size(); ++i)
            {
				int j = neural::random(0, shuffledData.size() - 1);
				std::swap(shuffledData[i], shuffledData[j]);
            }

			// Split shuffled training data up into mini batches
			std::vector<std::vector<const Result*>> miniBatches;
			miniBatches.emplace_back();
			for (auto i = 0; i < shuffledData.size(); ++i)
			{
				if (miniBatches.back().size() < miniBatchSize)
					miniBatches.back().emplace_back(shuffledData[i]);
				else
					miniBatches.emplace_back();
			}

			// Loop through mini batches
			for (auto& batch : miniBatches)
			{
				train(batch, learningRate);
			}

			// Log the current input and output data
			if (log)
			{
				std::string output;
				output.append("epoch " + std::to_string(epoch));
				output.append(" input (");
				for (auto& neuron : *mLayers.begin())
				{
					output.append(std::to_string(neuron->getValue()));
					if (neuron != mLayers.begin()->back())
						output.append(", ");
				}
				output.append(") output (");
				for (auto& neuron : mLayers.back())
				{
					output.append(std::to_string(neuron->getValue()));
					if (neuron != mLayers.back().back())
						output.append(", ");
				}
				output.append(")");
				std::cout << output << std::endl;
			}
        }
    }


	void Network::train(const std::vector<const Result*>& data, Value learningRate)
	{
		// Create and zero the weight and bias delta data
		std::vector<std::vector<std::vector<Value>>> mWeightDeltas(mLayers.size());
		std::vector<std::vector<Value>> mBiasDeltas(mLayers.size());
		for (auto i = 0; i < mLayers.size(); ++i)
		{
			auto& layerWeightDeltas = mWeightDeltas[i];
			layerWeightDeltas.resize(mLayers[i].size());
			auto& layerBiasDeltas = mBiasDeltas[i];
			layerBiasDeltas.resize(mLayers[i].size(), 0.f);
			for (auto neuronIndex = 0; neuronIndex < mLayers[i].size(); ++neuronIndex)
				layerWeightDeltas[neuronIndex].resize(mLayers[i][neuronIndex]->getInputCount(), 0.f);
		}

		// Perform back propagation for all data pairs
		for (auto& result : data)
		{
			process(result->mInputs);

			// Loop backwards through all layers, include the output layer, exclude the input layer
			for (auto layerIndex = mLayers.size() - 1; layerIndex > 0; --layerIndex)
			{
				auto& layer = mLayers[layerIndex];
				for (auto neuronIndex = 0; neuronIndex < layer.size(); ++neuronIndex)
				{
					auto& neuron = layer[neuronIndex];
					auto& neuronWeightDeltas = mWeightDeltas[layerIndex][neuronIndex];
					auto neuronBackProp = std::move(neuron->getBackPropagationDeltas(result->mOutputs));
					for (auto i = 0; i < neuronBackProp.mWeights.size(); ++i)
						neuronWeightDeltas[i] += neuronBackProp.mWeights[i];
					mBiasDeltas[layerIndex][neuronIndex] += neuronBackProp.mBias;
				}
			}
		}

		// Divide the weight and bias deltas by the batch size to get averages
		for (auto& layerDeltas : mWeightDeltas)
			for (auto& neuronDeltas : layerDeltas)
				for (auto& weightDelta : neuronDeltas)
					weightDelta /= data.size();

		for (auto& layerDeltas : mBiasDeltas)
			for (auto& biasDelta : layerDeltas)
				biasDelta /= data.size();

		// Shift the weights and biases using the delta data
		for (auto layerIndex = 0; layerIndex < mLayers.size(); layerIndex++)
		{
			auto& layer = mLayers[layerIndex];
			for (auto neuronIndex = 0; neuronIndex < layer.size(); neuronIndex++)
			{
				auto& neuron = layer[neuronIndex];
				for (auto inputIndex = 0; inputIndex < neuron->getInputCount(); ++inputIndex)
				{
					auto& input = neuron->getInput(inputIndex);
					input.setWeight(input.getWeight() + mWeightDeltas[layerIndex][neuronIndex][inputIndex] * learningRate);
				}
				neuron->setBias(mBiasDeltas[layerIndex][neuronIndex]);
			}
		}

	}


	std::vector<Value> Network::getErrorMargins(const std::vector<Result>& data)
	{
		std::vector<Value> result(data.begin()->mOutputs.size(), 0.f);

		for (auto i = 0; i < data.size(); ++i)
		{
			process(data[i].mInputs);
			auto outputValues = getOutputValues();
			for (auto outputIndex = 0; outputIndex < result.size(); ++outputIndex)
				result[outputIndex] += fabs(data[i].mOutputs[outputIndex] - outputValues[outputIndex]);
		}

		for (auto& value : result)
			value /= data.size();

		return result;
	}

}
