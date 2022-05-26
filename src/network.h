#pragma once

#include <neuron.h>

namespace neural
{

    class Network
    {
    public:
        using Layer = std::vector<std::unique_ptr<Neuron>>;

        struct Result
        {
			Result(const std::vector<Value>& inputs, const std::vector<Value>& outputs) : mInputs(inputs), mOutputs(outputs) { }
            std::vector<Value> mInputs;
            std::vector<Value> mOutputs;
        };

    public:
        Network() = default;
        virtual ~Network() = default;

        void addLayer(std::size_t size);
        void process(const std::vector<Value>& inputValues);
		std::vector<Value> getOutputValues() const;
		void randomize();

        void train(const std::vector<Result>& data, int epochs, int miniBatchSize, Value learningRate, bool log = false);
		void train(const std::vector<const Result*>& data, Value learningRate);
		std::vector<Value> getErrorMargins(const std::vector<Result>& data);

    private:
        std::vector<Layer> mLayers;
    };

}


