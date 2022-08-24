#pragma once

#include <neuron.h>
#include <neuralfunctions.h>
#include <mutex>
#include <thread>
#include <atomic>

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
        Network();
        virtual ~Network();

        void addLayer(std::size_t size);
		void randomize();

        void train(const std::vector<Result>& data, int epochs, int miniBatchSize, Value learningRate, bool log = false);
        void trainAsync(const std::vector<Result>& data, int miniBatchSize, Value learningRate, bool log = false);
        void train(const std::vector<const Result*>& data, Value learningRate);
        std::vector<Value> process(const std::vector<Value>& inputValues);
		std::vector<Value> getErrorMargins(const std::vector<Result>& data);

    private:
        void processInputValues(const std::vector<Value>& inputValues);
        std::vector<Value> getOutputValues() const;
        void trainEpoch(std::vector<const Result*>& shuffledData, int miniBatchSize, Value learningRate, bool log = false, int epochNr = 0);

        std::vector<Layer> mLayers;
        Sigmoid mSigmoid;

        std::mutex mMutex;
        std::thread mTrainThread;
        std::vector<Result> mAsyncTrainData;
        std::atomic<std::vector<Result>*> mAsyncTrainDataReady = { nullptr };
        std::atomic<int> mAsyncMiniBatchSize = 0;
        std::atomic<float> mAsyncLearningRate = 1.f;
        bool mAsyncLog = false;
        bool mStopping = false;
    };

}


