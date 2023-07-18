/*
  ==============================================================================

    Main.h
    Created: 1 Feb 2021 1:01:16pm
    Author:  Alexx Mitchell

  ==============================================================================
*/

#pragma once
#include <JuceHeader.h>
#include "external/json.hpp"
#include <complex>
#include <valarray>

using namespace juce;

using namespace dsp;

class Denoiser : private juce::Timer
{
public:
    Denoiser();
    ~Denoiser();

    void timerCallback() override;
    void pushNextSampleIntoFifo (float sample) noexcept;
    void updateFilter (float intensityLevel);
    void fillSpectralBuffer (int channel, const int bufferLength, const float* bufferData);
    void getFromSpectralBuffer (AudioBuffer<float>& buffer,
                                int channel,
                                const int bufferLength,
                                const int spectralBufferLength,
                                float intensityLevel,
                                int frame);
    enum
    {
        fftOrder = 10,
        fftSize = 1 << fftOrder
    };

    dsp::ProcessorDuplicator<dsp::IIR::Filter<float>, dsp::IIR::Coefficients<float>> lowPassFilter;
    float voiceSkinIntensityLevel = 0;

    // Core Functions
    void prepare (const ProcessSpec& spec);
    void reset();
    void process (AudioBuffer<float>& buffer, AudioProcessorValueTreeState &tree);

private:
    FFT forward_fft{ fftOrder };
    FFT inverse_fft{ fftOrder };

    AudioBuffer<float> spectralBuffer;
    float lastSampleRate;
    int frame = 0;
    
    AudioFormatReader* noise_reader;

    std::complex<float> in_val[fftSize] = { 0 };
    std::complex<float> out_val[fftSize] = { 0 };
    std::complex<float> phase[fftSize] = { 0 };
    std::complex<float> phase_info[fftSize] = { 0 };
    std::complex<float> input_mag[fftSize] = { 0 };
    std::complex<float> denoised_mag[fftSize] = { 0 };
    std::complex<float> clipped_val[fftSize] = { 0 };

    nlohmann::json skin_data;
    
    float fifo [fftSize];
    float inputFftData [2 * fftSize] = {0};
    int fifoIndex = 0;
    bool nextFFTBlockReady = false;
};
