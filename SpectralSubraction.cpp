/*
 ==============================================================================
 
 Main.cpp
 Created: 1 Feb 2021 1:01:16pm
 Author:  Alexx Mitchell
 
 ==============================================================================
 */

#include "DSPProcessor.h"
#include <algorithm>
#include <math.h>
#include <fstream>

Denoiser::Denoiser()
: lowPassFilter (dsp::IIR::Coefficients<float>::makeLowPass (44100, 20000.0f, 0.1f))
{
    
    File noise_profile = File::getSpecialLocation (File::currentExecutableFile)
    .getParentDirectory()
    .getChildFile("external")
    .getChildFile ("noise_averages.json"); //copy this file to executable directory
    std::string noise_json_path = noise_profile.getFullPathName().toStdString();
    std::ifstream ifs (noise_json_path);
    skin_data = nlohmann::json::parse (ifs);
}

Denoiser::~Denoiser() {}


void Denoiser::timerCallback()
{
    if (nextFFTBlockReady)
    {
        nextFFTBlockReady = false;
    }
}


void Denoiser::pushNextSampleIntoFifo (float sample) noexcept
{
    // if the fifo contains enough data, set a flag to say
    // that the next frame should now be rendered..
    if (fifoIndex == fftSize)
    {
        if (! nextFFTBlockReady)
        {
            juce::zeromem (inputFftData, sizeof (inputFftData));
            memcpy (inputFftData, fifo, sizeof (fifo));
            nextFFTBlockReady = true;
            timerCallback();
        }
        
        fifoIndex = 0;
    }
    
    fifo[fifoIndex++] = sample;
    
}


void Denoiser::updateFilter (float voiceSkinIntensityLevel)
{
    // This formula was derived by plotting intensity values in relation to the filter cutoff frequency using the Denoiser VST and a low pass filter. Plotted to a curve using (website).
    float cutoff =
    ((7500 - ((7750 * (voiceSkinIntensityLevel - 2)) / 3)) * (voiceSkinIntensityLevel - 1)
     - 16000)
    * voiceSkinIntensityLevel
    + 20000;
    *lowPassFilter.state =
    *dsp::IIR::Coefficients<float>::makeLowPass (lastSampleRate, cutoff, 0.1f);
}

void Denoiser::fillSpectralBuffer (int channel, const int bufferLength, const float* bufferData)
{
    spectralBuffer.copyFrom (channel, 0, bufferData, bufferLength);
}

void Denoiser::getFromSpectralBuffer (AudioBuffer<float>& buffer,
                                      int channel,
                                      const int bufferLength,
                                      const int spectralBufferLength,
                                      float voiceSkinIntensityLevel,
                                      int frame)
{
    int numSamples = buffer.getNumSamples();
    float spectralData[fftSize * 2] = { 0 };
    
    for (int sample = 0; sample < numSamples; sample++)
    {
        spectralData[sample] = spectralBuffer.getSample (channel, sample);
    }
    
    // perform Forward FFT, The output is interleaved (real1, imag1, real2,…)
    forward_fft.performRealOnlyForwardTransform (spectralData);
    
    auto noise_avg =
    std::complex<float> (skin_data["frame"][frame][0], skin_data["frame"][frame][1]);
    
    // get magnitude spectrum, phase spectrum
    for (int bin = 0; bin < fftSize; bin++)
    {
        int scale = bin * 2;
        std::complex<float> j;
        j.imag (1.0);
        j.real (0.0);
        in_val[bin] = std::complex<float> (spectralData[scale], spectralData[scale + 1]);
        phase[bin] = std::arg (in_val[bin]);
        input_mag[bin] = std::abs (in_val[bin]);
        phase_info[bin] = exp ((1.0f * j) * phase[bin]);
        
        // spectral subtraction
        denoised_mag[bin] = input_mag[bin] - (noise_avg * voiceSkinIntensityLevel); // Re-use
        clipped_val[bin] =
        std::complex<float> (std::clamp (denoised_mag[bin].real(), 0.0f, 1000.0f),
                             std::clamp (denoised_mag[bin].imag(), 0.0f, 1000.0f)); //Re-use
        
        // add phase info
        out_val[bin] = phase_info[bin] * clipped_val[bin];
        
        spectralData[scale] = out_val[bin].real();
        spectralData[scale + 1] = out_val[bin].imag();
    }
    
    // perform inverse transform on result
    inverse_fft.performRealOnlyInverseTransform (spectralData);
    
    buffer.copyFrom (channel, 0, spectralData, bufferLength);
}

void Denoiser::prepare (const ProcessSpec& spec)
{
    const int numOutputChannels = spec.numChannels;
    const int spectralBufferSize = (spec.sampleRate + spec.maximumBlockSize);
    lastSampleRate = (double) spec.sampleRate;
    
    //Update Spectral Buffer
    spectralBuffer.setSize (numOutputChannels, spectralBufferSize);
    
    //reset process spec
    lowPassFilter.prepare (spec);
}

void Denoiser::process (AudioBuffer<float>& buffer, AudioProcessorValueTreeState& tree)
{
    auto totalNumInputChannels = buffer.getNumChannels();
    
    int num_frames = skin_data["num_frames"];
    
    //Get Buffer Lengths
    const int bufferLength = buffer.getNumSamples();
    const int spectralBufferLength = spectralBuffer.getNumSamples();
    dsp::AudioBlock<float> block (buffer);
    
    voiceSkinIntensityLevel = *tree.getRawParameterValue("intensity");
    
    for (int channel = 0; channel < totalNumInputChannels; channel++)
    {
        const float* channelData = buffer.getReadPointer (channel, 0);
        fillSpectralBuffer (channel, bufferLength, channelData);
        getFromSpectralBuffer (
                               buffer, channel, bufferLength, spectralBufferLength, voiceSkinIntensityLevel, frame);
    }
    
    updateFilter (voiceSkinIntensityLevel);
    lowPassFilter.process (dsp::ProcessContextReplacing<float> (block));
    
    // Loop noise_averages.json
    frame = (frame + 1) % num_frames;
}

void Denoiser::reset() { lowPassFilter.reset(); }
