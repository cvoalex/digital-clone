//
//  MelProcessor.swift
//  mels
//
//  Audio pipeline mel spectrogram processor
//

import Foundation
import Accelerate
import AVFoundation

/// Mel Spectrogram Processor using Accelerate framework
class MelProcessor {
    // Parameters matching Python/Go implementations
    let sampleRate: Int = 16000
    let nFFT: Int = 800
    let hopLength: Int = 200
    let winLength: Int = 800
    let nMels: Int = 80
    let fmin: Float = 55.0
    let fmax: Float = 7600.0
    let preemphasisCoef: Float = 0.97
    let refLevelDB: Float = 20.0
    let minLevelDB: Float = -100.0
    let maxAbsValue: Float = 4.0
    
    private let melBasis: [[Float]]
    private let hannWindow: [Float]
    
    init() {
        // Build mel filterbank
        self.melBasis = Self.buildMelBasis(
            nFFT: nFFT,
            nMels: nMels,
            sampleRate: sampleRate,
            fmin: fmin,
            fmax: fmax
        )
        
        // Pre-compute Hann window
        self.hannWindow = Self.makeHannWindow(size: winLength)
        
        print("MelProcessor initialized")
        print("  Sample rate: \(sampleRate) Hz")
        print("  N-FFT: \(nFFT), Hop: \(hopLength)")
        print("  Mel bands: \(nMels)")
    }
    
    /// Load audio file using simple WAV loader
    func loadAudio(url: URL) throws -> [Float] {
        print("Loading WAV file: \(url.lastPathComponent)")
        
        let samples = try SimpleWAVLoader.loadWAV(url: url)
        
        print("Loaded audio: \(samples.count) samples, \(Float(samples.count)/Float(sampleRate))s")
        
        return samples
    }
    
    /// Apply pre-emphasis filter
    func preEmphasis(_ audio: [Float]) -> [Float] {
        var output = [Float](repeating: 0, count: audio.count)
        output[0] = audio[0]
        
        for i in 1..<audio.count {
            output[i] = audio[i] - preemphasisCoef * audio[i-1]
        }
        
        return output
    }
    
    /// Compute STFT using Accelerate vDSP (FAST!)
    func stft(_ audio: [Float]) -> [[Float]] {
        let numFrames = (audio.count - winLength) / hopLength + 1
        let fftSize = nFFT / 2 + 1
        let log2n = vDSP_Length(log2(Float(nFFT)))
        
        print("Computing STFT: \(numFrames) frames...")
        
        var magnitudes = Array(
            repeating: Array(repeating: Float(0), count: numFrames),
            count: fftSize
        )
        
        // Set up FFT
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            print("Failed to create FFT setup")
            return magnitudes
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }
        
        // Allocate buffers once
        var realBuffer = [Float](repeating: 0, count: nFFT / 2)
        var imagBuffer = [Float](repeating: 0, count: nFFT / 2)
        
        // Process each frame
        for frameIdx in 0..<numFrames {
            if frameIdx % 100 == 0 {
                print("  Frame \(frameIdx)/\(numFrames)...")
            }
            
            let start = frameIdx * hopLength
            let end = start + winLength
            
            guard end <= audio.count else { break }
            
            // Apply window
            var frame = [Float](repeating: 0, count: nFFT)
            vDSP_vmul(
                Array(audio[start..<end]), 1,
                hannWindow, 1,
                &frame, 1,
                vDSP_Length(winLength)
            )
            
            // Perform real FFT
            frame.withUnsafeBytes { frameBytes in
                var complexBuffer = DSPSplitComplex(realp: &realBuffer, imagp: &imagBuffer)
                
                // Convert to split complex format
                frameBytes.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: nFFT/2) { complexPtr in
                    vDSP_ctoz(complexPtr, 2, &complexBuffer, 1, vDSP_Length(nFFT/2))
                }
                
                // Perform FFT
                vDSP_fft_zrip(fftSetup, &complexBuffer, 1, log2n, FFTDirection(FFT_FORWARD))
                
                // Calculate magnitude: sqrt(real^2 + imag^2)
                var mags = [Float](repeating: 0, count: nFFT / 2)
                vDSP_zvmags(&complexBuffer, 1, &mags, 1, vDSP_Length(nFFT / 2))
                
                // Take square root (copy to avoid overlap)
                var sqrtMags = mags
                var count = Int32(nFFT / 2)
                vvsqrtf(&sqrtMags, &mags, &count)
                mags = sqrtMags
                
                // Store magnitudes
                for i in 0..<min(fftSize, mags.count) {
                    magnitudes[i][frameIdx] = mags[i]
                }
            }
        }
        
        print("✓ STFT complete")
        
        return magnitudes
    }
    
    /// Convert linear spectrogram to mel scale
    func linearToMel(_ spectrogram: [[Float]]) -> [[Float]] {
        let numFrames = spectrogram[0].count
        var melSpec = Array(
            repeating: Array(repeating: Float(0), count: numFrames),
            count: nMels
        )
        
        // Matrix multiplication: melBasis × spectrogram
        for melIdx in 0..<nMels {
            for frameIdx in 0..<numFrames {
                var sum: Float = 0
                for freqIdx in 0..<min(spectrogram.count, melBasis[melIdx].count) {
                    sum += melBasis[melIdx][freqIdx] * spectrogram[freqIdx][frameIdx]
                }
                melSpec[melIdx][frameIdx] = sum
            }
        }
        
        return melSpec
    }
    
    /// Convert amplitude to dB
    func ampToDb(_ spec: [[Float]]) -> [[Float]] {
        let minLevel = Float(exp(-5.0 * log(10.0)))
        
        return spec.map { row in
            row.map { amp in
                Float(20.0) * log10(max(minLevel, amp))
            }
        }
    }
    
    /// Normalize spectrogram to [-4, 4]
    func normalize(_ spec: [[Float]]) -> [[Float]] {
        return spec.map { row in
            row.map { val in
                let normalized = (2.0 * maxAbsValue) * ((val - minLevelDB) / (-minLevelDB)) - maxAbsValue
                return max(-maxAbsValue, min(maxAbsValue, normalized))
            }
        }
    }
    
    /// Process audio to mel spectrogram
    func process(_ audio: [Float]) -> [[Float]] {
        print("Processing audio: \(audio.count) samples")
        
        // 1. Pre-emphasis
        let preEmphasized = preEmphasis(audio)
        
        // 2. STFT (returns magnitude)
        let magnitude = stft(preEmphasized)
        
        // 3. Linear to Mel
        let melSpec = linearToMel(magnitude)
        
        // 4. Amplitude to dB
        var melDB = ampToDb(melSpec)
        
        // 5. Apply reference level
        for i in 0..<melDB.count {
            for j in 0..<melDB[i].count {
                melDB[i][j] -= refLevelDB
            }
        }
        
        // 6. Normalize
        let normalized = normalize(melDB)
        
        print("Mel spectrogram shape: (\(normalized.count), \(normalized[0].count))")
        let minVal = normalized.flatMap { $0 }.min() ?? 0
        let maxVal = normalized.flatMap { $0 }.max() ?? 0
        print("  Value range: [\(minVal), \(maxVal)]")
        
        return normalized
    }
    
    // MARK: - Helper Functions
    
    private static func makeHannWindow(size: Int) -> [Float] {
        var window = [Float](repeating: 0, count: size)
        for i in 0..<size {
            window[i] = 0.5 * (1.0 - cos(2.0 * .pi * Float(i) / Float(size - 1)))
        }
        return window
    }
    
    private static func freqToMel(_ freq: Float) -> Float {
        return 2595.0 * log10(1.0 + freq / 700.0)
    }
    
    private static func melToFreq(_ mel: Float) -> Float {
        return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }
    
    private static func buildMelBasis(
        nFFT: Int,
        nMels: Int,
        sampleRate: Int,
        fmin: Float,
        fmax: Float
    ) -> [[Float]] {
        let nFreqs = nFFT / 2 + 1
        
        // FFT frequencies
        var fftFreqs = [Float](repeating: 0, count: nFreqs)
        for i in 0..<nFreqs {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(nFFT)
        }
        
        // Mel scale conversion
        let melMin = freqToMel(fmin)
        let melMax = freqToMel(fmax)
        
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            melPoints[i] = melMin + (melMax - melMin) * Float(i) / Float(nMels + 1)
        }
        
        let freqPoints = melPoints.map { melToFreq($0) }
        
        // Build filterbank
        var filterbank = Array(
            repeating: Array(repeating: Float(0), count: nFreqs),
            count: nMels
        )
        
        for melIdx in 0..<nMels {
            let leftFreq = freqPoints[melIdx]
            let centerFreq = freqPoints[melIdx + 1]
            let rightFreq = freqPoints[melIdx + 2]
            
            for freqIdx in 0..<nFreqs {
                let freq = fftFreqs[freqIdx]
                
                if freq >= leftFreq && freq <= centerFreq {
                    filterbank[melIdx][freqIdx] = (freq - leftFreq) / (centerFreq - leftFreq)
                } else if freq >= centerFreq && freq <= rightFreq {
                    filterbank[melIdx][freqIdx] = (rightFreq - freq) / (rightFreq - centerFreq)
                }
            }
        }
        
        // Normalize
        for i in 0..<nMels {
            let enorm = 2.0 / (freqPoints[i + 2] - freqPoints[i])
            for j in 0..<nFreqs {
                filterbank[i][j] *= enorm
            }
        }
        
        return filterbank
    }
    
    // MARK: - Frame extraction methods
    
    func cropAudioWindow(melSpec: [[Float]], frameIdx: Int, fps: Double) throws -> [[Float]] {
        // melSpec is [80][time_frames], need to extract window from time dimension
        let startIdx = Int(80.0 * (Double(frameIdx) / fps))
        var endIdx = startIdx + 16
        
        let timeFrames = melSpec[0].count
        if endIdx > timeFrames {
            endIdx = timeFrames
        }
        
        // Extract columns startIdx to endIdx from all 80 mel bins
        var window: [[Float]] = []
        for melBin in 0..<80 {
            var frameWindow: [Float] = []
            for t in startIdx..<endIdx {
                frameWindow.append(melSpec[melBin][t])
            }
            window.append(frameWindow)
        }
        
        // Transpose to get [16][80] format (16 frames, 80 mels each)
        var transposed: [[Float]] = []
        for t in 0..<(endIdx - startIdx) {
            var frame: [Float] = []
            for melBin in 0..<80 {
                frame.append(window[melBin][t])
            }
            transposed.append(frame)
        }
        
        // Pad if needed
        if transposed.count < 16 {
            for _ in 0..<(16 - transposed.count) {
                transposed.append([Float](repeating: 0, count: 80))
            }
        }
        
        return transposed
    }
    
    func getFrameCount(melSpec: [[Float]], fps: Double) -> Int {
        // melSpec is [80][time_frames], we need the time dimension
        let timeFrames = melSpec[0].count
        return Int((Double(timeFrames) - 16.0) / 80.0 * fps) + 2
    }
}
