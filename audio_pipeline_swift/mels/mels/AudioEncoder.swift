//
//  AudioEncoder.swift
//  mels
//
//  ONNX Runtime integration - NATIVE, NO PYTHON!
//

import Foundation

/// AudioEncoder using native ONNX Runtime
class AudioEncoder {
    private let session: ONNXRuntimeSession
    
    init(modelPath: String) throws {
        self.session = try ONNXRuntimeSession(modelPath: modelPath)
        print("✓ AudioEncoder initialized (Native ONNX Runtime)")
        print("  Model: \(modelPath)")
    }
    
    /// Process a mel window through ONNX
    func process(melWindow: [[Float]]) throws -> [Float] {
        // Flatten mel window (16, 80) to input format (1, 1, 80, 16)
        var inputData = [Float](repeating: 0, count: 1 * 1 * 80 * 16)
        
        // Transpose: (16, 80) -> (1, 1, 80, 16) format
        var idx = 0
        for melIdx in 0..<80 {
            for frameIdx in 0..<16 {
                inputData[idx] = melWindow[frameIdx][melIdx]
                idx += 1
            }
        }
        
        // Run inference
        let output = try session.run(input: inputData)
        
        return output
    }
    
    /// Process batch of mel windows
    func processBatch(melWindows: [[[Float]]]) throws -> [[Float]] {
        var results: [[Float]] = []
        
        for (idx, window) in melWindows.enumerated() {
            if idx % 100 == 0 {
                print("  Processing window \(idx)/\(melWindows.count)...")
            }
            let features = try process(melWindow: window)
            results.append(features)
        }
        
        return results
    }
}

/// Complete Audio Pipeline
class AudioPipeline {
    private let melProcessor: MelProcessor
    private var audioEncoder: AudioEncoder?
    let fps: Int
    let mode: String
    
    init(modelPath: String, fps: Int = 25, mode: String = "ave") throws {
        self.melProcessor = MelProcessor()
        self.fps = fps
        self.mode = mode
        
        // Initialize audio encoder
        self.audioEncoder = try AudioEncoder(modelPath: modelPath)
        
        print("AudioPipeline initialized (FULL PIPELINE - NATIVE ONNX)")
        print("  FPS: \(fps), Mode: \(mode)")
    }
    
    /// Process audio file through COMPLETE pipeline
    func processAudioFile(url: URL) async throws -> ProcessedAudio {
        print("\n" + String(repeating: "=", count: 60))
        print("Processing Audio File (FULL PIPELINE)")
        print(String(repeating: "=", count: 60))
        
        // Step 1: Load audio
        print("Step 1: Loading audio...")
        let audio = try melProcessor.loadAudio(url: url)
        
        // Step 2: Convert to mel spectrogram
        print("Step 2: Converting to mel spectrogram...")
        let melSpec = melProcessor.process(audio)
        
        // Step 3: Extract mel windows
        print("Step 3: Extracting mel windows...")
        let nFrames = getFrameCount(melSpec: melSpec)
        print("  Total frames: \(nFrames)")
        
        var melWindows: [[[Float]]] = []
        for frameIdx in 0..<nFrames {
            if frameIdx % 500 == 0 {
                print("  Extracting window \(frameIdx)/\(nFrames)...")
            }
            let window = try cropAudioWindow(melSpec: melSpec, frameIdx: frameIdx)
            melWindows.append(window)
        }
        print("✓ Windows extracted")
        
        // Step 4: Process through AudioEncoder
        print("Step 4: Processing through ONNX AudioEncoder...")
        guard let encoder = audioEncoder else {
            throw NSError(domain: "AudioPipeline", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "AudioEncoder not initialized"])
        }
        
        let audioFeatures = try encoder.processBatch(melWindows: melWindows)
        
        // Step 5: Add temporal padding
        print("Step 5: Adding temporal padding...")
        let paddedFeatures = addTemporalPadding(audioFeatures)
        
        print("✓ Processing complete!")
        print("  Output shape: (\(paddedFeatures.count), \(paddedFeatures[0].count))")
        
        return ProcessedAudio(
            melSpec: melSpec,
            audioFeatures: paddedFeatures,
            nFrames: nFrames
        )
    }
    
    private func addTemporalPadding(_ features: [[Float]]) -> [[Float]] {
        var padded: [[Float]] = []
        padded.append(features[0])  // Repeat first
        padded.append(contentsOf: features)
        padded.append(features[features.count - 1])  // Repeat last
        return padded
    }
    
    private func getFrameCount(melSpec: [[Float]]) -> Int {
        let nMelFrames = melSpec[0].count
        return Int((Float(nMelFrames) - 16.0) / 80.0 * Float(fps)) + 2
    }
    
    private func cropAudioWindow(melSpec: [[Float]], frameIdx: Int) throws -> [[Float]] {
        var startIdx = Int(80.0 * Float(frameIdx) / Float(fps))
        var endIdx = startIdx + 16
        
        let nFrames = melSpec[0].count
        
        // Adjust if window goes past the end
        if endIdx > nFrames {
            endIdx = nFrames
            startIdx = endIdx - 16
            
            if startIdx < 0 {
                throw NSError(domain: "AudioPipeline", code: 2,
                             userInfo: [NSLocalizedDescriptionKey: "Frame index out of range"])
            }
        }
        
        // Extract window and transpose to (16, 80)
        var window: [[Float]] = Array(repeating: Array(repeating: 0, count: 80), count: 16)
        for i in 0..<16 {
            let melFrameIdx = startIdx + i
            if melFrameIdx < nFrames {
                for j in 0..<80 {
                    window[i][j] = melSpec[j][melFrameIdx]
                }
            }
        }
        
        return window
    }
}

/// Processed audio results
struct ProcessedAudio {
    let melSpec: [[Float]]
    let audioFeatures: [[Float]]
    let nFrames: Int
    
    var stats: [String: Any] {
        let melValues = melSpec.flatMap { $0 }
        let featValues = audioFeatures.flatMap { $0 }
        
        return [
            "mel_shape": [melSpec.count, melSpec[0].count],
            "mel_range": [melValues.min() ?? 0, melValues.max() ?? 0],
            "features_shape": [audioFeatures.count, audioFeatures[0].count],
            "features_range": [featValues.min() ?? 0, featValues.max() ?? 0],
            "n_frames": nFrames
        ]
    }
}
