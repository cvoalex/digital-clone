//
//  ContentView.swift
//  mels
//
//  Created by Alexander Rusich on 11/18/25.
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var audioURL: URL?
    @State private var processingStatus: String = "Ready"
    @State private var melSpec: [[Float]]?
    @State private var processedAudio: ProcessedAudio?
    @State private var isProcessing: Bool = false
    @State private var showFileImporter: Bool = false
    @State private var useFullPipeline: Bool = false
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Audio Pipeline - macOS")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            Text("Mel Spectrogram Processor")
                .font(.title3)
                .foregroundColor(.secondary)
            
            Divider()
            
            // Status display
            VStack(alignment: .leading, spacing: 10) {
                Text("Status:")
                    .font(.headline)
                
                Text(processingStatus)
                    .font(.body)
                    .foregroundColor(isProcessing ? .orange : .primary)
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
            }
            
            // Results display
            if let melSpec = melSpec {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Mel Spectrogram Results:")
                        .font(.headline)
                    
                    Text("Shape: (\(melSpec.count), \(melSpec[0].count))")
                    
                    let values = melSpec.flatMap { $0 }
                    if let min = values.min(), let max = values.max() {
                        Text("Range: [\(String(format: "%.3f", min)), \(String(format: "%.3f", max))]")
                    }
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.green.opacity(0.1))
                .cornerRadius(8)
            }
            
            // Full pipeline results
            if let processed = processedAudio {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Full Pipeline Results:")
                        .font(.headline)
                    
                    Text("Frames: \(processed.nFrames)")
                    Text("Audio Features: (\(processed.audioFeatures.count), \(processed.audioFeatures[0].count))")
                    
                    let featValues = processed.audioFeatures.flatMap { $0 }
                    if let min = featValues.min(), let max = featValues.max() {
                        Text("Features Range: [\(String(format: "%.3f", min)), \(String(format: "%.3f", max))]")
                    }
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.blue.opacity(0.1))
                .cornerRadius(8)
            }
            
            Spacer()
            
            // Info message
            Text("Native ONNX Runtime - NO PYTHON!")
                .font(.caption)
                .foregroundColor(.green)
                .padding(.horizontal)
            
            Toggle("Use Full Pipeline (Mel + ONNX)", isOn: $useFullPipeline)
                .padding(.horizontal)
            
            // Action buttons
            VStack(spacing: 15) {
                Button(action: {
                    showFileImporter = true
                }) {
                    Label("Select Audio File", systemImage: "doc.badge.plus")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .disabled(isProcessing)
                
                if audioURL != nil {
                    Button(action: {
                        processAudio()
                    }) {
                        Label(isProcessing ? "Processing..." : (useFullPipeline ? "Run Full Pipeline" : "Process Mel Only"), 
                              systemImage: useFullPipeline ? "gearshape.2" : "waveform")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(isProcessing ? Color.gray : Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .disabled(isProcessing)
                }
                
                Button(action: {
                    testWithDemoAudio()
                }) {
                    Label("Test with Demo Audio", systemImage: "play.circle")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.purple)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .disabled(isProcessing)
            }
        }
        .padding(30)
        .frame(minWidth: 500, minHeight: 500)
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let files):
                if let file = files.first {
                    audioURL = file
                    processingStatus = "Selected: \(file.lastPathComponent)"
                }
            case .failure(let error):
                processingStatus = "Error selecting file: \(error.localizedDescription)"
            }
        }
    }
    
    private func processAudio() {
        guard let url = audioURL else { return }
        
        isProcessing = true
        processingStatus = "Processing audio..."
        
        Task {
            do {
                if useFullPipeline {
                    // Full pipeline with ONNX
                    processingStatus = "Initializing pipeline..."
                    
                    let modelPath = "/Users/alexanderrusich/Projects/digital-clone/audio_pipeline_swift/mels/audio_encoder.onnx"
                    
                    let pipeline = try AudioPipeline(modelPath: modelPath, fps: 25, mode: "ave")
                    
                    processingStatus = "Running full pipeline..."
                    let result = try await pipeline.processAudioFile(url: url)
                    
                    await MainActor.run {
                        melSpec = result.melSpec
                        processedAudio = result
                        processingStatus = "✅ Full pipeline complete!"
                        isProcessing = false
                    }
                } else {
                    // Just mel spectrogram
                    let processor = MelProcessor()
                    
                    processingStatus = "Loading audio file..."
                    let audio = try processor.loadAudio(url: url)
                    
                    processingStatus = "Computing mel spectrogram..."
                    let mel = processor.process(audio)
                    
                    await MainActor.run {
                        melSpec = mel
                        processedAudio = nil
                        processingStatus = "✅ Mel processing complete!"
                        isProcessing = false
                    }
                }
                
            } catch {
                await MainActor.run {
                    processingStatus = "❌ Error: \(error.localizedDescription)"
                    isProcessing = false
                }
            }
        }
    }
    
    private func testWithDemoAudio() {
        // Try to find demo audio from the Python tests
        let demoPath = "../../../../audio_pipeline/test_data/reference_audio.wav"
        
        if let basePath = Bundle.main.resourcePath {
            let fullPath = URL(fileURLWithPath: basePath).appendingPathComponent(demoPath)
            if FileManager.default.fileExists(atPath: fullPath.path) {
                audioURL = fullPath
                processingStatus = "Selected demo audio"
                processAudio()
                return
            }
        }
        
        processingStatus = "Demo audio not found. Please select an audio file."
    }
}

#Preview {
    ContentView()
}
