import SwiftUI
import CoreML

struct ContentView: View {
    @State private var isProcessing = false
    @State private var progress: Double = 0
    @State private var generatedFrames: [UIImage] = []
    @State private var statusMessage = "Ready to generate frames"
    @State private var fps: Double = 0
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Frame Generator")
                    .font(.largeTitle)
                    .bold()
                
                Text(statusMessage)
                    .foregroundColor(.secondary)
                
                if isProcessing {
                    ProgressView(value: progress) {
                        Text("Generating frames...")
                    }
                    .padding()
                }
                
                if fps > 0 {
                    Text("\(String(format: "%.1f", fps)) FPS")
                        .font(.title2)
                        .foregroundColor(.green)
                }
                
                Button(action: {
                    generateFrames()
                }) {
                    Text(isProcessing ? "Processing..." : "Generate Demo Frames")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(isProcessing ? Color.gray : Color.blue)
                        .cornerRadius(10)
                }
                .disabled(isProcessing)
                .padding(.horizontal)
                
                if !generatedFrames.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 10) {
                            ForEach(0..<generatedFrames.count, id: \.self) { index in
                                Image(uiImage: generatedFrames[index])
                                    .resizable()
                                    .aspectRatio(contentMode: .fit)
                                    .frame(height: 200)
                                    .cornerRadius(8)
                            }
                        }
                        .padding()
                    }
                }
                
                Spacer()
            }
            .padding()
            .navigationBarTitleDisplayMode(.inline)
        }
    }
    
    func generateFrames() {
        isProcessing = true
        statusMessage = "Initializing..."
        progress = 0
        generatedFrames = []
        
        Task {
            do {
                statusMessage = "Loading models..."
                let generator = try FrameGeneratorIOS()
                
                // Load audio and process (only what we need!)
                statusMessage = "Processing audio..."
                guard let audioURL = Bundle.main.url(forResource: "talk_hb", withExtension: "wav") else {
                    throw NSError(domain: "App", code: 1, userInfo: [NSLocalizedDescriptionKey: "Audio file not found"])
                }
                
                let numFrames = 250
                let audioFeatures = try await generator.processAudio(audioPath: audioURL.path, maxFrames: numFrames)
                statusMessage = "Audio processed: \(audioFeatures.count) frames"
                statusMessage = "Generating \(numFrames) frames (parallel)..."
                let startTime = Date()
                
                // Parallel frame generation
                var frames: [UIImage?] = Array(repeating: nil, count: numFrames)
                
                await withTaskGroup(of: (Int, UIImage?).self) { group in
                    for i in 1...numFrames {
                        group.addTask {
                            do {
                                guard let roiURL = Bundle.main.url(forResource: "roi_\(i)", withExtension: "jpg"),
                                      let maskedURL = Bundle.main.url(forResource: "masked_\(i)", withExtension: "jpg"),
                                      let roiImage = UIImage(contentsOfFile: roiURL.path),
                                      let maskedImage = UIImage(contentsOfFile: maskedURL.path) else {
                                    return (i - 1, nil)
                                }
                                
                                    let audioIdx = min(i - 1, audioFeatures.count - 1)
                                    let generatedImage = try generator.generateFrame(
                                        roiImage: roiImage,
                                        maskedImage: maskedImage,
                                        audioFeatures: audioFeatures[audioIdx],
                                        roiCacheKey: "roi_\(i)",
                                        maskedCacheKey: "masked_\(i)"
                                    )
                                
                                return (i - 1, generatedImage)
                            } catch {
                                print("Error frame \(i): \(error)")
                                return (i - 1, nil)
                            }
                        }
                        
                        if i % 50 == 0 {
                            await MainActor.run {
                                statusMessage = "Generating frame \(i)/\(numFrames)..."
                                progress = Double(i) / Double(numFrames)
                            }
                        }
                    }
                    
                    // Collect results in order
                    for await (index, image) in group {
                        frames[index] = image
                    }
                }
                
                let validFrames = frames.compactMap { $0 }
                
                let elapsed = Date().timeIntervalSince(startTime)
                fps = Double(validFrames.count) / elapsed
                
                await MainActor.run {
                    generatedFrames = validFrames
                    statusMessage = "Complete! \(validFrames.count) frames at \(String(format: "%.1f", fps)) FPS"
                    isProcessing = false
                }
                
            } catch {
                await MainActor.run {
                    statusMessage = "Error: \(error.localizedDescription)"
                    isProcessing = false
                }
            }
        }
    }
}

#Preview {
    ContentView()
}

