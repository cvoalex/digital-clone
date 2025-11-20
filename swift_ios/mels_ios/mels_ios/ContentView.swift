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
                
                // Load audio and process
                statusMessage = "Processing audio..."
                guard let audioURL = Bundle.main.url(forResource: "talk_hb", withExtension: "wav") else {
                    throw NSError(domain: "App", code: 1, userInfo: [NSLocalizedDescriptionKey: "Audio file not found"])
                }
                
                let audioFeatures = try generator.processAudio(audioPath: audioURL.path)
                statusMessage = "Audio processed: \(audioFeatures.count) frames"
                
                // Generate frames
                let numFrames = min(250, audioFeatures.count)
                statusMessage = "Generating \(numFrames) frames..."
                let startTime = Date()
                
                var frames: [UIImage] = []
                
                for i in 1...numFrames {
                    progress = Double(i) / Double(numFrames)
                    
                    if i % 25 == 0 {
                        await MainActor.run {
                            statusMessage = "Frame \(i)/\(numFrames)..."
                        }
                    }
                    
                    // Load images from bundle
                    guard let roiURL = Bundle.main.url(forResource: "roi_\(i)", withExtension: "jpg"),
                          let maskedURL = Bundle.main.url(forResource: "masked_\(i)", withExtension: "jpg"),
                          let roiImage = UIImage(contentsOfFile: roiURL.path),
                          let maskedImage = UIImage(contentsOfFile: maskedURL.path) else {
                        print("Warning: Missing images for frame \(i)")
                        continue
                    }
                    
                    // Generate frame with REAL Core ML
                    let audioIdx = min(i - 1, audioFeatures.count - 1)
                    let generatedImage = try generator.generateFrame(
                        roiImage: roiImage,
                        maskedImage: maskedImage,
                        audioFeatures: audioFeatures[audioIdx]
                    )
                    
                    frames.append(generatedImage)
                }
                
                let elapsed = Date().timeIntervalSince(startTime)
                fps = Double(frames.count) / elapsed
                
                await MainActor.run {
                    generatedFrames = frames
                    statusMessage = "Complete! \(frames.count) frames at \(String(format: "%.1f", fps)) FPS"
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

