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
                
                statusMessage = "Generating demo frames..."
                let startTime = Date()
                
                // Generate a few demo frames
                // In production, would load from bundle resources
                let numFrames = 10
                var frames: [UIImage] = []
                
                for i in 1...numFrames {
                    progress = Double(i) / Double(numFrames)
                    statusMessage = "Generating frame \(i)/\(numFrames)..."
                    
                    // Would call actual generation here with real images/audio
                    // For demo, create placeholder
                    let size = CGSize(width: 320, height: 320)
                    UIGraphicsBeginImageContext(size)
                    UIColor.gray.setFill()
                    UIRectFill(CGRect(origin: .zero, size: size))
                    let demoImage = UIGraphicsGetImageFromCurrentImageContext()!
                    UIGraphicsEndImageContext()
                    
                    frames.append(demoImage)
                    
                    // Simulate processing time
                    try await Task.sleep(nanoseconds: 20_000_000)
                }
                
                let elapsed = Date().timeIntervalSince(startTime)
                fps = Double(numFrames) / elapsed
                
                await MainActor.run {
                    generatedFrames = frames
                    statusMessage = "Complete! Generated \(numFrames) frames"
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

