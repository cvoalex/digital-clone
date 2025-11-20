import Foundation
import CoreML

// Check macOS version
guard #available(macOS 13.0, *) else {
    print("Error: Requires macOS 13.0 or later")
    exit(1)
}

// Parse command line arguments
let args = CommandLine.arguments

var sandersDir = "../model/sanders_full_onnx"
var audioPath = ""  // Will default to sanders/aud.wav if not specified
var outputDir = "../comparison_results/swift_output/frames"
var numFrames = 250

// Simple argument parsing
var i = 1
while i < args.count {
    switch args[i] {
    case "--sanders":
        if i + 1 < args.count {
            sandersDir = args[i + 1]
            i += 1
        }
    case "--audio":
        if i + 1 < args.count {
            audioPath = args[i + 1]
            i += 1
        }
    case "--output":
        if i + 1 < args.count {
            outputDir = args[i + 1]
            i += 1
        }
    case "--frames":
        if i + 1 < args.count {
            numFrames = Int(args[i + 1]) ?? 250
            i += 1
        }
    case "--help", "-h":
        print("""
        Swift Inference - Frame Generation
        
        Usage:
          swift-infer [options]
        
        Options:
          --sanders PATH    Sanders directory (default: ../model/sanders_full_onnx)
          --audio PATH      Audio WAV file (default: sanders/aud.wav)
          --output PATH     Output directory (default: ../comparison_results/swift_output/frames)
          --frames N        Number of frames to generate (default: 250)
          --help, -h        Show this help
        """)
        exit(0)
    default:
        break
    }
    i += 1
}

// Set default audio if not specified
if audioPath.isEmpty {
    audioPath = "\(sandersDir)/aud.wav"
}

print("============================================================")
print("Swift Core ML Inference - Sanders Frame Generation")
print("============================================================")
print("Sanders directory: \(sandersDir)")
print("Audio file: \(audioPath)")
print("Output directory: \(outputDir)")
print("Number of frames: \(numFrames)")
print("============================================================")
print("Using Core ML with Neural Engine! 🚀")
print("============================================================")

let startTime = Date()

do {
    // Create frame generator
    print("\n[1/4] Loading Core ML models...")
    let generator = try FrameGeneratorCoreML(sandersDir: sandersDir)
    
    // Process audio
    print("\n[2/4] Processing audio with Core ML...")
    let audioFeatures = try generator.processAudio(audioPath: audioPath)
    
    // Limit frames
    let actualFrames = min(numFrames, audioFeatures.count)
    
    // Generate frames
    print("\n[3/4] Generating video frames with Core ML...")
    try generator.generateFrames(
        audioFeatures: audioFeatures,
        numFrames: actualFrames,
        outputDir: outputDir
    )
    
    let endTime = Date()
    let elapsed = endTime.timeIntervalSince(startTime)
    
    print("\n[4/4] Complete!")
    print("============================================================")
    print("✓ Generated \(actualFrames) frames in \(String(format: "%.2f", elapsed))s")
    print("✓ Performance: \(String(format: "%.1f", Double(actualFrames) / elapsed)) FPS")
    print("============================================================")
    print("\nTo create video, run:")
    print("  ffmpeg -framerate 25 -i \(outputDir)/frame_%05d.jpg \\")
    print("    -i \(audioPath) \\")
    print("    -vframes \(actualFrames) -shortest \\")
    print("    -c:v libx264 -c:a aac -crf 20 \\")
    print("    swift_output.mp4 -y")
    
} catch {
    print("\nError: \(error)")
    print("\nNote: Swift implementation requires ONNX Runtime")
    print("Make sure ONNXWrapper.swift is properly configured")
    exit(1)
}
