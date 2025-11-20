import Foundation
import AppKit

class FrameGenerator {
    private let sandersDir: String
    private let audioEncoder: ONNXModel
    private let generator: ONNXModel
    private let cropRectangles: [String: CropRect]
    
    struct CropRect: Codable {
        let rect: [Int]  // [x1, y1, x2, y2]
    }
    
    init(sandersDir: String) throws {
        self.sandersDir = sandersDir
        
        // Load models
        print("Loading models...")
        self.audioEncoder = try ONNXModel(
            modelPath: "\(sandersDir)/models/audio_encoder.onnx",
            inputNames: ["mel"],
            outputNames: ["emb"]
        )
        
        self.generator = try ONNXModel(
            modelPath: "\(sandersDir)/models/generator.onnx",
            inputNames: ["input", "audio"],
            outputNames: ["output"]
        )
        
        // Load crop rectangles
        let cropPath = "\(sandersDir)/cache/crop_rectangles.json"
        let cropData = try Data(contentsOf: URL(fileURLWithPath: cropPath))
        self.cropRectangles = try JSONDecoder().decode([String: CropRect].self, from: cropData)
        
        print("✓ Models loaded successfully")
    }
    
    func processAudio(audioPath: String) throws -> [[Float]] {
        print("Processing audio: \(audioPath)")
        
        // Load WAV
        let wavLoader = try SimpleWAVLoader(path: audioPath)
        let samples = wavLoader.samples
        print("  Loaded audio: \(samples.count) samples")
        
        // Process mel spectrograms
        let melProcessor = MelProcessor()
        let melSpec = melProcessor.process(audioSamples: samples)
        print("  Generated mel spectrogram: \(melSpec.count) x \(melSpec[0].count)")
        
        // Calculate number of frames
        let fps = 25.0
        let numFrames = melProcessor.getFrameCount(melSpec: melSpec, fps: fps)
        print("  Number of frames: \(numFrames)")
        
        // Encode each frame
        var audioFeatures: [[Float]] = []
        
        for i in 0..<numFrames {
            // Crop audio window
            let melWindow = try melProcessor.cropAudioWindow(melSpec: melSpec, frameIdx: i, fps: fps)
            
            // Flatten to (1, 1, 80, 16) format
            let melTensor = flattenMelWindow(melWindow)
            
            // Run audio encoder
            let features = try audioEncoder.predict(inputs: [melTensor])
            audioFeatures.append(features[0])
            
            if (i + 1) % 100 == 0 {
                print("  Encoded \(i + 1)/\(numFrames) frames")
            }
        }
        
        print("✓ Generated \(audioFeatures.count) audio feature frames")
        return audioFeatures
    }
    
    func generateFrames(audioFeatures: [[Float]], numFrames: Int, outputDir: String) throws {
        print("Generating \(numFrames) frames...")
        
        // Create output directory
        try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
        
        for i in 1...numFrames {
            if i % 50 == 1 {
                print("Processing frame \(i)/\(numFrames)...")
            }
            
            // Load pre-cut frames
            let roiPath = "\(sandersDir)/rois_320/\(i).jpg"
            let maskedPath = "\(sandersDir)/model_inputs/\(i).jpg"
            let fullBodyPath = "\(sandersDir)/full_body_img/\(i).jpg"
            
            guard let roiImage = NSImage(contentsOfFile: roiPath),
                  let maskedImage = NSImage(contentsOfFile: maskedPath),
                  let fullBodyImage = NSImage(contentsOfFile: fullBodyPath) else {
                throw NSError(domain: "FrameGenerator", code: 1, 
                            userInfo: [NSLocalizedDescriptionKey: "Failed to load images for frame \(i)"])
            }
            
            // Convert to tensors (BGR format)
            let roiTensor = try imageToTensor(roiImage, normalize: true)
            let maskedTensor = try imageToTensor(maskedImage, normalize: true)
            
            // Concatenate to 6-channel input
            let imageTensor = roiTensor + maskedTensor
            
            // Get audio features and reshape
            let audioIdx = i - 1
            let audioFeat = audioFeatures[min(audioIdx, audioFeatures.count - 1)]
            let audioTensor = reshapeAudioFeatures(audioFeat)
            
            // Run generator
            let output = try generator.predict(inputs: [imageTensor, audioTensor])
            let outputTensor = output[0].map { $0 * 255.0 }
            
            // Convert to image
            let generatedImage = try tensorToImage(outputTensor, width: 320, height: 320)
            
            // Get crop rectangle
            let rectKey = String(i - 1)
            guard let cropRect = cropRectangles[rectKey] else {
                throw NSError(domain: "FrameGenerator", code: 2,
                            userInfo: [NSLocalizedDescriptionKey: "No crop rect for frame \(i)"])
            }
            
            // Paste into full frame
            let finalFrame = try pasteIntoFrame(fullBodyImage, generated: generatedImage, rect: cropRect.rect)
            
            // Save
            let outputPath = "\(outputDir)/frame_\(String(format: "%05d", i)).jpg"
            try saveImage(finalFrame, path: outputPath)
        }
        
        print("✓ Generated \(numFrames) frames successfully!")
    }
    
    // Helper functions
    
    private func flattenMelWindow(_ melWindow: [[Float]]) -> [Float] {
        // Input: (16, 80) mel window
        // Output: (1, 1, 80, 16) flattened = 1280 floats
        var result: [Float] = []
        
        for mel in 0..<80 {
            for frame in 0..<16 {
                result.append(melWindow[frame][mel])
            }
        }
        
        return result
    }
    
    private func imageToTensor(_ image: NSImage, normalize: Bool) throws -> [Float] {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw NSError(domain: "FrameGenerator", code: 3)
        }
        
        let width = cgImage.width
        let height = cgImage.height
        
        // Get pixel data
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw NSError(domain: "FrameGenerator", code: 4)
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Convert to CHW BGR format
        var tensor = [Float](repeating: 0, count: 3 * width * height)
        let scale: Float = normalize ? (1.0 / 255.0) : 1.0
        
        for c in 0..<3 {
            for y in 0..<height {
                for x in 0..<width {
                    let pixelIndex = (y * width + x) * 4
                    let tensorIndex = c * height * width + y * width + x
                    
                    // RGB to BGR: swap channels
                    let srcChannel = 2 - c  // R=0→B=2, G=1→G=1, B=2→R=0
                    let value = Float(pixelData[pixelIndex + srcChannel]) * scale
                    tensor[tensorIndex] = value
                }
            }
        }
        
        return tensor
    }
    
    private func tensorToImage(_ tensor: [Float], width: Int, height: Int) throws -> NSImage {
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        
        // Convert from CHW BGR to HWC RGB
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = (y * width + x) * 4
                
                // BGR to RGB: swap channels
                let b = UInt8(min(255, max(0, tensor[0 * height * width + y * width + x])))
                let g = UInt8(min(255, max(0, tensor[1 * height * width + y * width + x])))
                let r = UInt8(min(255, max(0, tensor[2 * height * width + y * width + x])))
                
                pixelData[pixelIndex + 0] = r
                pixelData[pixelIndex + 1] = g
                pixelData[pixelIndex + 2] = b
                pixelData[pixelIndex + 3] = 255
            }
        }
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(pixelData) as CFData),
              let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
              ) else {
            throw NSError(domain: "FrameGenerator", code: 5)
        }
        
        return NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
    }
    
    private func pasteIntoFrame(_ fullFrame: NSImage, generated: NSImage, rect: [Int]) throws -> NSImage {
        let x1 = rect[0], y1 = rect[1], x2 = rect[2], y2 = rect[3]
        
        guard let fullCG = fullFrame.cgImage(forProposedRect: nil, context: nil, hints: nil),
              let genCG = generated.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw NSError(domain: "FrameGenerator", code: 6)
        }
        
        let width = fullCG.width
        let height = fullCG.height
        
        // Create new image
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw NSError(domain: "FrameGenerator", code: 7)
        }
        
        // Draw full frame
        context.draw(fullCG, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Draw generated region (resized to fit rect)
        let targetRect = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
        context.draw(genCG, in: targetRect)
        
        guard let resultCG = context.makeImage() else {
            throw NSError(domain: "FrameGenerator", code: 8)
        }
        
        return NSImage(cgImage: resultCG, size: NSSize(width: width, height: height))
    }
    
    private func saveImage(_ image: NSImage, path: String) throws {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw NSError(domain: "FrameGenerator", code: 9)
        }
        
        let url = URL(fileURLWithPath: path)
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, kUTTypeJPEG, 1, nil) else {
            throw NSError(domain: "FrameGenerator", code: 10)
        }
        
        CGImageDestinationAddImage(destination, cgImage, [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary)
        
        if !CGImageDestinationFinalize(destination) {
            throw NSError(domain: "FrameGenerator", code: 11)
        }
    }
    
    private func reshapeAudioFeatures(_ features: [Float]) -> [Float] {
        // Reshape 512 features to (32, 16, 16) = 8192
        var result = [Float](repeating: 0, count: 32 * 16 * 16)
        
        for i in 0..<result.count {
            result[i] = features[i % 512]
        }
        
        return result
    }
}

