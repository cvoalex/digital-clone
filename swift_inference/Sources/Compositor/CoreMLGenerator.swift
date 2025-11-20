import Foundation
import CoreML
import AppKit

@available(macOS 13.0, *)
class CoreMLFrameGenerator {
    private let sandersDir: String
    private let audioEncoder: AudioEncoder
    private let generator: Generator
    private let cropRectangles: [String: CropRect]
    private let melProcessor: MelProcessor
    
    struct CropRect: Codable {
        let rect: [Int]
    }
    
    init(sandersDir: String) throws {
        self.sandersDir = sandersDir
        
        print("Loading Core ML models...")
        
        // Load Core ML models
        let audioEncoderURL = URL(fileURLWithPath: "swift_inference/AudioEncoder.mlpackage")
        let generatorURL = URL(fileURLWithPath: "swift_inference/Generator.mlpackage")
        
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Use CPU + GPU + Neural Engine!
        
        self.audioEncoder = try AudioEncoder(configuration: config)
        self.generator = try Generator(configuration: config)
        
        print("✓ Core ML models loaded (Neural Engine enabled!)")
        
        // Load crop rectangles
        let cropPath = "\(sandersDir)/cache/crop_rectangles.json"
        let cropData = try Data(contentsOf: URL(fileURLWithPath: cropPath))
        self.cropRectangles = try JSONDecoder().decode([String: CropRect].self, from: cropData)
        
        // Initialize mel processor
        self.melProcessor = MelProcessor()
        
        print("✓ All resources loaded")
    }
    
    func processAudio(audioPath: String) throws -> [MLMultiArray] {
        print("Processing audio: \(audioPath)")
        let startTime = Date()
        
        // Load WAV
        let wavLoader = try SimpleWAVLoader(path: audioPath)
        let samples = wavLoader.samples
        print("  Loaded: \(samples.count) samples")
        
        // Process mel spectrograms
        let melSpec = melProcessor.process(audioSamples: samples)
        print("  Mel spectrogram: \(melSpec.count) x \(melSpec[0].count)")
        
        // Get frame count
        let fps = 25.0
        let numFrames = melProcessor.getFrameCount(melSpec: melSpec, fps: fps)
        print("  Frames: \(numFrames)")
        
        // Encode each frame
        var audioFeatures: [MLMultiArray] = []
        
        for i in 0..<numFrames {
            // Crop mel window
            let melWindow = try melProcessor.cropAudioWindow(melSpec: melSpec, frameIdx: i, fps: fps)
            
            // Convert to MLMultiArray shape [1, 1, 80, 16]
            let melArray = try createMLMultiArray(shape: [1, 1, 80, 16], data: flattenMelWindow(melWindow))
            
            // Run audio encoder (Core ML!)
            let input = AudioEncoderInput(mel: melArray)
            let output = try audioEncoder.prediction(input: input)
            
            audioFeatures.append(output.emb)
            
            if (i + 1) % 100 == 0 {
                print("  Encoded \(i + 1)/\(numFrames)")
            }
        }
        
        let elapsed = Date().timeIntervalSince(startTime)
        print("✓ Audio processing: \(String(format: "%.2f", elapsed))s (\(String(format: "%.1f", Double(numFrames)/elapsed)) FPS)")
        
        return audioFeatures
    }
    
    func generateFrames(audioFeatures: [MLMultiArray], numFrames: Int, outputDir: String) throws {
        print("Generating \(numFrames) frames...")
        let startTime = Date()
        
        // Create output directory
        try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
        
        for i in 1...numFrames {
            if i % 50 == 1 {
                print("Processing frame \(i)/\(numFrames)...")
            }
            
            // Load pre-cut images
            let roiPath = "\(sandersDir)/rois_320/\(i).jpg"
            let maskedPath = "\(sandersDir)/model_inputs/\(i).jpg"
            let fullBodyPath = "\(sandersDir)/full_body_img/\(i).jpg"
            
            guard let roiImage = NSImage(contentsOfFile: roiPath),
                  let maskedImage = NSImage(contentsOfFile: maskedPath),
                  let fullBodyImage = NSImage(contentsOfFile: fullBodyPath) else {
                throw NSError(domain: "Generator", code: 1)
            }
            
            // Convert to MLMultiArrays
            let roiArray = try imageToMLMultiArray(roiImage, normalize: true)
            let maskedArray = try imageToMLMultiArray(maskedImage, normalize: true)
            
            // Concatenate to 6-channel input [1, 6, 320, 320]
            let imageInput = try concatenateArrays(roiArray, maskedArray)
            
            // Get audio features and reshape to [1, 32, 16, 16]
            let audioIdx = min(i - 1, audioFeatures.count - 1)
            let audioInput = try reshapeAudioFeatures(audioFeatures[audioIdx])
            
            // Run generator (Core ML on Neural Engine!)
            let genInput = GeneratorInput(input: imageInput, audio: audioInput)
            let genOutput = try generator.prediction(input: genInput)
            
            // Convert output to image
            let generatedImage = try mlMultiArrayToImage(genOutput.output, width: 320, height: 320)
            
            // Paste into full frame
            let rectKey = String(i - 1)
            guard let cropRect = cropRectangles[rectKey] else {
                throw NSError(domain: "Generator", code: 2)
            }
            
            let finalFrame = try pasteIntoFrame(fullBodyImage, generated: generatedImage, rect: cropRect.rect)
            
            // Save
            let outputPath = "\(outputDir)/frame_\(String(format: "%05d", i)).jpg"
            try saveJPEG(finalFrame, path: outputPath)
        }
        
        let elapsed = Date().timeIntervalSince(startTime)
        print("✓ Frame generation: \(String(format: "%.2f", elapsed))s (\(String(format: "%.1f", Double(numFrames)/elapsed)) FPS)")
    }
    
    // MARK: - Helper Functions
    
    private func createMLMultiArray(shape: [Int], data: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape as [NSNumber], dataType: .float32)
        for (index, value) in data.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }
    
    private func flattenMelWindow(_ melWindow: [[Float]]) -> [Float] {
        var result: [Float] = []
        for mel in 0..<80 {
            for frame in 0..<16 {
                result.append(melWindow[frame][mel])
            }
        }
        return result
    }
    
    private func imageToMLMultiArray(_ image: NSImage, normalize: Bool) throws -> MLMultiArray {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw NSError(domain: "Generator", code: 3)
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
            throw NSError(domain: "Generator", code: 4)
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Convert to CHW BGR format in MLMultiArray [1, 3, height, width]
        let array = try MLMultiArray(shape: [1, 3, height, width] as [NSNumber], dataType: .float32)
        let scale: Float = normalize ? (1.0 / 255.0) : 1.0
        
        for c in 0..<3 {
            for y in 0..<height {
                for x in 0..<width {
                    let pixelIndex = (y * width + x) * 4
                    let arrayIndex = [0, c, y, x] as [NSNumber]
                    
                    // RGB to BGR
                    let srcChannel = 2 - c
                    let value = Float(pixelData[pixelIndex + srcChannel]) * scale
                    array[arrayIndex] = NSNumber(value: value)
                }
            }
        }
        
        return array
    }
    
    private func concatenateArrays(_ arr1: MLMultiArray, _ arr2: MLMultiArray) throws -> MLMultiArray {
        // Concatenate along channel dimension [1,3,H,W] + [1,3,H,W] = [1,6,H,W]
        let result = try MLMultiArray(shape: [1, 6, 320, 320] as [NSNumber], dataType: .float32)
        
        // Copy first 3 channels from arr1
        for c in 0..<3 {
            for y in 0..<320 {
                for x in 0..<320 {
                    let srcIdx = [0, c, y, x] as [NSNumber]
                    let dstIdx = [0, c, y, x] as [NSNumber]
                    result[dstIdx] = arr1[srcIdx]
                }
            }
        }
        
        // Copy next 3 channels from arr2
        for c in 0..<3 {
            for y in 0..<320 {
                for x in 0..<320 {
                    let srcIdx = [0, c, y, x] as [NSNumber]
                    let dstIdx = [0, c + 3, y, x] as [NSNumber]
                    result[dstIdx] = arr2[srcIdx]
                }
            }
        }
        
        return result
    }
    
    private func reshapeAudioFeatures(_ features: MLMultiArray) throws -> MLMultiArray {
        // Input: [1, 512]
        // Output: [1, 32, 16, 16] = 8192 values
        let result = try MLMultiArray(shape: [1, 32, 16, 16] as [NSNumber], dataType: .float32)
        
        // Tile/repeat the 512 values to fill 8192
        for i in 0..<(32*16*16) {
            let srcIdx = i % 512
            result[i] = features[srcIdx]
        }
        
        return result
    }
    
    private func mlMultiArrayToImage(_ array: MLMultiArray, width: Int, height: Int) throws -> NSImage {
        // Input: [1, 3, 320, 320] in BGR format, values 0-1
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = (y * width + x) * 4
                
                // BGR to RGB
                let bIdx = [0, 0, y, x] as [NSNumber]
                let gIdx = [0, 1, y, x] as [NSNumber]
                let rIdx = [0, 2, y, x] as [NSNumber]
                
                let b = UInt8(min(255, max(0, array[bIdx].floatValue * 255)))
                let g = UInt8(min(255, max(0, array[gIdx].floatValue * 255)))
                let r = UInt8(min(255, max(0, array[rIdx].floatValue * 255)))
                
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
            throw NSError(domain: "Generator", code: 5)
        }
        
        return NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
    }
    
    private func pasteIntoFrame(_ fullFrame: NSImage, generated: NSImage, rect: [Int]) throws -> NSImage {
        let x1 = rect[0], y1 = rect[1], x2 = rect[2], y2 = rect[3]
        
        guard let fullCG = fullFrame.cgImage(forProposedRect: nil, context: nil, hints: nil),
              let genCG = generated.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw NSError(domain: "Generator", code: 6)
        }
        
        let width = fullCG.width
        let height = fullCG.height
        
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
            throw NSError(domain: "Generator", code: 7)
        }
        
        // Draw full frame
        context.draw(fullCG, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Draw generated (resized to fit rect)
        let targetRect = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
        context.draw(genCG, in: targetRect)
        
        guard let resultCG = context.makeImage() else {
            throw NSError(domain: "Generator", code: 8)
        }
        
        return NSImage(cgImage: resultCG, size: NSSize(width: width, height: height))
    }
    
    private func saveJPEG(_ image: NSImage, path: String) throws {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw NSError(domain: "Generator", code: 9)
        }
        
        let url = URL(fileURLWithPath: path)
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, kUTTypeJPEG, 1, nil) else {
            throw NSError(domain: "Generator", code: 10)
        }
        
        CGImageDestinationAddImage(destination, cgImage, [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary)
        
        if !CGImageDestinationFinalize(destination) {
            throw NSError(domain: "Generator", code: 11)
        }
    }
}

