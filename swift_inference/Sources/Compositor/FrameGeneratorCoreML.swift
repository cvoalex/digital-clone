import Foundation
import CoreML
import AppKit

// Simple wrapper since we're using the models directly
@available(macOS 13.0, *)
class FrameGeneratorCoreML {
    private let sandersDir: String
    private let audioEncoderModel: MLModel
    private let generatorModel: MLModel
    private let cropRectangles: [String: CropRect]
    private let melProcessor: MelProcessor
    private let tensorCache: TensorCache
    private let metalDevice: MTLDevice?
    private let metalQueue: MTLCommandQueue?
    
    struct CropRect: Codable {
        let rect: [Int]
    }
    
    init(sandersDir: String) throws {
        self.sandersDir = sandersDir
        
        print("Loading Core ML models...")
        
        // Load compiled Core ML models
        let config = MLModelConfiguration()
        
        // Explicitly request Neural Engine + GPU
        if #available(macOS 13.0, *) {
            config.computeUnits = .cpuAndNeuralEngine  // Try Neural Engine first
        } else {
            config.computeUnits = .all
        }
        
        config.allowLowPrecisionAccumulationOnGPU = true  // Enable GPU optimizations
        
        print("  Requesting compute units: CPU + Neural Engine + GPU")
        
        let audioEncoderURL = URL(fileURLWithPath: "AudioEncoder.mlmodelc", relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
        let generatorURL = URL(fileURLWithPath: "Generator.mlmodelc", relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
        
        print("  Loading: \(audioEncoderURL.path)")
        self.audioEncoderModel = try MLModel(contentsOf: audioEncoderURL, configuration: config)
        print("  Audio encoder loaded")
        
        print("  Loading: \(generatorURL.path)")
        self.generatorModel = try MLModel(contentsOf: generatorURL, configuration: config)
        print("  Generator loaded")
        
        // Try to verify what's actually being used
        if #available(macOS 14.0, *) {
            print("  Model configuration: \(config.computeUnits)")
        }
        
        print("✓ Core ML models loaded")
        
        // Load crop rectangles
        let cropPath = "\(sandersDir)/cache/crop_rectangles.json"
        let cropData = try Data(contentsOf: URL(fileURLWithPath: cropPath))
        self.cropRectangles = try JSONDecoder().decode([String: CropRect].self, from: cropData)
        
        // Initialize mel processor
        self.melProcessor = MelProcessor()
        
        // Initialize Metal for GPU operations
        self.metalDevice = MTLCreateSystemDefaultDevice()
        self.metalQueue = metalDevice?.makeCommandQueue()
        
        if metalDevice != nil {
            print("✓ Metal GPU initialized for parallel operations")
        }
        
        // Initialize tensor cache (Metal GPU + disk cache)
        let cacheDir = "\(sandersDir)/cache/tensors"
        self.tensorCache = TensorCache(cacheDir: cacheDir)
        
        print("✓ All resources loaded (GPU accelerated)")
    }
    
    func processAudio(audioPath: String) throws -> [MLMultiArray] {
        print("Processing audio: \(audioPath)")
        let startTime = Date()
        
        // Load WAV
        let samples = try SimpleWAVLoader.loadWAV(url: URL(fileURLWithPath: audioPath))
        print("  Loaded: \(samples.count) samples")
        
        // Process mel spectrograms
        let melSpec = melProcessor.process(samples)
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
            
            // Run audio encoder
            let input = try MLDictionaryFeatureProvider(dictionary: ["mel_spectrogram": melArray])
            let output = try audioEncoderModel.prediction(from: input)
            
            // Get output (name is var_242 from model)
            guard let features = output.featureValue(for: "var_242")?.multiArrayValue else {
                throw NSError(domain: "FrameGenerator", code: 1, 
                            userInfo: [NSLocalizedDescriptionKey: "Failed to get audio features"])
            }
            
            audioFeatures.append(features)
            
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
        
        // Timing accumulators
        var totalLoadTime = 0.0
        var totalTensorConvTime = 0.0
        var totalConcatTime = 0.0
        var totalReshapeTime = 0.0
        var totalInferenceTime = 0.0
        var totalToImageTime = 0.0
        var totalPasteTime = 0.0
        var totalSaveTime = 0.0
        
        for i in 1...numFrames {
            let frameStart = Date()
            
            if i % 10 == 1 || i == numFrames {
                print("Processing frame \(i)/\(numFrames)...")
            }
            
            // Load pre-cut images
            let t1 = Date()
            let roiPath = "\(sandersDir)/rois_320/\(i).jpg"
            let maskedPath = "\(sandersDir)/model_inputs/\(i).jpg"
            let fullBodyPath = "\(sandersDir)/full_body_img/\(i).jpg"
            
            guard let fullBodyImage = NSImage(contentsOfFile: fullBodyPath) else {
                throw NSError(domain: "Generator", code: 1)
            }
            totalLoadTime += Date().timeIntervalSince(t1)
            
            // Convert to MLMultiArrays (using cache!)
            let t2 = Date()
            let roiArray = try tensorCache.getTensor(imagePath: roiPath, normalize: true, cacheKey: "roi_\(i)")
            let maskedArray = try tensorCache.getTensor(imagePath: maskedPath, normalize: true, cacheKey: "masked_\(i)")
            totalTensorConvTime += Date().timeIntervalSince(t2)
            
            // Concatenate to 6-channel input [1, 6, 320, 320]
            let t3 = Date()
            let imageInput = try concatenateArrays(roiArray, maskedArray)
            totalConcatTime += Date().timeIntervalSince(t3)
            
            // Get audio features and reshape to [1, 32, 16, 16]
            let t4 = Date()
            let audioIdx = min(i - 1, audioFeatures.count - 1)
            let audioInput = try reshapeAudioFeatures(audioFeatures[audioIdx])
            totalReshapeTime += Date().timeIntervalSince(t4)
            
            // Run generator
            let t5 = Date()
            let genInput = try MLDictionaryFeatureProvider(dictionary: [
                "visual_input": MLFeatureValue(multiArray: imageInput),
                "audio_input": MLFeatureValue(multiArray: audioInput)
            ])
            let genOutput = try generatorModel.prediction(from: genInput)
            totalInferenceTime += Date().timeIntervalSince(t5)
            
            // Get output (name is var_1677 from model)
            guard let outputArray = genOutput.featureValue(for: "var_1677")?.multiArrayValue else {
                throw NSError(domain: "Generator", code: 2)
            }
            
            // Convert output to image
            let t6 = Date()
            let generatedImage = try mlMultiArrayToImage(outputArray, width: 320, height: 320)
            totalToImageTime += Date().timeIntervalSince(t6)
            
            // Paste into full frame
            let t7 = Date()
            let rectKey = String(i - 1)
            guard let cropRect = cropRectangles[rectKey] else {
                throw NSError(domain: "Generator", code: 3)
            }
            
            let finalFrame = try pasteIntoFrame(fullBodyImage, generated: generatedImage, rect: cropRect.rect)
            totalPasteTime += Date().timeIntervalSince(t7)
            
            // Save
            let t8 = Date()
            let outputPath = "\(outputDir)/frame_\(String(format: "%05d", i)).jpg"
            try saveJPEG(finalFrame, path: outputPath)
            totalSaveTime += Date().timeIntervalSince(t8)
            
            let frameTime = Date().timeIntervalSince(frameStart)
            if i <= 3 || i % 50 == 0 {
                print("    Frame \(i) took \(String(format: "%.3f", frameTime))s")
            }
        }  // End for loop
        
        let elapsed = Date().timeIntervalSince(startTime)
        print("✓ Frame generation: \(String(format: "%.2f", elapsed))s (\(String(format: "%.1f", Double(numFrames)/elapsed)) FPS)")
        
        // Print detailed timing breakdown
        print()
        print("Detailed timing breakdown (total for \(numFrames) frames):")
        print("  Image loading:        \(String(format: "%6.2f", totalLoadTime))s  (\(String(format: "%.1f", totalLoadTime/Double(numFrames)*1000))ms/frame)")
        print("  Tensor conversion:    \(String(format: "%6.2f", totalTensorConvTime))s  (\(String(format: "%.1f", totalTensorConvTime/Double(numFrames)*1000))ms/frame)")
        print("  Concatenation:        \(String(format: "%6.2f", totalConcatTime))s  (\(String(format: "%.1f", totalConcatTime/Double(numFrames)*1000))ms/frame)")
        print("  Audio reshape:        \(String(format: "%6.2f", totalReshapeTime))s  (\(String(format: "%.1f", totalReshapeTime/Double(numFrames)*1000))ms/frame)")
        print("  Core ML inference:    \(String(format: "%6.2f", totalInferenceTime))s  (\(String(format: "%.1f", totalInferenceTime/Double(numFrames)*1000))ms/frame)")
        print("  MLArray→Image:        \(String(format: "%6.2f", totalToImageTime))s  (\(String(format: "%.1f", totalToImageTime/Double(numFrames)*1000))ms/frame)")
        print("  Image pasting:        \(String(format: "%6.2f", totalPasteTime))s  (\(String(format: "%.1f", totalPasteTime/Double(numFrames)*1000))ms/frame)")
        print("  Image saving:         \(String(format: "%6.2f", totalSaveTime))s  (\(String(format: "%.1f", totalSaveTime/Double(numFrames)*1000))ms/frame)")
        print("  ─────────────────────────────────────────────────")
        let accountedTime = totalLoadTime + totalTensorConvTime + totalConcatTime + totalReshapeTime + totalInferenceTime + totalToImageTime + totalPasteTime + totalSaveTime
        print("  Accounted time:       \(String(format: "%6.2f", accountedTime))s")
        print("  Total time:           \(String(format: "%6.2f", elapsed))s")
        print("  Unaccounted/overhead: \(String(format: "%6.2f", elapsed - accountedTime))s")
        print()
        print("BOTTLENECK ANALYSIS:")
        let operations = [
            ("Image loading", totalLoadTime),
            ("Tensor conversion", totalTensorConvTime),
            ("Concatenation", totalConcatTime),
            ("Audio reshape", totalReshapeTime),
            ("Core ML inference", totalInferenceTime),
            ("MLArray→Image", totalToImageTime),
            ("Image pasting", totalPasteTime),
            ("Image saving", totalSaveTime)
        ]
        let sorted = operations.sorted { $0.1 > $1.1 }
        for (idx, (name, time)) in sorted.enumerated() {
            let percent = (time / accountedTime) * 100
            print("  \(idx + 1). \(name): \(String(format: "%.1f%%", percent)) of time")
        }
    }  // End generateFrames
    
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
        let result = try MLMultiArray(shape: [1, 6, 320, 320] as [NSNumber], dataType: .float32)
        
        let size = 3 * 320 * 320  // Size of one 3-channel image
        
        // Use Metal GPU if available for parallel copy
        if let device = metalDevice, let queue = metalQueue {
            // Use Metal compute for parallel memory operations
            return try concatenateWithMetal(arr1, arr2, result, device: device, queue: queue)
        }
        
        // Fallback: Direct pointer copy (still much faster than subscripts!)
        result.dataPointer.withMemoryRebound(to: Float.self, capacity: 6 * 320 * 320) { destPtr in
            arr1.dataPointer.withMemoryRebound(to: Float.self, capacity: size) { src1Ptr in
                // Copy first 3 channels
                destPtr.update(from: src1Ptr, count: size)
            }
            
            arr2.dataPointer.withMemoryRebound(to: Float.self, capacity: size) { src2Ptr in
                // Copy next 3 channels
                (destPtr + size).update(from: src2Ptr, count: size)
            }
        }
        
        return result
    }
    
    private func concatenateWithMetal(_ arr1: MLMultiArray, _ arr2: MLMultiArray, _ result: MLMultiArray, device: MTLDevice, queue: MTLCommandQueue) throws -> MLMultiArray {
        let size = 3 * 320 * 320
        
        // Create Metal buffers from MLMultiArray data
        guard let buffer1 = device.makeBuffer(bytes: arr1.dataPointer, length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let buffer2 = device.makeBuffer(bytes: arr2.dataPointer, length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let resultBuffer = device.makeBuffer(length: 6 * 320 * 320 * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "Metal", code: 1)
        }
        
        // Use compute command to copy in parallel on GPU
        guard let commandBuffer = queue.makeCommandBuffer(),
              let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            throw NSError(domain: "Metal", code: 2)
        }
        
        // Copy first 3 channels
        blitEncoder.copy(from: buffer1, sourceOffset: 0, 
                        to: resultBuffer, destinationOffset: 0, 
                        size: size * MemoryLayout<Float>.size)
        
        // Copy next 3 channels
        blitEncoder.copy(from: buffer2, sourceOffset: 0,
                        to: resultBuffer, destinationOffset: size * MemoryLayout<Float>.size,
                        size: size * MemoryLayout<Float>.size)
        
        blitEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Copy result back to MLMultiArray
        result.dataPointer.withMemoryRebound(to: Float.self, capacity: 6 * 320 * 320) { destPtr in
            let contents = resultBuffer.contents().assumingMemoryBound(to: Float.self)
            destPtr.update(from: contents, count: 6 * 320 * 320)
        }
        
        return result
    }
    
    private func reshapeAudioFeatures(_ features: MLMultiArray) throws -> MLMultiArray {
        // Input: [1, 512], Output: [1, 32, 16, 16]
        let result = try MLMultiArray(shape: [1, 32, 16, 16] as [NSNumber], dataType: .float32)
        
        // Tile the 512 values
        for i in 0..<(32*16*16) {
            let srcIdx = i % 512
            result[i] = features[srcIdx]
        }
        
        return result
    }
    
    private func mlMultiArrayToImage(_ array: MLMultiArray, width: Int, height: Int) throws -> NSImage {
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        
        // FAST: Use direct pointer access instead of subscripts!
        array.dataPointer.withMemoryRebound(to: Float.self, capacity: 3 * height * width) { floatPtr in
            for y in 0..<height {
                for x in 0..<width {
                    let pixelIndex = (y * width + x) * 4
                    let tensorIndex = y * width + x
                    
                    // BGR to RGB with direct pointer access
                    let b = UInt8(min(255, max(0, floatPtr[0 * height * width + tensorIndex] * 255)))
                    let g = UInt8(min(255, max(0, floatPtr[1 * height * width + tensorIndex] * 255)))
                    let r = UInt8(min(255, max(0, floatPtr[2 * height * width + tensorIndex] * 255)))
                    
                    pixelData[pixelIndex + 0] = r
                    pixelData[pixelIndex + 1] = g
                    pixelData[pixelIndex + 2] = b
                    pixelData[pixelIndex + 3] = 255
                }
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
        
        // Core Graphics uses bottom-left origin, but our coordinates are top-left
        // We need to flip the Y coordinate
        
        // Draw full frame normally
        context.draw(fullCG, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Draw generated with Y coordinate flipped
        // Convert top-left coordinates to bottom-left
        let y1_flipped = height - y2
        let targetRect = CGRect(x: x1, y: y1_flipped, width: x2 - x1, height: y2 - y1)
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
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.jpeg" as CFString, 1, nil) else {
            throw NSError(domain: "Generator", code: 10)
        }
        
        CGImageDestinationAddImage(destination, cgImage, [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary)
        
        if !CGImageDestinationFinalize(destination) {
            throw NSError(domain: "Generator", code: 11)
        }
    }
}

