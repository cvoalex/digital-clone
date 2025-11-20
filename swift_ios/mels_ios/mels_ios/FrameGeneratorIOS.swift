import Foundation
import CoreML
import UIKit
import Metal
import Accelerate
import Accelerate.vImage

@available(iOS 16.0, *)
class FrameGeneratorIOS {
    private let audioEncoderModel: MLModel
    private let generatorModel: MLModel
    private let metalDevice: MTLDevice?
    private let metalQueue: MTLCommandQueue?
    private let melProcessor: MelProcessor
    
    // Cache audio features to avoid reprocessing
    private static var cachedAudioFeatures: [MLMultiArray]?
    private static var cachedAudioPath: String?
    
    // Cache converted image tensors (HUGE speedup!)
    private static var imageTensorCache: [String: MLMultiArray] = [:]
    
    init() throws {
        print("Loading Core ML models for iOS...")
        
        // Load models from bundle
        guard let audioEncoderURL = Bundle.main.url(forResource: "AudioEncoder", withExtension: "mlmodelc"),
              let generatorURL = Bundle.main.url(forResource: "Generator", withExtension: "mlmodelc") else {
            throw NSError(domain: "FrameGenerator", code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "Models not found in app bundle"])
        }
        
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Use CPU + GPU + Neural Engine on iOS!
        
        self.audioEncoderModel = try MLModel(contentsOf: audioEncoderURL, configuration: config)
        self.generatorModel = try MLModel(contentsOf: generatorURL, configuration: config)
        
        // Initialize Metal for GPU operations
        self.metalDevice = MTLCreateSystemDefaultDevice()
        self.metalQueue = metalDevice?.makeCommandQueue()
        
        // Initialize mel processor
        self.melProcessor = MelProcessor()
        
        print("✓ Core ML models loaded (Neural Engine + GPU enabled)")
    }
    
    func processAudio(audioPath: String, maxFrames: Int? = nil) async throws -> [MLMultiArray] {
        // Check cache first!
        if let cached = Self.cachedAudioFeatures, Self.cachedAudioPath == audioPath {
            print("✓ Using cached audio features (\(cached.count) frames)")
            if let max = maxFrames {
                return Array(cached.prefix(max))
            }
            return cached
        }
        
        print("Processing audio from WAV file (first time - will cache)...")
        let startTime = Date()
        
        // Load WAV
        var samples = try SimpleWAVLoader.loadWAV(url: URL(fileURLWithPath: audioPath))
        print("  Loaded \(samples.count) audio samples")
        
        // If maxFrames specified, trim audio to needed length (+10% buffer)
        if let maxFrames = maxFrames {
            // Calculate: maxFrames * 25fps = seconds needed
            // seconds * 16000 sample rate = samples
            // Add 20% buffer for windowing
            let secondsNeeded = Double(maxFrames) / 25.0 * 1.2
            let samplesNeeded = Int(secondsNeeded * 16000.0)
            
            if samplesNeeded < samples.count {
                samples = Array(samples.prefix(samplesNeeded))
                print("  ⚡ Trimmed audio: \(samples.count) samples (for \(maxFrames) frames, was \(samples.count + (samples.count - samplesNeeded)))")
            }
        }
        
        // Process mel spectrogram (now only what we need!)
        let melSpec = melProcessor.process(samples)
        print("  Generated mel spectrogram: \(melSpec.count) x \(melSpec[0].count)")
        
        // Get frame count
        let fps = 25.0
        let numFrames = melProcessor.getFrameCount(melSpec: melSpec, fps: fps)
        print("  Will encode: \(numFrames) frames")
        
        // Encode frames in parallel batches
        var audioFeatures: [MLMultiArray] = Array(repeating: try MLMultiArray(shape: [1, 512] as [NSNumber], dataType: .float32), count: numFrames)
        
        print("  Encoding in parallel...")
        let batchSize = 50
        
        await withTaskGroup(of: (Int, MLMultiArray).self) { group in
            for batchStart in stride(from: 0, to: numFrames, by: batchSize) {
                let batchEnd = min(batchStart + batchSize, numFrames)
                
                group.addTask {
                    var batchResults: [(Int, MLMultiArray)] = []
                    
                    for i in batchStart..<batchEnd {
                        do {
                            let melWindow = try self.melProcessor.cropAudioWindow(melSpec: melSpec, frameIdx: i, fps: fps)
                            let melFlat = self.flattenMelWindow(melWindow)
                            let melArray = try self.createMLMultiArray(shape: [1, 1, 80, 16], data: melFlat)
                            
                            let input = try MLDictionaryFeatureProvider(dictionary: ["mel_spectrogram": melArray])
                            let output = try self.audioEncoderModel.prediction(from: input)
                            
                            if let features = output.featureValue(for: "var_242")?.multiArrayValue {
                                batchResults.append((i, features))
                            }
                        } catch {
                            print("Error encoding frame \(i): \(error)")
                        }
                    }
                    
                    return batchResults.isEmpty ? (batchStart, try! MLMultiArray(shape: [1, 512] as [NSNumber], dataType: .float32)) : batchResults[0]
                }
                
                if batchStart % 200 == 0 {
                    print("  Dispatched batch \(batchStart)...")
                }
            }
            
            // Collect results
            for await (index, features) in group {
                if index < numFrames {
                    audioFeatures[index] = features
                }
            }
        }
        
        print("  ✓ Parallel encoding complete")
        
        let elapsed = Date().timeIntervalSince(startTime)
        print("✓ Audio processing complete: \(String(format: "%.2f", elapsed))s (cached for next run)")
        
        // Cache the results
        Self.cachedAudioFeatures = audioFeatures
        Self.cachedAudioPath = audioPath
        
        return audioFeatures
    }
    
    private static var timingStats = (
        imageToArray: 0.0,
        concat: 0.0,
        reshape: 0.0,
        inference: 0.0,
        arrayToImage: 0.0,
        frameCount: 0
    )
    
    func generateFrame(
        roiImage: UIImage,
        maskedImage: UIImage,
        audioFeatures: MLMultiArray
    ) throws -> UIImage {
        let frameStart = Date()
        
        // Convert images to MLMultiArrays (with caching!)
        let t1 = Date()
        let roiArray = try imageToMLMultiArrayCached(roiImage, cacheKey: "roi_\(roiImage.hash)", normalize: true)
        let maskedArray = try imageToMLMultiArrayCached(maskedImage, cacheKey: "masked_\(maskedImage.hash)", normalize: true)
        Self.timingStats.imageToArray += Date().timeIntervalSince(t1)
        
        // Concatenate to 6-channel input
        let t2 = Date()
        let imageInput = try concatenateArraysMetal(roiArray, maskedArray)
        Self.timingStats.concat += Date().timeIntervalSince(t2)
        
        // Reshape audio features
        let t3 = Date()
        let audioInput = try reshapeAudioFeatures(audioFeatures)
        Self.timingStats.reshape += Date().timeIntervalSince(t3)
        
        // Run generator
        let t4 = Date()
        let genInput = try MLDictionaryFeatureProvider(dictionary: [
            "visual_input": MLFeatureValue(multiArray: imageInput),
            "audio_input": MLFeatureValue(multiArray: audioInput)
        ])
        
        let output = try generatorModel.prediction(from: genInput)
        Self.timingStats.inference += Date().timeIntervalSince(t4)
        
        guard let outputArray = output.featureValue(for: "var_1677")?.multiArrayValue else {
            throw NSError(domain: "Generator", code: 2)
        }
        
        // Convert to image
        let t5 = Date()
        let generatedImage = try mlMultiArrayToImage(outputArray, width: 320, height: 320)
        Self.timingStats.arrayToImage += Date().timeIntervalSince(t5)
        
        Self.timingStats.frameCount += 1
        
        // Log every 50 frames
        if Self.timingStats.frameCount % 50 == 0 {
            let total = Self.timingStats.imageToArray + Self.timingStats.concat + Self.timingStats.reshape + Self.timingStats.inference + Self.timingStats.arrayToImage
            print("Timing (\(Self.timingStats.frameCount) frames):")
            print("  Image→Array: \(String(format: "%.2f", Self.timingStats.imageToArray))s (\(String(format: "%.1f", Self.timingStats.imageToArray/total*100))%)")
            print("  Concatenate: \(String(format: "%.2f", Self.timingStats.concat))s (\(String(format: "%.1f", Self.timingStats.concat/total*100))%)")
            print("  Reshape: \(String(format: "%.2f", Self.timingStats.reshape))s (\(String(format: "%.1f", Self.timingStats.reshape/total*100))%)")
            print("  Inference: \(String(format: "%.2f", Self.timingStats.inference))s (\(String(format: "%.1f", Self.timingStats.inference/total*100))%)")
            print("  Array→Image: \(String(format: "%.2f", Self.timingStats.arrayToImage))s (\(String(format: "%.1f", Self.timingStats.arrayToImage/total*100))%)")
        }
        
        return generatedImage
    }
    
    // MARK: - iOS-specific helper methods (UIImage instead of NSImage)
    
    private func imageToMLMultiArrayCached(_ image: UIImage, cacheKey: String, normalize: Bool) throws -> MLMultiArray {
        // Check cache first!
        if let cached = Self.imageTensorCache[cacheKey] {
            return cached
        }
        
        // Convert and cache
        let tensor = try imageToMLMultiArray(image, normalize: normalize)
        Self.imageTensorCache[cacheKey] = tensor
        return tensor
    }
    
    private func imageToMLMultiArray(_ image: UIImage, normalize: Bool) throws -> MLMultiArray {
        guard let cgImage = image.cgImage else {
            throw NSError(domain: "Generator", code: 3)
        }
        
        let width = cgImage.width
        let height = cgImage.height
        
        // Use Metal/vImage for FAST conversion if available
        if let device = metalDevice {
            return try imageToMLMultiArrayMetal(cgImage: cgImage, width: width, height: height, normalize: normalize, device: device)
        }
        
        // Fallback: Direct buffer operations (still faster than subscripts!)
        return try imageToMLMultiArrayDirect(cgImage: cgImage, width: width, height: height, normalize: normalize)
    }
    
    private func imageToMLMultiArrayDirect(cgImage: CGImage, width: Int, height: Int, normalize: Bool) throws -> MLMultiArray {
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
        
        let array = try MLMultiArray(shape: [1, 3, height, width] as [NSNumber], dataType: .float32)
        let scale: Float = normalize ? (1.0 / 255.0) : 1.0
        
        // FASTEST: Use vImage for vectorized SIMD conversion!
        array.dataPointer.withMemoryRebound(to: Float.self, capacity: 3 * height * width) { destPtr in
            pixelData.withUnsafeBytes { srcBytes in
                let srcPtr = srcBytes.baseAddress!.assumingMemoryBound(to: UInt8.self)
                
                // Use vImage to convert interleaved RGBA → planar BGR
                var srcBuffer = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcPtr),
                    height: vImagePixelCount(height),
                    width: vImagePixelCount(width),
                    rowBytes: width * 4
                )
                
                // Separate R, G, B channels
                var rDest = [Float](repeating: 0, count: width * height)
                var gDest = [Float](repeating: 0, count: width * height)
                var bDest = [Float](repeating: 0, count: width * height)
                
                // Convert UInt8 to Float with scaling (SIMD vectorized!)
                vDSP_vfltu8(srcPtr.advanced(by: 0), 4, &rDest, 1, vDSP_Length(width * height))
                vDSP_vfltu8(srcPtr.advanced(by: 1), 4, &gDest, 1, vDSP_Length(width * height))
                vDSP_vfltu8(srcPtr.advanced(by: 2), 4, &bDest, 1, vDSP_Length(width * height))
                
                if normalize {
                    // Vectorized division by 255
                    var divisor: Float = 255.0
                    vDSP_vsdiv(rDest, 1, &divisor, &rDest, 1, vDSP_Length(width * height))
                    vDSP_vsdiv(gDest, 1, &divisor, &gDest, 1, vDSP_Length(width * height))
                    vDSP_vsdiv(bDest, 1, &divisor, &bDest, 1, vDSP_Length(width * height))
                }
                
                // Copy to MLMultiArray in BGR order
                bDest.withUnsafeBytes { bBytes in
                    let bPtr = bBytes.baseAddress!.assumingMemoryBound(to: Float.self)
                    destPtr.advanced(by: 0).update(from: bPtr, count: width * height)
                }
                gDest.withUnsafeBytes { gBytes in
                    let gPtr = gBytes.baseAddress!.assumingMemoryBound(to: Float.self)
                    destPtr.advanced(by: width * height).update(from: gPtr, count: width * height)
                }
                rDest.withUnsafeBytes { rBytes in
                    let rPtr = rBytes.baseAddress!.assumingMemoryBound(to: Float.self)
                    destPtr.advanced(by: 2 * width * height).update(from: rPtr, count: width * height)
                }
            }
        }
        
        return array
    }
    
    private func imageToMLMultiArrayMetal(cgImage: CGImage, width: Int, height: Int, normalize: Bool, device: MTLDevice) throws -> MLMultiArray {
        // Use Metal for GPU-accelerated conversion
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
        
        // Create Metal buffer for parallel processing
        guard let pixelBuffer = device.makeBuffer(bytes: pixelData, length: pixelData.count, options: .storageModeShared) else {
            return try imageToMLMultiArrayDirect(cgImage: cgImage, width: width, height: height, normalize: normalize)
        }
        
        let array = try MLMultiArray(shape: [1, 3, height, width] as [NSNumber], dataType: .float32)
        let scale: Float = normalize ? (1.0 / 255.0) : 1.0
        
        // Fast conversion with direct pointer access
        array.dataPointer.withMemoryRebound(to: Float.self, capacity: 3 * height * width) { destPtr in
            let srcPtr = pixelBuffer.contents().assumingMemoryBound(to: UInt8.self)
            
            // Single-pass conversion
            for i in 0..<(width * height) {
                let pixelIdx = i * 4
                let r = Float(srcPtr[pixelIdx + 0]) * scale
                let g = Float(srcPtr[pixelIdx + 1]) * scale
                let b = Float(srcPtr[pixelIdx + 2]) * scale
                
                destPtr[0 * width * height + i] = b
                destPtr[1 * width * height + i] = g
                destPtr[2 * width * height + i] = r
            }
        }
        
        return array
    }
    
    private func concatenateArraysMetal(_ arr1: MLMultiArray, _ arr2: MLMultiArray) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: [1, 6, 320, 320] as [NSNumber], dataType: .float32)
        let size = 3 * 320 * 320
        
        // Use Metal if available
        if let device = metalDevice, let queue = metalQueue {
            return try concatenateWithMetal(arr1, arr2, result, device: device, queue: queue)
        }
        
        // Fallback: Direct pointer copy
        result.dataPointer.withMemoryRebound(to: Float.self, capacity: 6 * 320 * 320) { destPtr in
            arr1.dataPointer.withMemoryRebound(to: Float.self, capacity: size) { src1Ptr in
                destPtr.update(from: src1Ptr, count: size)
            }
            arr2.dataPointer.withMemoryRebound(to: Float.self, capacity: size) { src2Ptr in
                (destPtr + size).update(from: src2Ptr, count: size)
            }
        }
        
        return result
    }
    
    private func concatenateWithMetal(_ arr1: MLMultiArray, _ arr2: MLMultiArray, _ result: MLMultiArray, device: MTLDevice, queue: MTLCommandQueue) throws -> MLMultiArray {
        let size = 3 * 320 * 320
        
        guard let buffer1 = device.makeBuffer(bytes: arr1.dataPointer, length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let buffer2 = device.makeBuffer(bytes: arr2.dataPointer, length: size * MemoryLayout<Float>.size, options: .storageModeShared),
              let resultBuffer = device.makeBuffer(length: 6 * 320 * 320 * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "Metal", code: 1)
        }
        
        guard let commandBuffer = queue.makeCommandBuffer(),
              let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            throw NSError(domain: "Metal", code: 2)
        }
        
        blitEncoder.copy(from: buffer1, sourceOffset: 0, to: resultBuffer, destinationOffset: 0, size: size * MemoryLayout<Float>.size)
        blitEncoder.copy(from: buffer2, sourceOffset: 0, to: resultBuffer, destinationOffset: size * MemoryLayout<Float>.size, size: size * MemoryLayout<Float>.size)
        blitEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        result.dataPointer.withMemoryRebound(to: Float.self, capacity: 6 * 320 * 320) { destPtr in
            let contents = resultBuffer.contents().assumingMemoryBound(to: Float.self)
            destPtr.update(from: contents, count: 6 * 320 * 320)
        }
        
        return result
    }
    
    private func reshapeAudioFeatures(_ features: MLMultiArray) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: [1, 32, 16, 16] as [NSNumber], dataType: .float32)
        
        for i in 0..<(32*16*16) {
            let srcIdx = i % 512
            result[i] = features[srcIdx]
        }
        
        return result
    }
    
    private func mlMultiArrayToImage(_ array: MLMultiArray, width: Int, height: Int) throws -> UIImage {
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        
        // FAST: Single-pass conversion with direct pointer access
        array.dataPointer.withMemoryRebound(to: Float.self, capacity: 3 * height * width) { floatPtr in
            // Single loop instead of nested loops!
            for i in 0..<(width * height) {
                let b = UInt8(min(255, max(0, floatPtr[0 * width * height + i] * 255)))
                let g = UInt8(min(255, max(0, floatPtr[1 * width * height + i] * 255)))
                let r = UInt8(min(255, max(0, floatPtr[2 * width * height + i] * 255)))
                
                let pixelIdx = i * 4
                pixelData[pixelIdx + 0] = r
                pixelData[pixelIdx + 1] = g
                pixelData[pixelIdx + 2] = b
                pixelData[pixelIdx + 3] = 255
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
        
        return UIImage(cgImage: cgImage)
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
    
    private func createMLMultiArray(shape: [Int], data: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape as [NSNumber], dataType: .float32)
        for (index, value) in data.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }
}

