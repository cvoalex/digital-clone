import Foundation
import CoreML
import UIKit
import Metal
import Accelerate

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
    
    func processAudio(audioPath: String, maxFrames: Int? = nil) throws -> [MLMultiArray] {
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
            // Calculate how many samples we need for maxFrames
            // 25 fps, 80 mel windows per frame window, 200 hop size
            let samplesNeeded = Int(Double(maxFrames) * 1.1 * 80.0 * 200.0)
            if samplesNeeded < samples.count {
                samples = Array(samples.prefix(samplesNeeded))
                print("  Trimmed to \(samples.count) samples (for \(maxFrames) frames)")
            }
        }
        
        // Process mel spectrogram (now only what we need!)
        let melSpec = melProcessor.process(samples)
        print("  Generated mel spectrogram: \(melSpec.count) x \(melSpec[0].count)")
        
        // Get frame count
        let fps = 25.0
        let numFrames = melProcessor.getFrameCount(melSpec: melSpec, fps: fps)
        print("  Will encode: \(numFrames) frames")
        
        // Encode each frame
        var audioFeatures: [MLMultiArray] = []
        
        for i in 0..<numFrames {
            let melWindow = try melProcessor.cropAudioWindow(melSpec: melSpec, frameIdx: i, fps: fps)
            let melFlat = flattenMelWindow(melWindow)
            let melArray = try createMLMultiArray(shape: [1, 1, 80, 16], data: melFlat)
            
            let input = try MLDictionaryFeatureProvider(dictionary: ["mel_spectrogram": melArray])
            let output = try audioEncoderModel.prediction(from: input)
            
            guard let features = output.featureValue(for: "var_242")?.multiArrayValue else {
                throw NSError(domain: "FrameGenerator", code: 2)
            }
            
            audioFeatures.append(features)
            
            if (i + 1) % 100 == 0 {
                print("  Encoded \(i + 1) frames")
            }
        }
        
        let elapsed = Date().timeIntervalSince(startTime)
        print("✓ Audio processing complete: \(String(format: "%.2f", elapsed))s (cached for next run)")
        
        // Cache the results
        Self.cachedAudioFeatures = audioFeatures
        Self.cachedAudioPath = audioPath
        
        return audioFeatures
    }
    
    func generateFrame(
        roiImage: UIImage,
        maskedImage: UIImage,
        audioFeatures: MLMultiArray
    ) throws -> UIImage {
        // Convert images to MLMultiArrays
        let roiArray = try imageToMLMultiArray(roiImage, normalize: true)
        let maskedArray = try imageToMLMultiArray(maskedImage, normalize: true)
        
        // Concatenate to 6-channel input
        let imageInput = try concatenateArraysMetal(roiArray, maskedArray)
        
        // Reshape audio features
        let audioInput = try reshapeAudioFeatures(audioFeatures)
        
        // Run generator
        let genInput = try MLDictionaryFeatureProvider(dictionary: [
            "visual_input": MLFeatureValue(multiArray: imageInput),
            "audio_input": MLFeatureValue(multiArray: audioInput)
        ])
        
        let output = try generatorModel.prediction(from: genInput)
        
        guard let outputArray = output.featureValue(for: "var_1677")?.multiArrayValue else {
            throw NSError(domain: "Generator", code: 2)
        }
        
        // Convert to image
        let generatedImage = try mlMultiArrayToImage(outputArray, width: 320, height: 320)
        
        return generatedImage
    }
    
    // MARK: - iOS-specific helper methods (UIImage instead of NSImage)
    
    private func imageToMLMultiArray(_ image: UIImage, normalize: Bool) throws -> MLMultiArray {
        guard let cgImage = image.cgImage else {
            throw NSError(domain: "Generator", code: 3)
        }
        
        let width = cgImage.width
        let height = cgImage.height
        
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
        
        // Fast: Direct pointer access
        array.dataPointer.withMemoryRebound(to: Float.self, capacity: 3 * height * width) { ptr in
            for c in 0..<3 {
                for y in 0..<height {
                    for x in 0..<width {
                        let pixelIndex = (y * width + x) * 4
                        let tensorIndex = c * height * width + y * width + x
                        let srcChannel = 2 - c  // RGB to BGR
                        let value = Float(pixelData[pixelIndex + srcChannel]) * scale
                        ptr[tensorIndex] = value
                    }
                }
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
        
        array.dataPointer.withMemoryRebound(to: Float.self, capacity: 3 * height * width) { floatPtr in
            for y in 0..<height {
                for x in 0..<width {
                    let pixelIndex = (y * width + x) * 4
                    let tensorIndex = y * width + x
                    
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

