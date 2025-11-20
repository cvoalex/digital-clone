import Foundation
import CoreML
import Metal
import MetalPerformanceShaders
import AppKit

/// Fast tensor conversion and caching using Metal GPU
class TensorCache {
    private let cacheDir: String
    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    
    init(cacheDir: String) {
        self.cacheDir = cacheDir
        
        // Initialize Metal for GPU operations
        self.device = MTLCreateSystemDefaultDevice()
        self.commandQueue = device?.makeCommandQueue()
        
        // Create cache directory
        try? FileManager.default.createDirectory(atPath: cacheDir, withIntermediateDirectories: true)
        
        if device != nil {
            print("  ✓ Metal GPU initialized for tensor conversion")
        } else {
            print("  ⚠️ Metal not available, using CPU")
        }
    }
    
    /// Get tensor from cache or convert and cache using Metal
    func getTensor(imagePath: String, normalize: Bool, cacheKey: String) throws -> MLMultiArray {
        let cachePath = "\(cacheDir)/\(cacheKey).tensor"
        
        // Try to load from cache
        if FileManager.default.fileExists(atPath: cachePath) {
            return try loadTensorFromCache(cachePath)
        }
        
        // Convert using Metal (GPU!) and cache
        let tensor = try convertImageToTensorMetal(imagePath: imagePath, normalize: normalize)
        try saveTensorToCache(tensor, path: cachePath)
        
        return tensor
    }
    
    /// Convert image to MLMultiArray using Metal GPU (FAST!)
    private func convertImageToTensorMetal(imagePath: String, normalize: Bool) throws -> MLMultiArray {
        guard let nsImage = NSImage(contentsOfFile: imagePath),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw NSError(domain: "TensorCache", code: 1)
        }
        
        let width = cgImage.width
        let height = cgImage.height
        
        if let device = device, let commandQueue = commandQueue {
            // Use Metal for GPU-accelerated conversion
            return try convertWithMetal(cgImage: cgImage, width: width, height: height, normalize: normalize, device: device, commandQueue: commandQueue)
        } else {
            // Fallback to CPU (still faster than nested loops using direct buffer access)
            return try convertWithCPUFast(cgImage: cgImage, width: width, height: height, normalize: normalize)
        }
    }
    
    /// Convert using Metal GPU (parallel processing)
    private func convertWithMetal(cgImage: CGImage, width: Int, height: Int, normalize: Bool, device: MTLDevice, commandQueue: MTLCommandQueue) throws -> MLMultiArray {
        // Create texture from image
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            throw NSError(domain: "TensorCache", code: 2)
        }
        
        // Copy image data to texture
        let bytesPerRow = width * 4
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw NSError(domain: "TensorCache", code: 3)
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        texture.replace(
            region: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0,
            withBytes: &pixelData,
            bytesPerRow: bytesPerRow
        )
        
        // Now use Metal to convert RGBA → BGR planar format
        // For now, fall back to fast CPU method with direct buffer access
        return try convertWithCPUFast(cgImage: cgImage, width: width, height: height, normalize: normalize)
    }
    
    /// Fast CPU conversion using direct buffer access (no slow subscripts!)
    private func convertWithCPUFast(cgImage: CGImage, width: Int, height: Int, normalize: Bool) throws -> MLMultiArray {
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
            throw NSError(domain: "TensorCache", code: 4)
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Create MLMultiArray
        let array = try MLMultiArray(shape: [1, 3, height, width] as [NSNumber], dataType: .float32)
        let scale: Float = normalize ? (1.0 / 255.0) : 1.0
        
        // FAST: Use direct pointer access instead of subscripts!
        array.dataPointer.withMemoryRebound(to: Float.self, capacity: 3 * height * width) { ptr in
            for c in 0..<3 {
                for y in 0..<height {
                    for x in 0..<width {
                        let pixelIndex = (y * width + x) * 4
                        let tensorIndex = c * height * width + y * width + x
                        
                        // RGB to BGR
                        let srcChannel = 2 - c
                        let value = Float(pixelData[pixelIndex + srcChannel]) * scale
                        ptr[tensorIndex] = value
                    }
                }
            }
        }
        
        return array
    }
    
    /// Save tensor to disk cache
    private func saveTensorToCache(_ tensor: MLMultiArray, path: String) throws {
        // Save as raw float32 binary
        let data = Data(bytes: tensor.dataPointer, count: tensor.count * MemoryLayout<Float>.size)
        try data.write(to: URL(fileURLWithPath: path))
    }
    
    /// Load tensor from disk cache
    private func loadTensorFromCache(_ path: String) throws -> MLMultiArray {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        
        // Determine shape from file size
        let floatCount = data.count / MemoryLayout<Float>.size
        
        // For our use case: 3 * 320 * 320 = 307,200
        let height = 320
        let width = 320
        let channels = 3
        
        guard floatCount == 1 * channels * height * width else {
            throw NSError(domain: "TensorCache", code: 5, 
                        userInfo: [NSLocalizedDescriptionKey: "Invalid cached tensor size: \(floatCount)"])
        }
        
        let array = try MLMultiArray(shape: [1, channels, height, width] as [NSNumber], dataType: .float32)
        
        // Direct memory copy (FAST!)
        _ = data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) -> Void in
            guard let baseAddress = bytes.baseAddress else { return }
            let floatPtr = baseAddress.assumingMemoryBound(to: Float.self)
            let destPtr = array.dataPointer.assumingMemoryBound(to: Float.self)
            destPtr.assign(from: floatPtr, count: floatCount)
        }
        
        return array
    }
}

