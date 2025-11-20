import Foundation
import CoreML

/// Memory buffer pools to eliminate allocations during frame generation
class BufferPool {
    // Pools for different buffer types
    private var pixelBuffers: [UnsafeMutablePointer<UInt8>] = []
    private var floatBuffers: [UnsafeMutablePointer<Float>] = []
    private var mlArrays: [MLMultiArray] = []
    
    private let pixelBufferSize: Int
    private let floatBufferSize: Int
    private let arrayShape: [NSNumber]
    
    private var pixelBufferIndex = 0
    private var floatBufferIndex = 0
    private var mlArrayIndex = 0
    
    init(pixelBufferSize: Int = 320 * 320 * 4, 
         floatBufferSize: Int = 320 * 320 * 3,
         arrayShape: [NSNumber] = [1, 3, 320, 320],
         poolSize: Int = 10) {
        self.pixelBufferSize = pixelBufferSize
        self.floatBufferSize = floatBufferSize
        self.arrayShape = arrayShape
        
        // Pre-allocate buffers
        for _ in 0..<poolSize {
            pixelBuffers.append(UnsafeMutablePointer<UInt8>.allocate(capacity: pixelBufferSize))
            floatBuffers.append(UnsafeMutablePointer<Float>.allocate(capacity: floatBufferSize))
            
            if let array = try? MLMultiArray(shape: arrayShape, dataType: .float32) {
                mlArrays.append(array)
            }
        }
    }
    
    deinit {
        for buffer in pixelBuffers {
            buffer.deallocate()
        }
        for buffer in floatBuffers {
            buffer.deallocate()
        }
    }
    
    func getPixelBuffer() -> UnsafeMutablePointer<UInt8> {
        let buffer = pixelBuffers[pixelBufferIndex % pixelBuffers.count]
        pixelBufferIndex += 1
        return buffer
    }
    
    func getFloatBuffer() -> UnsafeMutablePointer<Float> {
        let buffer = floatBuffers[floatBufferIndex % floatBuffers.count]
        floatBufferIndex += 1
        return buffer
    }
    
    func getMLArray() -> MLMultiArray? {
        guard !mlArrays.isEmpty else { return nil }
        let array = mlArrays[mlArrayIndex % mlArrays.count]
        mlArrayIndex += 1
        return array
    }
    
    func reset() {
        pixelBufferIndex = 0
        floatBufferIndex = 0
        mlArrayIndex = 0
    }
}

