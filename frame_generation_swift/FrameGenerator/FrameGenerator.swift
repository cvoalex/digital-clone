//
//  FrameGenerator.swift
//  FrameGenerator
//
//  Main frame generation logic
//

import Foundation
import UIKit

/// Frame generator that orchestrates image processing and model inference
class FrameGenerator {
    
    private let model: UNetModel
    private let imageProcessor: ImageProcessor
    private let mode: String
    
    init(modelPath: String, mode: String = "ave") throws {
        self.mode = mode
        self.imageProcessor = ImageProcessor()
        
        // Initialize model
        let config = UNetConfig(modelPath: modelPath, mode: mode)
        self.model = try UNetModel(config: config)
    }
    
    /// Generate a single frame from template image and audio features
    func generateFrame(
        templateImage: UIImage,
        landmarks: [Landmark],
        audioFeatures: [Float]
    ) throws -> UIImage {
        // Crop face region
        let (cropImg, coords) = try imageProcessor.cropFaceRegion(
            image: templateImage,
            landmarks: landmarks
        )
        
        // Get original crop size
        guard let cgImage = cropImg.cgImage else {
            throw FrameGeneratorError.imageProcessingFailed("Cannot get CGImage")
        }
        let originalWidth = cgImage.width
        let originalHeight = cgImage.height
        
        // Resize to 328x328
        let crop328 = try imageProcessor.resizeImage(
            image: cropImg,
            width: 328,
            height: 328
        )
        
        // Extract inner region [4:324, 4:324] -> 320x320
        let innerCrop = try extractInnerRegion(image: crop328)
        
        // Prepare input tensors
        let imageTensor = try imageProcessor.prepareInputTensors(image: innerCrop)
        
        // Run U-Net inference
        let outputTensor = try model.predict(
            imageTensor: imageTensor,
            audioFeatures: audioFeatures
        )
        
        // Convert output tensor to image
        let generatedRegion = try imageProcessor.tensorToImage(
            tensor: outputTensor,
            width: 320,
            height: 320
        )
        
        // Paste back into full frame
        let outputFrame = try imageProcessor.pasteGeneratedRegion(
            fullFrame: templateImage,
            generatedRegion: generatedRegion,
            coords: coords,
            originalCropWidth: originalWidth,
            originalCropHeight: originalHeight
        )
        
        return outputFrame
    }
    
    /// Generate frames from a template image sequence
    func generateFramesFromSequence(
        imgDir: String,
        lmsDir: String,
        audioFeatures: [[Float]],
        startFrame: Int = 0,
        progressCallback: ((Int, Int) -> Void)? = nil
    ) throws -> [UIImage] {
        let numFrames = audioFeatures.count
        
        // Get number of template images
        let imgFiles = try FileManager.default.contentsOfDirectory(atPath: imgDir)
            .filter { $0.hasSuffix(".jpg") }
            .sorted()
        
        let lenImg = imgFiles.count - 1  // Max index
        
        print("Generating \(numFrames) frames from \(imgFiles.count) template images")
        
        var frames: [UIImage] = []
        
        // Initialize ping-pong motion
        var stepStride = 0
        var imgIdx = 0
        
        for i in 0..<numFrames {
            // Ping-pong logic
            if imgIdx > lenImg - 1 {
                stepStride = -1
            }
            if imgIdx < 1 {
                stepStride = 1
            }
            imgIdx += stepStride
            
            // Load template image and landmarks
            let imgPath = "\(imgDir)/\(imgIdx + startFrame).jpg"
            let lmsPath = "\(lmsDir)/\(imgIdx + startFrame).lms"
            
            let templateImg = try imageProcessor.loadImage(path: imgPath)
            let landmarks = try imageProcessor.loadLandmarks(path: lmsPath)
            
            // Generate frame
            let frame = try generateFrame(
                templateImage: templateImg,
                landmarks: landmarks,
                audioFeatures: audioFeatures[i]
            )
            
            frames.append(frame)
            
            // Progress callback
            if (i + 1) % 10 == 0 || i == numFrames - 1 {
                progressCallback?(i + 1, numFrames)
            }
        }
        
        return frames
    }
    
    /// Save frames to disk
    func saveFrames(frames: [UIImage], outputDir: String, prefix: String = "frame") throws {
        // Create output directory
        try FileManager.default.createDirectory(
            atPath: outputDir,
            withIntermediateDirectories: true,
            attributes: nil
        )
        
        for (i, frame) in frames.enumerated() {
            let filename = String(format: "%@_%05d.jpg", prefix, i)
            let path = "\(outputDir)/\(filename)"
            
            guard let data = frame.jpegData(compressionQuality: 0.95) else {
                throw FrameGeneratorError.imageProcessingFailed("Cannot convert frame to JPEG")
            }
            
            try data.write(to: URL(fileURLWithPath: path))
        }
        
        print("Saved \(frames.count) frames to \(outputDir)")
    }
    
    // MARK: - Helper Methods
    
    /// Extract inner region [4:324, 4:324] from 328x328 image
    private func extractInnerRegion(image: UIImage) throws -> UIImage {
        guard let cgImage = image.cgImage else {
            throw FrameGeneratorError.imageProcessingFailed("Cannot get CGImage")
        }
        
        let rect = CGRect(x: 4, y: 4, width: 320, height: 320)
        
        guard let cropped = cgImage.cropping(to: rect) else {
            throw FrameGeneratorError.imageProcessingFailed("Cannot extract inner region")
        }
        
        return UIImage(cgImage: cropped)
    }
    
    /// Get audio features for a specific frame
    func getAudioFeaturesForFrame(
        allFeatures: [[Float]],
        frameIdx: Int
    ) -> [Float] {
        // Extract window around frame
        var left = frameIdx - 8
        var right = frameIdx + 8
        var padLeft = 0
        var padRight = 0
        
        if left < 0 {
            padLeft = -left
            left = 0
        }
        if right > allFeatures.count {
            padRight = right - allFeatures.count
            right = allFeatures.count
        }
        
        // Extract features
        var features: [[Float]] = []
        
        // Pad left
        for _ in 0..<padLeft {
            features.append([Float](repeating: 0, count: allFeatures[0].count))
        }
        
        // Copy actual features
        features.append(contentsOf: allFeatures[left..<right])
        
        // Pad right
        for _ in 0..<padRight {
            features.append([Float](repeating: 0, count: allFeatures[0].count))
        }
        
        // Reshape based on mode
        return reshapeAudioFeatures(features: features)
    }
    
    /// Reshape audio features based on mode
    private func reshapeAudioFeatures(features: [[Float]]) -> [Float] {
        // Flatten features
        return features.flatMap { $0 }
    }
}

// MARK: - Audio Features Loading

/// Load audio features from binary format
func loadAudioFeatures(path: String) throws -> [[Float]] {
    // Load metadata
    let metadataPath = path + ".json"
    let metadataData = try Data(contentsOf: URL(fileURLWithPath: metadataPath))
    
    struct Metadata: Codable {
        let numFrames: Int
        let featureSize: Int
        let shape: [Int]
        
        enum CodingKeys: String, CodingKey {
            case numFrames = "num_frames"
            case featureSize = "feature_size"
            case shape
        }
    }
    
    let metadata = try JSONDecoder().decode(Metadata.self, from: metadataData)
    
    // Load binary data
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    
    // Convert to float array
    let floatCount = data.count / MemoryLayout<Float>.size
    var floats = [Float](repeating: 0, count: floatCount)
    _ = floats.withUnsafeMutableBytes { data.copyBytes(to: $0) }
    
    // Reshape to [numFrames][featureSize]
    var features: [[Float]] = []
    for i in 0..<metadata.numFrames {
        let start = i * metadata.featureSize
        let end = start + metadata.featureSize
        features.append(Array(floats[start..<end]))
    }
    
    return features
}

