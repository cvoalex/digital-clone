//
//  ImageProcessor.swift
//  FrameGenerator
//
//  Image processing module for frame generation pipeline
//  Handles cropping, masking, resizing, and pasting operations
//

import Foundation
import Accelerate
import CoreImage
import CoreGraphics
import UIKit

/// Represents a facial landmark point
struct Landmark {
    let x: Int
    let y: Int
}

/// Represents crop coordinates
struct CropCoords {
    let xMin: Int
    let yMin: Int
    let xMax: Int
    let yMax: Int
    
    var width: Int { xMax - xMin }
    var height: Int { yMax - yMin }
}

/// Image processor for frame generation
class ImageProcessor {
    
    // MARK: - Image Loading
    
    /// Load an image from disk
    func loadImage(path: String) throws -> UIImage {
        guard let image = UIImage(contentsOfFile: path) else {
            throw FrameGeneratorError.imageLoadFailed(path)
        }
        return image
    }
    
    /// Load landmarks from a .lms file
    func loadLandmarks(path: String) throws -> [Landmark] {
        guard let content = try? String(contentsOfFile: path, encoding: .utf8) else {
            throw FrameGeneratorError.landmarkLoadFailed(path)
        }
        
        var landmarks: [Landmark] = []
        let lines = content.components(separatedBy: .newlines)
        
        for line in lines {
            let parts = line.components(separatedBy: " ")
            if parts.count >= 2,
               let x = Int(parts[0]),
               let y = Int(parts[1]) {
                landmarks.append(Landmark(x: x, y: y))
            }
        }
        
        guard !landmarks.isEmpty else {
            throw FrameGeneratorError.invalidLandmarks("No landmarks found")
        }
        
        return landmarks
    }
    
    // MARK: - Cropping
    
    /// Calculate crop region based on landmarks
    func getCropRegion(landmarks: [Landmark]) throws -> CropCoords {
        guard landmarks.count >= 53 else {
            throw FrameGeneratorError.invalidLandmarks("Need at least 53 landmarks")
        }
        
        let xMin = landmarks[1].x
        let yMin = landmarks[52].y
        let xMax = landmarks[31].x
        let width = xMax - xMin
        let yMax = yMin + width
        
        return CropCoords(xMin: xMin, yMin: yMin, xMax: xMax, yMax: yMax)
    }
    
    /// Crop face region from image
    func cropFaceRegion(image: UIImage, landmarks: [Landmark]) throws -> (UIImage, CropCoords) {
        let coords = try getCropRegion(landmarks: landmarks)
        
        guard let cgImage = image.cgImage else {
            throw FrameGeneratorError.imageProcessingFailed("Cannot get CGImage")
        }
        
        let rect = CGRect(x: coords.xMin, y: coords.yMin,
                         width: coords.width, height: coords.height)
        
        guard let croppedCGImage = cgImage.cropping(to: rect) else {
            throw FrameGeneratorError.imageProcessingFailed("Crop failed")
        }
        
        let croppedImage = UIImage(cgImage: croppedCGImage)
        return (croppedImage, coords)
    }
    
    // MARK: - Resizing
    
    /// Resize image using cubic interpolation
    func resizeImage(image: UIImage, width: Int, height: Int) throws -> UIImage {
        let targetSize = CGSize(width: width, height: height)
        
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        
        let resized = renderer.image { context in
            // Use high quality interpolation
            context.cgContext.interpolationQuality = .high
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
        
        return resized
    }
    
    // MARK: - Masking
    
    /// Create masked version with lower face blacked out
    func createMaskedRegion(image: UIImage) throws -> UIImage {
        guard let cgImage = image.cgImage else {
            throw FrameGeneratorError.imageProcessingFailed("Cannot get CGImage")
        }
        
        let width = cgImage.width
        let height = cgImage.height
        
        UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), false, 1.0)
        guard let context = UIGraphicsGetCurrentContext() else {
            UIGraphicsEndImageContext()
            throw FrameGeneratorError.imageProcessingFailed("Cannot get graphics context")
        }
        
        // Draw original image
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Draw black rectangle
        context.setFillColor(UIColor.black.cgColor)
        context.fill(CGRect(x: 5, y: 5, width: 305, height: 305))
        
        let maskedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        guard let result = maskedImage else {
            throw FrameGeneratorError.imageProcessingFailed("Failed to create masked image")
        }
        
        return result
    }
    
    // MARK: - Tensor Preparation
    
    /// Prepare input tensors for U-Net model
    /// Returns 6-channel tensor: [original, masked] concatenated
    func prepareInputTensors(image: UIImage) throws -> [Float] {
        // Create masked version
        let masked = try createMaskedRegion(image: image)
        
        // Convert both to tensors
        let originalTensor = try imageToTensor(image: image, normalize: true)
        let maskedTensor = try imageToTensor(image: masked, normalize: true)
        
        // Concatenate along channel dimension
        return originalTensor + maskedTensor
    }
    
    /// Convert UIImage to tensor in CHW format (RGB)
    private func imageToTensor(image: UIImage, normalize: Bool) throws -> [Float] {
        guard let cgImage = image.cgImage else {
            throw FrameGeneratorError.imageProcessingFailed("Cannot get CGImage")
        }
        
        let width = cgImage.width
        let height = cgImage.height
        
        // Create pixel buffer
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
            throw FrameGeneratorError.imageProcessingFailed("Cannot create context")
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Convert to CHW format (RGB)
        var tensor = [Float](repeating: 0, count: 3 * width * height)
        let scale: Float = normalize ? (1.0 / 255.0) : 1.0
        
        for c in 0..<3 {
            for y in 0..<height {
                for x in 0..<width {
                    let pixelIndex = (y * width + x) * 4
                    let tensorIndex = c * height * width + y * width + x
                    
                    // RGBA to RGB conversion
                    let value = Float(pixelData[pixelIndex + c]) * scale
                    tensor[tensorIndex] = value
                }
            }
        }
        
        return tensor
    }
    
    // MARK: - Pasting
    
    /// Paste generated region back into full frame
    func pasteGeneratedRegion(
        fullFrame: UIImage,
        generatedRegion: UIImage,
        coords: CropCoords,
        originalCropWidth: Int,
        originalCropHeight: Int
    ) throws -> UIImage {
        // Create 328x328 canvas
        let canvas = try createCanvas(size: CGSize(width: 328, height: 328))
        
        // Paste generated region in center [4:324, 4:324]
        let canvasWithRegion = try pasteImage(
            background: canvas,
            foreground: generatedRegion,
            at: CGPoint(x: 4, y: 4)
        )
        
        // Resize back to original crop size
        let resized = try resizeImage(
            image: canvasWithRegion,
            width: originalCropWidth,
            height: originalCropHeight
        )
        
        // Paste back into full frame
        let result = try pasteImage(
            background: fullFrame,
            foreground: resized,
            at: CGPoint(x: coords.xMin, y: coords.yMin)
        )
        
        return result
    }
    
    /// Create blank canvas
    private func createCanvas(size: CGSize) throws -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        
        let canvas = renderer.image { context in
            UIColor.black.setFill()
            context.fill(CGRect(origin: .zero, size: size))
        }
        
        return canvas
    }
    
    /// Paste one image onto another
    private func pasteImage(background: UIImage, foreground: UIImage, at point: CGPoint) throws -> UIImage {
        guard let bgCGImage = background.cgImage else {
            throw FrameGeneratorError.imageProcessingFailed("Cannot get background CGImage")
        }
        
        let size = CGSize(width: bgCGImage.width, height: bgCGImage.height)
        
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        
        let result = renderer.image { context in
            background.draw(at: .zero)
            foreground.draw(at: point)
        }
        
        return result
    }
    
    // MARK: - Tensor to Image
    
    /// Convert tensor (CHW format, RGB) to UIImage (BGR)
    func tensorToImage(tensor: [Float], width: Int, height: Int) throws -> UIImage {
        // Convert from CHW RGB to HWC BGR for UIImage
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = (y * width + x) * 4
                
                // RGB to BGR conversion
                let r = UInt8(max(0, min(255, tensor[0 * height * width + y * width + x])))
                let g = UInt8(max(0, min(255, tensor[1 * height * width + y * width + x])))
                let b = UInt8(max(0, min(255, tensor[2 * height * width + y * width + x])))
                
                pixelData[pixelIndex + 0] = r
                pixelData[pixelIndex + 1] = g
                pixelData[pixelIndex + 2] = b
                pixelData[pixelIndex + 3] = 255 // Alpha
            }
        }
        
        // Create CGImage from pixel data
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
            throw FrameGeneratorError.imageProcessingFailed("Cannot create CGImage from tensor")
        }
        
        return UIImage(cgImage: cgImage)
    }
}

// MARK: - Errors

enum FrameGeneratorError: Error {
    case imageLoadFailed(String)
    case landmarkLoadFailed(String)
    case invalidLandmarks(String)
    case imageProcessingFailed(String)
    case modelError(String)
    case tensorConversionFailed(String)
}

