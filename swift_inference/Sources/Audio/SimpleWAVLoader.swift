//
//  SimpleWAVLoader.swift
//  mels
//
//  Simple WAV file loader that handles various formats
//

import Foundation

class SimpleWAVLoader {
    struct WAVHeader {
        var riff: UInt32 = 0
        var fileSize: UInt32 = 0
        var wave: UInt32 = 0
        var fmt: UInt32 = 0
        var fmtSize: UInt32 = 0
        var audioFormat: UInt16 = 0
        var numChannels: UInt16 = 0
        var sampleRate: UInt32 = 0
        var byteRate: UInt32 = 0
        var blockAlign: UInt16 = 0
        var bitsPerSample: UInt16 = 0
    }
    
    static func loadWAV(url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        
        guard data.count > 44 else {
            throw NSError(domain: "WAVLoader", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "File too small to be a valid WAV"])
        }
        
        // Parse header
        var offset = 0
        
        // RIFF header
        let riff = String(data: data[0..<4], encoding: .ascii)
        guard riff == "RIFF" else {
            throw NSError(domain: "WAVLoader", code: 2,
                         userInfo: [NSLocalizedDescriptionKey: "Not a valid RIFF file"])
        }
        offset += 4
        
        // File size
        offset += 4  // Skip file size
        
        // WAVE header
        let wave = String(data: data[8..<12], encoding: .ascii)
        guard wave == "WAVE" else {
            throw NSError(domain: "WAVLoader", code: 3,
                         userInfo: [NSLocalizedDescriptionKey: "Not a valid WAVE file"])
        }
        offset = 12
        
        // Find fmt chunk
        while offset < data.count - 8 {
            let chunkID = String(data: data[offset..<offset+4], encoding: .ascii)
            var chunkSize: UInt32 = 0
            _ = withUnsafeMutableBytes(of: &chunkSize) { ptr in
                data.copyBytes(to: ptr, from: offset+4..<offset+8)
            }
            offset += 8
            
            if chunkID == "fmt " {
                // Parse format with proper alignment
                var audioFormat: UInt16 = 0
                var numChannels: UInt16 = 0
                var sampleRate: UInt32 = 0
                var bitsPerSample: UInt16 = 0
                
                _ = withUnsafeMutableBytes(of: &audioFormat) { ptr in
                    data.copyBytes(to: ptr, from: offset..<offset+2)
                }
                _ = withUnsafeMutableBytes(of: &numChannels) { ptr in
                    data.copyBytes(to: ptr, from: offset+2..<offset+4)
                }
                _ = withUnsafeMutableBytes(of: &sampleRate) { ptr in
                    data.copyBytes(to: ptr, from: offset+4..<offset+8)
                }
                _ = withUnsafeMutableBytes(of: &bitsPerSample) { ptr in
                    data.copyBytes(to: ptr, from: offset+14..<offset+16)
                }
                
                print("WAV Format:")
                print("  Channels: \(numChannels)")
                print("  Sample Rate: \(sampleRate) Hz")
                print("  Bits: \(bitsPerSample)")
                
                offset += Int(chunkSize)
            } else if chunkID == "data" {
                // Audio data found
                let dataSize = Int(chunkSize)
                let dataStart = offset
                let dataEnd = min(offset + dataSize, data.count)
                
                // Parse 16-bit PCM samples
                let bytesPerSample = 2
                let totalBytes = dataEnd - dataStart
                let numSamples = totalBytes / bytesPerSample
                var samples = [Float](repeating: 0, count: numSamples)
                
                for i in 0..<numSamples {
                    let absoluteOffset = dataStart + (i * bytesPerSample)
                    if absoluteOffset + bytesPerSample <= data.count {
                        var sample: Int16 = 0
                        _ = withUnsafeMutableBytes(of: &sample) { ptr in
                            data.copyBytes(to: ptr, from: absoluteOffset..<absoluteOffset+bytesPerSample)
                        }
                        samples[i] = Float(sample) / 32768.0
                    }
                }
                
                print("Loaded \(samples.count) samples")
                
                return samples
            } else {
                // Skip unknown chunk
                offset += Int(chunkSize)
            }
        }
        
        throw NSError(domain: "WAVLoader", code: 4,
                     userInfo: [NSLocalizedDescriptionKey: "No audio data found in WAV file"])
    }
}

