//
//  ONNXWrapper.swift
//  mels
//
//  Native ONNX Runtime wrapper - NO PYTHON!
//

import Foundation

class ONNXRuntimeSession {
    private var env: OpaquePointer?
    private var session: OpaquePointer?
    private let api: UnsafePointer<OrtApi>
    
    init(modelPath: String) throws {
        // Get ONNX Runtime API
        guard let apiBase = OrtGetApiBase() else {
            throw NSError(domain: "ONNX", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to get ORT API"])
        }
        
        guard let apiPtr = apiBase.pointee.GetApi?(UInt32(ORT_API_VERSION)) else {
            throw NSError(domain: "ONNX", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to get ORT API pointer"])
        }
        
        self.api = apiPtr
        
        // Create environment
        var envPtr: OpaquePointer?
        let envStatus = api.pointee.CreateEnv?(ORT_LOGGING_LEVEL_WARNING, "mels", &envPtr)
        guard envStatus == nil, let env = envPtr else {
            throw NSError(domain: "ONNX", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create environment"])
        }
        self.env = env
        
        // Create session options
        var sessionOptionsPtr: OpaquePointer?
        let optStatus = api.pointee.CreateSessionOptions?(&sessionOptionsPtr)
        guard optStatus == nil, let sessionOptions = sessionOptionsPtr else {
            throw NSError(domain: "ONNX", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create session options"])
        }
        defer { api.pointee.ReleaseSessionOptions?(sessionOptions) }
        
        // Set number of threads
        _ = api.pointee.SetIntraOpNumThreads?(sessionOptions, 4)
        
        // Create session
        var sessionPtr: OpaquePointer?
        let sessionStatus = modelPath.withCString { pathCString in
            api.pointee.CreateSession?(env, pathCString, sessionOptions, &sessionPtr)
        }
        
        guard sessionStatus == nil, let session = sessionPtr else {
            throw NSError(domain: "ONNX", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed to create session"])
        }
        self.session = session
        
        print("✓ ONNX Runtime session created (NATIVE - NO PYTHON!)")
    }
    
    deinit {
        if let session = session {
            api.pointee.ReleaseSession?(session)
        }
        if let env = env {
            api.pointee.ReleaseEnv?(env)
        }
    }
    
    func run(input: [Float]) throws -> [Float] {
        guard let session = session else {
            throw NSError(domain: "ONNX", code: 6, userInfo: [NSLocalizedDescriptionKey: "Session not initialized"])
        }
        
        // Create memory info
        var memoryInfoPtr: OpaquePointer?
        let memStatus = api.pointee.CreateCpuMemoryInfo?(OrtAllocatorType(0), OrtMemType(0), &memoryInfoPtr)
        guard memStatus == nil, let memoryInfo = memoryInfoPtr else {
            throw NSError(domain: "ONNX", code: 7, userInfo: [NSLocalizedDescriptionKey: "Failed to create memory info"])
        }
        defer { api.pointee.ReleaseMemoryInfo?(memoryInfo) }
        
        // Input shape: [1, 1, 80, 16]
        let inputShape: [Int64] = [1, 1, 80, 16]
        let inputSize = 1 * 1 * 80 * 16
        
        // Create input tensor
        var inputTensorPtr: OpaquePointer?
        var inputData = input
        let tensorStatus = inputData.withUnsafeMutableBytes { rawPtr in
            api.pointee.CreateTensorWithDataAsOrtValue?(
                memoryInfo,
                rawPtr.baseAddress,
                inputSize * MemoryLayout<Float>.size,
                inputShape,
                4, // number of dimensions
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &inputTensorPtr
            )
        }
        
        guard tensorStatus == nil, let inputTensor = inputTensorPtr else {
            throw NSError(domain: "ONNX", code: 8, userInfo: [NSLocalizedDescriptionKey: "Failed to create input tensor"])
        }
        defer { api.pointee.ReleaseValue?(inputTensor) }
        
        // Run inference
        let inputNames = ["mel_input"]
        let outputNames = ["features"]
        
        var outputTensorPtr: OpaquePointer?
        
        let runStatus = inputNames[0].withCString { inputNameCStr -> OpaquePointer? in
            outputNames[0].withCString { outputNameCStr -> OpaquePointer? in
                var inputNamePtrs: [UnsafePointer<Int8>?] = [inputNameCStr]
                var outputNamePtrs: [UnsafePointer<Int8>?] = [outputNameCStr]
                var inputTensors: [OpaquePointer?] = [inputTensor]
                
                return inputNamePtrs.withUnsafeBufferPointer { inputNamesPtr in
                    outputNamePtrs.withUnsafeBufferPointer { outputNamesPtr in
                        inputTensors.withUnsafeBufferPointer { inputTensorsPtr in
                            api.pointee.Run?(
                                session,
                                nil, // run options
                                inputNamesPtr.baseAddress,
                                inputTensorsPtr.baseAddress,
                                1, // num inputs
                                outputNamesPtr.baseAddress,
                                1, // num outputs
                                &outputTensorPtr
                            )
                        }
                    }
                }
            }
        }
        
        guard runStatus == nil, let outputTensor = outputTensorPtr else {
            throw NSError(domain: "ONNX", code: 9, userInfo: [NSLocalizedDescriptionKey: "Inference failed"])
        }
        defer { api.pointee.ReleaseValue?(outputTensor) }
        
        // Get output data
        var outputDataPtr: UnsafeMutableRawPointer?
        let getDataStatus = api.pointee.GetTensorMutableData?(outputTensor, &outputDataPtr)
        guard getDataStatus == nil, let outputData = outputDataPtr else {
            throw NSError(domain: "ONNX", code: 10, userInfo: [NSLocalizedDescriptionKey: "Failed to get output data"])
        }
        
        // Copy output (512 float32 values)
        let outputSize = 512
        let outputBuffer = outputData.bindMemory(to: Float.self, capacity: outputSize)
        let result = Array(UnsafeBufferPointer(start: outputBuffer, count: outputSize))
        
        return result
    }
}

