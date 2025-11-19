package onnx

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
)

// AudioEncoderBridge uses Python ONNX Runtime via subprocess
type AudioEncoderBridge struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
}

// NewAudioEncoderBridge creates a new encoder using Python bridge
func NewAudioEncoderBridge(modelPath string) (*AudioEncoderBridge, error) {
	// Start Python ONNX server
	cmd := exec.Command("python3", "onnx_server.py", modelPath)
	
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to get stdin: %w", err)
	}
	
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to get stdout: %w", err)
	}
	
	// Start the process
	err = cmd.Start()
	if err != nil {
		return nil, fmt.Errorf("failed to start ONNX server: %w", err)
	}
	
	reader := bufio.NewReader(stdout)
	
	// Wait for READY signal
	line, err := reader.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read ready signal: %w", err)
	}
	
	if line != "READY\n" {
		return nil, fmt.Errorf("unexpected response: %s", line)
	}
	
	return &AudioEncoderBridge{
		cmd:    cmd,
		stdin:  stdin,
		stdout: reader,
	}, nil
}

// Close stops the bridge
func (e *AudioEncoderBridge) Close() error {
	if e.stdin != nil {
		e.stdin.Close()
	}
	if e.cmd != nil && e.cmd.Process != nil {
		return e.cmd.Process.Kill()
	}
	return nil
}

// Infer runs inference on a mel window
func (e *AudioEncoderBridge) Infer(melWindow [][]float64) ([]float32, error) {
	// Flatten mel window (16, 80) to 1D array
	inputSize := 16 * 80
	inputData := make([]float64, inputSize)
	
	idx := 0
	for mel := 0; mel < 80; mel++ {
		for frame := 0; frame < 16; frame++ {
			inputData[idx] = melWindow[frame][mel]
			idx++
		}
	}
	
	// Create request
	request := map[string]interface{}{
		"input": inputData,
	}
	
	requestJSON, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	
	// Send request
	_, err = fmt.Fprintf(e.stdin, "%s\n", requestJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	
	// Read response
	responseLine, err := e.stdout.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}
	
	// Parse response
	var response map[string]interface{}
	err = json.Unmarshal([]byte(responseLine), &response)
	if err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	
	// Check for error
	if errMsg, ok := response["error"]; ok {
		return nil, fmt.Errorf("inference error: %v", errMsg)
	}
	
	// Extract output
	outputInterface, ok := response["output"]
	if !ok {
		return nil, fmt.Errorf("no output in response")
	}
	
	outputSlice, ok := outputInterface.([]interface{})
	if !ok {
		return nil, fmt.Errorf("output is not an array")
	}
	
	// Convert to float32
	output := make([]float32, len(outputSlice))
	for i, val := range outputSlice {
		floatVal, ok := val.(float64)
		if !ok {
			return nil, fmt.Errorf("output value is not a number")
		}
		output[i] = float32(floatVal)
	}
	
	return output, nil
}

// ProcessBatch processes multiple mel windows
func (e *AudioEncoderBridge) ProcessBatch(melWindows [][][]float64) ([][]float32, error) {
	results := make([][]float32, len(melWindows))
	
	for i, window := range melWindows {
		features, err := e.Infer(window)
		if err != nil {
			return nil, fmt.Errorf("failed to process window %d: %w", i, err)
		}
		results[i] = features
	}
	
	return results, nil
}

