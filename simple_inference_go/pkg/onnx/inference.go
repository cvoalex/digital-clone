package onnx

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// UNetModel wraps the ONNX U-Net model
type UNetModel struct {
	session *ort.DynamicAdvancedSession
}

// NewUNetModel creates a new U-Net model
func NewUNetModel(modelPath string) (*UNetModel, error) {
	// Initialize ONNX Runtime environment
	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime: %w", err)
	}

	// Create session options
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()

	// Create session
	session, err := ort.NewDynamicAdvancedSession(modelPath, 
		[]string{"input", "audio"}, 
		[]string{"output"}, 
		options)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &UNetModel{
		session: session,
	}, nil
}

// Predict runs inference on the model
// imageTensor: 6-channel input (original + masked) shape (1, 6, 320, 320)
// audioFeatures: audio features shape (1, 32, 16, 16)
// Returns: output tensor shape (1, 3, 320, 320), values 0-255
func (m *UNetModel) Predict(imageTensor []float32, audioFeatures []float32) ([]float32, error) {
	// Create input tensors
	imageShape := ort.NewShape(1, 6, 320, 320)
	imageTensorONNX, err := ort.NewTensor(imageShape, imageTensor)
	if err != nil {
		return nil, fmt.Errorf("failed to create image tensor: %w", err)
	}
	defer imageTensorONNX.Destroy()

	audioShape := ort.NewShape(1, 32, 16, 16)
	audioTensorONNX, err := ort.NewTensor(audioShape, audioFeatures)
	if err != nil {
		return nil, fmt.Errorf("failed to create audio tensor: %w", err)
	}
	defer audioTensorONNX.Destroy()

	// Create output tensor
	outputShape := ort.NewShape(1, 3, 320, 320)
	outputData := make([]float32, 1*3*320*320)
	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = m.session.Run(
		[]ort.Value{imageTensorONNX, audioTensorONNX},
		[]ort.Value{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Get output data
	outputData = outputTensor.GetData()

	// Convert to 0-255 range (model outputs sigmoid [0, 1])
	result := make([]float32, len(outputData))
	for i, v := range outputData {
		result[i] = v * 255.0
		if result[i] < 0 {
			result[i] = 0
		}
		if result[i] > 255 {
			result[i] = 255
		}
	}

	return result, nil
}

// Close releases model resources
func (m *UNetModel) Close() error {
	if m.session != nil {
		m.session.Destroy()
	}
	return nil
}

