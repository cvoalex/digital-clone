package unet

import (
	"fmt"

	onnxruntime "github.com/yalue/onnxruntime_go"
)

// Model wraps the ONNX U-Net model
type Model struct {
	session      *onnxruntime.AdvancedSession
	inputShape   []int64
	audioShape   []int64
	outputShape  []int64
	inputNames   []string
	outputNames  []string
}

// ModelConfig holds configuration for the U-Net model
type ModelConfig struct {
	ModelPath string
	Mode      string // "ave", "hubert", or "wenet"
}

// NewModel creates a new U-Net model instance
func NewModel(config ModelConfig) (*Model, error) {
	// Initialize ONNX Runtime
	err := onnxruntime.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime: %w", err)
	}

	// Set input shapes based on mode
	var audioShape []int64
	switch config.Mode {
	case "ave":
		audioShape = []int64{1, 32, 16, 16}
	case "hubert":
		audioShape = []int64{1, 32, 32, 32}
	case "wenet":
		audioShape = []int64{1, 256, 16, 32}
	default:
		return nil, fmt.Errorf("unknown mode: %s", config.Mode)
	}

	inputShape := []int64{1, 6, 320, 320}
	outputShape := []int64{1, 3, 320, 320}

	// Create input and output tensors info
	inputTensor, err := onnxruntime.NewTensor(inputShape, make([]float32, 1*6*320*320))
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	audioTensor, err := onnxruntime.NewTensor(audioShape, make([]float32, calculateSize(audioShape)))
	if err != nil {
		return nil, fmt.Errorf("failed to create audio tensor: %w", err)
	}
	defer audioTensor.Destroy()

	outputTensor, err := onnxruntime.NewTensor(outputShape, make([]float32, 1*3*320*320))
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Create session
	session, err := onnxruntime.NewAdvancedSession(
		config.ModelPath,
		[]string{"image", "audio"},
		[]string{"output"},
		[]onnxruntime.ArbitraryTensor{inputTensor, audioTensor},
		[]onnxruntime.ArbitraryTensor{outputTensor},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &Model{
		session:     session,
		inputShape:  inputShape,
		audioShape:  audioShape,
		outputShape: outputShape,
		inputNames:  []string{"image", "audio"},
		outputNames: []string{"output"},
	}, nil
}

// Predict runs inference on the model
// imageTensor: shape (1, 6, 320, 320)
// audioFeatures: shape based on mode
// Returns: output tensor shape (1, 3, 320, 320)
func (m *Model) Predict(imageTensor []float32, audioFeatures []float32) ([]float32, error) {
	// Validate input sizes
	expectedImageSize := int(m.inputShape[1] * m.inputShape[2] * m.inputShape[3])
	if len(imageTensor) != expectedImageSize {
		return nil, fmt.Errorf("invalid image tensor size: got %d, expected %d", len(imageTensor), expectedImageSize)
	}

	expectedAudioSize := calculateSize(m.audioShape)
	if len(audioFeatures) != expectedAudioSize {
		return nil, fmt.Errorf("invalid audio tensor size: got %d, expected %d", len(audioFeatures), expectedAudioSize)
	}

	// Create input tensors
	inputTensor, err := onnxruntime.NewTensor(m.inputShape, imageTensor)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	audioTensor, err := onnxruntime.NewTensor(m.audioShape, audioFeatures)
	if err != nil {
		return nil, fmt.Errorf("failed to create audio tensor: %w", err)
	}
	defer audioTensor.Destroy()

	// Create output tensor
	outputSize := int(m.outputShape[1] * m.outputShape[2] * m.outputShape[3])
	outputData := make([]float32, outputSize)
	outputTensor, err := onnxruntime.NewTensor(m.outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = m.session.Run(
		[]onnxruntime.ArbitraryTensor{inputTensor, audioTensor},
		[]onnxruntime.ArbitraryTensor{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Get output data
	output := outputTensor.GetData().([]float32)

	// Convert to 0-255 range and apply sigmoid if needed
	result := make([]float32, len(output))
	for i, v := range output {
		// The output is already sigmoid activated from the model
		// Scale to 0-255 range
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
func (m *Model) Close() error {
	if m.session != nil {
		return m.session.Destroy()
	}
	return nil
}

// GetInputShapes returns the expected input shapes
func (m *Model) GetInputShapes() ([]int64, []int64) {
	return m.inputShape, m.audioShape
}

// GetOutputShape returns the output shape
func (m *Model) GetOutputShape() []int64 {
	return m.outputShape
}

// calculateSize calculates the total size of a tensor shape
func calculateSize(shape []int64) int {
	size := 1
	for _, dim := range shape {
		size *= int(dim)
	}
	return size
}

