package audio

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// AudioEncoder wraps the audio encoder ONNX model
type AudioEncoder struct {
	session *ort.DynamicAdvancedSession
}

// NewAudioEncoder creates a new audio encoder
func NewAudioEncoder(modelPath string) (*AudioEncoder, error) {
	// Create session options
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()

	// Create session
	// Audio encoder expects input shape: (1, 1, 80, 16)
	// Outputs: (1, 512)
	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"mel"},
		[]string{"emb"},
		options)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &AudioEncoder{
		session: session,
	}, nil
}

// Encode processes a mel spectrogram window into audio features
// melWindow: shape (1, 1, 80, 16) - one window of mel spectrogram
// Returns: (512,) audio features
func (e *AudioEncoder) Encode(melWindow []float32) ([]float32, error) {
	// Create input tensor
	inputShape := ort.NewShape(1, 1, 80, 16)
	inputTensor, err := ort.NewTensor(inputShape, melWindow)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensor
	outputShape := ort.NewShape(1, 512)
	outputData := make([]float32, 512)
	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = e.session.Run(
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Get output
	result := outputTensor.GetData()

	return result, nil
}

// Close releases resources
func (e *AudioEncoder) Close() error {
	if e.session != nil {
		e.session.Destroy()
	}
	return nil
}

