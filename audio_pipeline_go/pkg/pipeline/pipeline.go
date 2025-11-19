package pipeline

import (
	"fmt"
	"math"

	"github.com/alexanderrusich/audio_pipeline_go/pkg/mel"
	"github.com/alexanderrusich/audio_pipeline_go/pkg/onnx"
)

type AudioEncoder interface {
	ProcessBatch(melWindows [][][]float64) ([][]float32, error)
	Close() error
}

// Pipeline handles the complete audio processing pipeline
type Pipeline struct {
	melProcessor  *mel.Processor
	audioEncoder  AudioEncoder
	fps           int
	mode          string
}

// New creates a new audio processing pipeline
func New(modelPath string, fps int, mode string) (*Pipeline, error) {
	melProc := mel.NewProcessor()
	
	// Use Python bridge for ONNX inference
	encoder, err := onnx.NewAudioEncoderBridge(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create audio encoder: %w", err)
	}
	
	return &Pipeline{
		melProcessor: melProc,
		audioEncoder: encoder,
		fps:          fps,
		mode:         mode,
	}, nil
}

// Close cleans up resources
func (p *Pipeline) Close() error {
	return p.audioEncoder.Close()
}

// ProcessAudioFile processes an audio file through the complete pipeline
func (p *Pipeline) ProcessAudioFile(audioPath string) (*ProcessedAudio, error) {
	fmt.Println("Step 1: Loading audio file...")
	audio, err := p.melProcessor.LoadWAV(audioPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load audio: %w", err)
	}
	fmt.Printf("  Duration: %.2f seconds\n", float64(len(audio))/float64(p.melProcessor.SampleRate))
	
	fmt.Println("Step 2: Converting to mel spectrogram...")
	melSpec, err := p.melProcessor.Process(audio)
	if err != nil {
		return nil, fmt.Errorf("failed to process mel spectrogram: %w", err)
	}
	fmt.Printf("  Mel shape: (%d, %d)\n", len(melSpec), len(melSpec[0]))
	
	fmt.Println("Step 3: Extracting mel windows...")
	nFrames := p.melProcessor.GetFrameCount(melSpec, p.fps)
	fmt.Printf("  Total frames: %d\n", nFrames)
	
	melWindows := make([][][]float64, nFrames)
	for i := 0; i < nFrames; i++ {
		window, err := p.melProcessor.CropAudioWindow(melSpec, i, p.fps)
		if err != nil {
			return nil, fmt.Errorf("failed to crop window %d: %w", i, err)
		}
		melWindows[i] = window
	}
	
	fmt.Println("Step 4: Processing through AudioEncoder...")
	audioFeatures, err := p.audioEncoder.ProcessBatch(melWindows)
	if err != nil {
		return nil, fmt.Errorf("failed to encode audio: %w", err)
	}
	fmt.Printf("  Encoded %d frames\n", len(audioFeatures))
	
	fmt.Println("Step 5: Adding temporal padding...")
	paddedFeatures := p.addTemporalPadding(audioFeatures)
	
	return &ProcessedAudio{
		MelSpec:        melSpec,
		AudioFeatures:  paddedFeatures,
		NFrames:        nFrames,
	}, nil
}

// addTemporalPadding adds padding by repeating first and last frames
func (p *Pipeline) addTemporalPadding(features [][]float32) [][]float32 {
	padded := make([][]float32, len(features)+2)
	
	// Repeat first frame
	padded[0] = make([]float32, len(features[0]))
	copy(padded[0], features[0])
	
	// Copy all frames
	for i := 0; i < len(features); i++ {
		padded[i+1] = features[i]
	}
	
	// Repeat last frame
	padded[len(padded)-1] = make([]float32, len(features[len(features)-1]))
	copy(padded[len(padded)-1], features[len(features)-1])
	
	return padded
}

// GetFrameFeatures extracts features for a specific frame with context
func (p *Pipeline) GetFrameFeatures(allFeatures [][]float32, frameIdx int) ([]float32, error) {
	contextSize := 8
	left := frameIdx - contextSize
	right := frameIdx + contextSize
	
	padLeft := 0
	
	if left < 0 {
		padLeft = -left
		left = 0
	}
	if right > len(allFeatures) {
		right = len(allFeatures)
	}
	
	// Extract window
	windowSize := 16
	featureSize := len(allFeatures[0])
	window := make([]float32, windowSize*featureSize)
	
	// Pad left with zeros
	for i := 0; i < padLeft; i++ {
		for j := 0; j < featureSize; j++ {
			window[i*featureSize+j] = 0
		}
	}
	
	// Copy actual features
	srcIdx := left
	dstIdx := padLeft
	for srcIdx < right {
		copy(window[dstIdx*featureSize:(dstIdx+1)*featureSize], allFeatures[srcIdx])
		srcIdx++
		dstIdx++
	}
	
	// Pad right with zeros
	for i := dstIdx; i < windowSize; i++ {
		for j := 0; j < featureSize; j++ {
			window[i*featureSize+j] = 0
		}
	}
	
	return window, nil
}

// ReshapeForModel reshapes features for the U-Net model
func (p *Pipeline) ReshapeForModel(features []float32) ([]float32, error) {
	switch p.mode {
	case "ave":
		// Reshape to (32, 16, 16) = 8192 values
		if len(features) != 16*512 {
			return nil, fmt.Errorf("expected 8192 values, got %d", len(features))
		}
		
		// Just reshape - the data stays the same
		reshaped := make([]float32, 32*16*16)
		copy(reshaped, features[:32*16*16])
		
		return reshaped, nil
		
	case "hubert":
		// Pad to (32, 32, 32) = 32768 values
		reshaped := make([]float32, 32*32*32)
		srcSize := len(features)
		if srcSize > len(reshaped) {
			srcSize = len(reshaped)
		}
		copy(reshaped, features[:srcSize])
		return reshaped, nil
		
	case "wenet":
		// Pad to (256, 16, 32) = 131072 values
		reshaped := make([]float32, 256*16*32)
		srcSize := len(features)
		if srcSize > len(reshaped) {
			srcSize = len(reshaped)
		}
		copy(reshaped, features[:srcSize])
		return reshaped, nil
		
	default:
		return nil, fmt.Errorf("unknown mode: %s", p.mode)
	}
}

// ProcessedAudio contains the results of audio processing
type ProcessedAudio struct {
	MelSpec       [][]float64
	AudioFeatures [][]float32
	NFrames       int
}

// GetStats returns statistics about the processed audio
func (a *ProcessedAudio) GetStats() map[string]interface{} {
	stats := make(map[string]interface{})
	
	// Mel spec stats
	stats["mel_shape"] = []int{len(a.MelSpec), len(a.MelSpec[0])}
	melMin, melMax := minMax2D(a.MelSpec)
	stats["mel_range"] = []float64{melMin, melMax}
	
	// Audio features stats
	stats["features_shape"] = []int{len(a.AudioFeatures), len(a.AudioFeatures[0])}
	featMin, featMax := minMaxFeatures(a.AudioFeatures)
	stats["features_range"] = []float32{featMin, featMax}
	
	stats["n_frames"] = a.NFrames
	
	return stats
}

func minMax2D(data [][]float64) (float64, float64) {
	min := math.Inf(1)
	max := math.Inf(-1)
	
	for i := range data {
		for j := range data[i] {
			if data[i][j] < min {
				min = data[i][j]
			}
			if data[i][j] > max {
				max = data[i][j]
			}
		}
	}
	
	return min, max
}

func minMaxFeatures(data [][]float32) (float32, float32) {
	var min float32 = math.MaxFloat32
	var max float32 = -math.MaxFloat32
	
	for i := range data {
		for j := range data[i] {
			if data[i][j] < min {
				min = data[i][j]
			}
			if data[i][j] > max {
				max = data[i][j]
			}
		}
	}
	
	return min, max
}

