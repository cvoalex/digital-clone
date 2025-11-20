package compositor

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"

	"github.com/alexanderrusich/simple_inference_go/pkg/audio"
	"github.com/alexanderrusich/simple_inference_go/pkg/loader"
	"github.com/alexanderrusich/simple_inference_go/pkg/mel"
	"github.com/alexanderrusich/simple_inference_go/pkg/onnx"
)

// Compositor handles the frame generation process
type Compositor struct {
	model          *onnx.UNetModel
	audioEncoder   *audio.AudioEncoder
	melProcessor   *mel.Processor
	cropRectangles map[string]loader.CropRect
}

// NewCompositor creates a new compositor
func NewCompositor(modelPath string, audioEncoderPath string, cropRectsPath string) (*Compositor, error) {
	// Load U-Net model
	model, err := onnx.NewUNetModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load U-Net model: %w", err)
	}

	// Load audio encoder
	audioEnc, err := audio.NewAudioEncoder(audioEncoderPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load audio encoder: %w", err)
	}

	// Create mel processor
	melProc := mel.NewProcessor()

	// Load crop rectangles
	rects, err := loader.LoadCropRectangles(cropRectsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load crop rectangles: %w", err)
	}

	return &Compositor{
		model:          model,
		audioEncoder:   audioEnc,
		melProcessor:   melProc,
		cropRectangles: rects,
	}, nil
}

// ProcessAudioFile processes a WAV file into audio features
func (c *Compositor) ProcessAudioFile(audioPath string) ([][]float32, error) {
	fmt.Printf("Processing audio file: %s\n", audioPath)

	// Load WAV file
	audioSamples, err := c.melProcessor.LoadWAV(audioPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load WAV: %w", err)
	}
	fmt.Printf("  Loaded audio: %d samples\n", len(audioSamples))

	// Process to mel spectrogram
	melSpec, err := c.melProcessor.Process(audioSamples)
	if err != nil {
		return nil, fmt.Errorf("failed to process mel: %w", err)
	}
	fmt.Printf("  Generated mel spectrogram: %d x %d\n", len(melSpec), len(melSpec[0]))

	// Get number of frames
	fps := 25
	numFrames := c.melProcessor.GetFrameCount(melSpec, fps)
	fmt.Printf("  Number of frames: %d\n", numFrames)

	// Encode each frame
	audioFeatures := make([][]float32, numFrames)

	for i := 0; i < numFrames; i++ {
		// Crop audio window for this frame
		melWindow, err := c.melProcessor.CropAudioWindow(melSpec, i, fps)
		if err != nil {
			return nil, fmt.Errorf("failed to crop window %d: %w", i, err)
		}

		// Flatten mel window to (1, 1, 80, 16) format
		melTensor := flattenMelWindow(melWindow)

		// Encode
		features, err := c.audioEncoder.Encode(melTensor)
		if err != nil {
			return nil, fmt.Errorf("failed to encode window %d: %w", i, err)
		}

		audioFeatures[i] = features

		if (i+1)%100 == 0 {
			fmt.Printf("  Encoded %d/%d frames\n", i+1, numFrames)
		}
	}

	fmt.Printf("✓ Generated %d audio feature frames\n", len(audioFeatures))
	return audioFeatures, nil
}

// GenerateFrames generates all frames
func (c *Compositor) GenerateFrames(
	roisDir string,
	maskedDir string,
	fullBodyDir string,
	audioFeatures [][]float32,
	outputDir string,
	numFrames int,
) error {
	// Use provided audio features
	audioFeats := audioFeatures

	// Create output directory
	err := os.MkdirAll(outputDir, 0755)
	if err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	fmt.Printf("Generating %d frames...\n", numFrames)

	// Process each frame
	for i := 1; i <= numFrames; i++ {
		if i%50 == 0 || i == 1 {
			fmt.Printf("Processing frame %d/%d...\n", i, numFrames)
		}

		// Load pre-cut frames
		roiPath := filepath.Join(roisDir, fmt.Sprintf("%d.jpg", i))
		maskedPath := filepath.Join(maskedDir, fmt.Sprintf("%d.jpg", i))
		fullBodyPath := filepath.Join(fullBodyDir, fmt.Sprintf("%d.jpg", i))

		roiImg, err := loader.LoadImage(roiPath)
		if err != nil {
			return fmt.Errorf("failed to load ROI %d: %w", i, err)
		}

		maskedImg, err := loader.LoadImage(maskedPath)
		if err != nil {
			return fmt.Errorf("failed to load masked %d: %w", i, err)
		}

		fullBodyImg, err := loader.LoadImage(fullBodyPath)
		if err != nil {
			return fmt.Errorf("failed to load full body %d: %w", i, err)
		}

		// Convert to tensors (normalized to [0, 1])
		roiTensor := loader.ImageToTensor(roiImg, true)
		maskedTensor := loader.ImageToTensor(maskedImg, true)

		// Concatenate into 6-channel input
		imageTensor := append(roiTensor, maskedTensor...)

		// Get audio features for this frame (index i-1 since audio is 0-indexed but frames are 1-indexed)
		audioIdx := i - 1
		if audioIdx >= len(audioFeats) {
			audioIdx = len(audioFeats) - 1
		}
		audioTensor := reshapeAudioFeatures(audioFeats[audioIdx])

		// DEBUG: Save audio tensor for first 5 frames
		if i <= 5 {
			debugPath := filepath.Join(outputDir, "..", fmt.Sprintf("debug_audio_go_frame%d.bin", i))
			debugFile, _ := os.Create(debugPath)
			binary.Write(debugFile, binary.LittleEndian, audioTensor)
			debugFile.Close()
			fmt.Printf("    DEBUG: Saved audio tensor for frame %d\n", i)
		}

		// Run inference
		output, err := c.model.Predict(imageTensor, audioTensor)
		if err != nil {
			return fmt.Errorf("inference failed for frame %d: %w", i, err)
		}

		// Convert output tensor to image
		generatedImg := loader.TensorToImage(output, 320, 320)

		// Get crop rectangle
		rectKey := fmt.Sprintf("%d", i-1) // JSON uses 0-indexed keys
		cropRect, ok := c.cropRectangles[rectKey]
		if !ok {
			return fmt.Errorf("no crop rectangle for frame %d", i)
		}

		// Paste into full frame
		finalFrame := loader.PasteIntoFrame(fullBodyImg, generatedImg, cropRect.Rect)

		// Save output
		outputPath := filepath.Join(outputDir, fmt.Sprintf("frame_%05d.jpg", i))
		err = loader.SaveImage(outputPath, finalFrame)
		if err != nil {
			return fmt.Errorf("failed to save frame %d: %w", i, err)
		}
	}

	fmt.Printf("✓ Generated %d frames successfully!\n", numFrames)
	return nil
}

// Close releases resources
func (c *Compositor) Close() error {
	if c.audioEncoder != nil {
		c.audioEncoder.Close()
	}
	if c.model != nil {
		return c.model.Close()
	}
	return nil
}

// flattenMelWindow converts mel window to tensor format
func flattenMelWindow(melWindow [][]float64) []float32 {
	// Input: (16, 80) mel window
	// Output: (1, 1, 80, 16) flattened to (1280,) float32 array

	result := make([]float32, 80*16)

	// Convert from (16, 80) to (80, 16) and flatten
	idx := 0
	for mel := 0; mel < 80; mel++ {
		for frame := 0; frame < 16; frame++ {
			result[idx] = float32(melWindow[frame][mel])
			idx++
		}
	}

	return result
}

// reshapeAudioFeatures reshapes audio features to (1, 32, 16, 16)
func reshapeAudioFeatures(features []float32) []float32 {
	// The features are 512 floats that need to be reshaped to (32, 16, 16) = 8192
	// But we only have 512, so we need to upsample or pad

	// For now, just return the features as-is and let the model handle it
	// In practice, you'd need to properly reshape based on how the features were generated
	
	// Create (32, 16, 16) = 8192 tensor
	target := make([]float32, 32*16*16)
	
	// Simple approach: repeat the 512 features to fill 8192
	for i := 0; i < len(target); i++ {
		target[i] = features[i%512]
	}

	return target
}

