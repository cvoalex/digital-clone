package generator

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/alexanderrusich/digital-clone/frame_generation_go/pkg/imageproc"
	"github.com/alexanderrusich/digital-clone/frame_generation_go/pkg/unet"
	"gocv.io/x/gocv"
)

// FrameGenerator handles frame generation from audio features and templates
type FrameGenerator struct {
	model     *unet.Model
	processor *imageproc.ImageProcessor
	mode      string
}

// Config holds configuration for the frame generator
type Config struct {
	ModelPath string
	Mode      string
}

// NewFrameGenerator creates a new frame generator
func NewFrameGenerator(config Config) (*FrameGenerator, error) {
	// Initialize U-Net model
	model, err := unet.NewModel(unet.ModelConfig{
		ModelPath: config.ModelPath,
		Mode:      config.Mode,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create model: %w", err)
	}

	// Initialize image processor
	processor := imageproc.NewImageProcessor()

	return &FrameGenerator{
		model:     model,
		processor: processor,
		mode:      config.Mode,
	}, nil
}

// GenerateFrame generates a single frame from template image and audio features
func (g *FrameGenerator) GenerateFrame(
	templateImg gocv.Mat,
	landmarks []imageproc.Landmark,
	audioFeatures []float32,
) (gocv.Mat, error) {
	// Crop face region
	cropImg, coords := g.processor.CropFaceRegion(templateImg, landmarks)
	defer cropImg.Close()

	// Get original crop size
	originalHeight := cropImg.Rows()
	originalWidth := cropImg.Cols()

	// Resize to 328x328
	crop328 := g.processor.ResizeImage(cropImg, 328, 328)
	defer crop328.Close()

	// Extract inner region [4:324, 4:324] -> 320x320
	innerCrop := crop328.Region(gocv.NewRect(4, 4, 320, 320))
	defer innerCrop.Close()

	// Prepare input tensors
	imageTensor, err := g.processor.PrepareInputTensors(innerCrop)
	if err != nil {
		return gocv.Mat{}, fmt.Errorf("failed to prepare input tensors: %w", err)
	}

	// Run U-Net inference
	output, err := g.model.Predict(imageTensor, audioFeatures)
	if err != nil {
		return gocv.Mat{}, fmt.Errorf("inference failed: %w", err)
	}

	// Convert output tensor to image
	generatedRegion := g.processor.TensorToMat(output, 320, 320)
	defer generatedRegion.Close()

	// Paste back into full frame
	outputFrame := g.processor.PasteGeneratedRegion(
		templateImg,
		generatedRegion,
		coords,
		originalHeight,
		originalWidth,
	)

	return outputFrame, nil
}

// GenerateFramesFromSequence generates frames from a template image sequence
func (g *FrameGenerator) GenerateFramesFromSequence(
	imgDir string,
	lmsDir string,
	audioFeatures [][]float32,
	startFrame int,
) ([]gocv.Mat, error) {
	numFrames := len(audioFeatures)

	// Get number of template images
	files, err := os.ReadDir(imgDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read image directory: %w", err)
	}

	lenImg := 0
	for _, file := range files {
		if filepath.Ext(file.Name()) == ".jpg" {
			lenImg++
		}
	}
	lenImg-- // Max index

	fmt.Printf("Generating %d frames from %d template images\n", numFrames, lenImg+1)

	frames := make([]gocv.Mat, 0, numFrames)

	// Initialize ping-pong motion
	stepStride := 0
	imgIdx := 0

	for i := 0; i < numFrames; i++ {
		// Ping-pong logic
		if imgIdx > lenImg-1 {
			stepStride = -1
		}
		if imgIdx < 1 {
			stepStride = 1
		}
		imgIdx += stepStride

		// Load template image and landmarks
		imgPath := filepath.Join(imgDir, fmt.Sprintf("%d.jpg", imgIdx+startFrame))
		lmsPath := filepath.Join(lmsDir, fmt.Sprintf("%d.lms", imgIdx+startFrame))

		templateImg, err := g.processor.LoadImage(imgPath)
		if err != nil {
			return frames, fmt.Errorf("failed to load image %s: %w", imgPath, err)
		}

		landmarks, err := g.processor.LoadLandmarks(lmsPath)
		if err != nil {
			templateImg.Close()
			return frames, fmt.Errorf("failed to load landmarks %s: %w", lmsPath, err)
		}

		// Generate frame
		frame, err := g.GenerateFrame(templateImg, landmarks, audioFeatures[i])
		templateImg.Close()

		if err != nil {
			return frames, fmt.Errorf("failed to generate frame %d: %w", i, err)
		}

		frames = append(frames, frame)

		if (i+1)%100 == 0 {
			fmt.Printf("Generated %d/%d frames\n", i+1, numFrames)
		}
	}

	return frames, nil
}

// SaveFrames saves frames to disk
func (g *FrameGenerator) SaveFrames(frames []gocv.Mat, outputDir string, prefix string) error {
	err := os.MkdirAll(outputDir, 0755)
	if err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	for i, frame := range frames {
		outputPath := filepath.Join(outputDir, fmt.Sprintf("%s_%05d.jpg", prefix, i))
		ok := gocv.IMWrite(outputPath, frame)
		if !ok {
			return fmt.Errorf("failed to write frame %d to %s", i, outputPath)
		}
	}

	fmt.Printf("Saved %d frames to %s\n", len(frames), outputDir)
	return nil
}

// GetAudioFeaturesForFrame extracts audio features for a specific frame
// This replicates the Python get_audio_features logic
func (g *FrameGenerator) GetAudioFeaturesForFrame(
	allFeatures [][]float32,
	frameIdx int,
) []float32 {
	// Extract window around frame
	left := frameIdx - 8
	right := frameIdx + 8
	padLeft := 0
	padRight := 0

	if left < 0 {
		padLeft = -left
		left = 0
	}
	if right > len(allFeatures) {
		padRight = right - len(allFeatures)
		right = len(allFeatures)
	}

	// Extract features
	features := make([][]float32, 0, 16)

	// Pad left
	for i := 0; i < padLeft; i++ {
		features = append(features, make([]float32, len(allFeatures[0])))
	}

	// Copy actual features
	features = append(features, allFeatures[left:right]...)

	// Pad right
	for i := 0; i < padRight; i++ {
		features = append(features, make([]float32, len(allFeatures[0])))
	}

	// Reshape based on mode
	return g.reshapeAudioFeatures(features)
}

// reshapeAudioFeatures reshapes audio features based on mode
func (g *FrameGenerator) reshapeAudioFeatures(features [][]float32) []float32 {
	// Flatten features
	flat := make([]float32, 0)
	for _, feat := range features {
		flat = append(flat, feat...)
	}

	// The features are already in the correct size, just return them
	return flat
}

// Close releases resources
func (g *FrameGenerator) Close() error {
	if g.model != nil {
		return g.model.Close()
	}
	return nil
}

// LoadAudioFeatures loads audio features from a numpy file
// This is a placeholder - in practice, you'd use a proper numpy reader
func LoadAudioFeatures(path string) ([][]float32, error) {
	// This would need a proper numpy file reader
	// For now, return an error suggesting to use the Python export
	return nil, fmt.Errorf("numpy loading not implemented in Go - use Python to export to binary format")
}

// Helper function to calculate standard deviation
func stdDev(data []float32) float32 {
	if len(data) == 0 {
		return 0
	}

	var sum float32
	for _, v := range data {
		sum += v
	}
	mean := sum / float32(len(data))

	var variance float32
	for _, v := range data {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float32(len(data))

	return float32(math.Sqrt(float64(variance)))
}

