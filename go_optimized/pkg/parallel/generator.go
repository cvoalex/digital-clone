package parallel

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"os"
	"path/filepath"
	"runtime"
	"sync/atomic"

	"github.com/alexanderrusich/go_optimized/pkg/batch"
	ort "github.com/yalue/onnxruntime_go"
)

// OptimizedGenerator is a highly optimized frame generator
type OptimizedGenerator struct {
	// Model session pools for TRUE parallel inference
	audioEncoderPool *SessionPool
	generatorPool    *SessionPool
	
	// Batch processor with memory pools
	batchProcessor *batch.BatchProcessor
	
	// Data
	cropRectangles map[string]CropRect
	sandersDir     string
	
	// Statistics
	framesProcessed atomic.Int64
}

type CropRect struct {
	Rect []int `json:"rect"`
}

// NewOptimizedGenerator creates an optimized generator
func NewOptimizedGenerator(sandersDir string, batchSize int) (*OptimizedGenerator, error) {
	numWorkers := runtime.NumCPU() // Use all CPU cores
	
	fmt.Printf("Creating optimized generator:\n")
	fmt.Printf("  CPU cores: %d\n", numWorkers)
	fmt.Printf("  Batch size: %d\n", batchSize)
	fmt.Printf("  Workers: %d\n", numWorkers)
	
	// Initialize ONNX Runtime
	ort.InitializeEnvironment() // Ignore error if already initialized
	
	// Load models as session pools (TRUE parallel inference!)
	audioPath := filepath.Join(sandersDir, "models/audio_encoder.onnx")
	genPath := filepath.Join(sandersDir, "models/generator.onnx")
	
	// Create session pool for generator (one session per worker)
	genPool, err := NewSessionPool(genPath, []string{"input", "audio"}, []string{"output"}, numWorkers)
	if err != nil {
		return nil, fmt.Errorf("failed to create generator pool: %w", err)
	}
	
	// Audio encoder pool (smaller, audio processing is sequential anyway)
	audioPool, err := NewSessionPool(audioPath, []string{"mel"}, []string{"emb"}, 2)
	if err != nil {
		genPool.Close()
		return nil, fmt.Errorf("failed to create audio encoder pool: %w", err)
	}
	
	// Load crop rectangles
	cropPath := filepath.Join(sandersDir, "cache/crop_rectangles.json")
	cropFile, err := os.Open(cropPath)
	if err != nil {
		return nil, err
	}
	defer cropFile.Close()
	
	var rects map[string]CropRect
	err = json.NewDecoder(cropFile).Decode(&rects)
	if err != nil {
		return nil, err
	}
	
	// Create batch processor
	bp := batch.NewBatchProcessor(batchSize, numWorkers)
	
	return &OptimizedGenerator{
		audioEncoderPool: audioPool,
		generatorPool:    genPool,
		batchProcessor:   bp,
		cropRectangles:   rects,
		sandersDir:       sandersDir,
	}, nil
}

// ProcessAudioParallel processes audio in parallel batches
func (g *OptimizedGenerator) ProcessAudioParallel(audioPath string) ([][]float32, error) {
	fmt.Printf("Processing audio (parallel): %s\n", audioPath)
	
	// Load audio (TODO: integrate mel processor)
	// For now, load from binary if exists
	binPath := filepath.Join(g.sandersDir, "aud_ave.bin")
	
	file, err := os.Open(binPath)
	if err != nil {
		return nil, fmt.Errorf("audio processing not yet implemented, use pre-computed: %w", err)
	}
	defer file.Close()
	
	stat, err := file.Stat()
	if err != nil {
		return nil, err
	}
	
	numFloats := int(stat.Size()) / 4
	data := make([]float32, numFloats)
	err = binary.Read(file, binary.LittleEndian, data)
	if err != nil {
		return nil, err
	}
	
	// Reshape to [num_frames][512]
	featureSize := 512
	numFrames := numFloats / featureSize
	
	features := make([][]float32, numFrames)
	for i := 0; i < numFrames; i++ {
		features[i] = data[i*featureSize : (i+1)*featureSize]
	}
	
	fmt.Printf("  Loaded %d audio feature frames\n", numFrames)
	return features, nil
}

// GenerateFramesOptimized generates frames with optimizations
func (g *OptimizedGenerator) GenerateFramesOptimized(
	audioFeatures [][]float32,
	numFrames int,
	outputDir string,
) error {
	fmt.Printf("Generating %d frames (optimized)...\n", numFrames)
	
	// Create output directory
	os.MkdirAll(outputDir, 0755)
	
	// Create batches
	batches := g.batchProcessor.CreateBatches(numFrames)
	fmt.Printf("  Created %d batches of ~%d frames each\n", 
		len(batches), g.batchProcessor.Stats())
	
	// Process each batch
	for batchIdx, batch := range batches {
		fmt.Printf("  Batch %d/%d: frames %d-%d\n", 
			batchIdx+1, len(batches), batch.StartIdx+1, batch.EndIdx)
		
		err := g.batchProcessor.ProcessBatchParallel(batch, func(frameIdx int, tensor6, tensor3, audioTensor []float32) error {
			return g.processFrame(frameIdx, audioFeatures, tensor6, tensor3, audioTensor, outputDir)
		})
		
		if err != nil {
			return err
		}
		
		processed := g.framesProcessed.Load()
		fmt.Printf("    Progress: %d/%d frames\n", processed, numFrames)
	}
	
	fmt.Printf("✓ Generated %d frames\n", numFrames)
	return nil
}

// processFrame processes a single frame (called in parallel)
func (g *OptimizedGenerator) processFrame(
	frameIdx int,
	audioFeatures [][]float32,
	tensor6, tensor3, audioTensor []float32,
	outputDir string,
) error {
	// Load images (reuse buffers)
	roiPath := filepath.Join(g.sandersDir, "rois_320", fmt.Sprintf("%d.jpg", frameIdx))
	maskedPath := filepath.Join(g.sandersDir, "model_inputs", fmt.Sprintf("%d.jpg", frameIdx))
	fullBodyPath := filepath.Join(g.sandersDir, "full_body_img", fmt.Sprintf("%d.jpg", frameIdx))
	
	roiImg, err := loadImageFast(roiPath)
	if err != nil {
		return err
	}
	
	maskedImg, err := loadImageFast(maskedPath)
	if err != nil {
		return err
	}
	
	fullBodyImg, err := loadImageFast(fullBodyPath)
	if err != nil {
		return err
	}
	
	// Convert to tensors (reuse tensor6 buffer)
	imageToTensorBGR(roiImg, tensor6[:1*3*320*320], true)
	imageToTensorBGR(maskedImg, tensor6[1*3*320*320:], true)
	
	// Get audio features
	audioIdx := frameIdx - 1
	if audioIdx >= len(audioFeatures) {
		audioIdx = len(audioFeatures) - 1
	}
	reshapeAudioFeatures(audioFeatures[audioIdx], audioTensor)
	
	// Get a generator session from pool (blocks if all busy)
	session := g.generatorPool.Get()
	output, err := g.runGeneratorWithSession(session, tensor6, audioTensor)
	g.generatorPool.Put(session) // Return session to pool
	
	if err != nil {
		return err
	}
	
	// Copy output to tensor3
	copy(tensor3, output)
	
	// Convert to image
	generatedImg := tensorToImageBGR(tensor3, 320, 320)
	
	// Paste into full frame
	rectKey := fmt.Sprintf("%d", frameIdx-1)
	cropRect, ok := g.cropRectangles[rectKey]
	if !ok {
		return fmt.Errorf("no crop rect for frame %d", frameIdx)
	}
	
	finalImg := pasteIntoFrameFast(fullBodyImg, generatedImg, cropRect.Rect)
	
	// Save
	outputPath := filepath.Join(outputDir, fmt.Sprintf("frame_%05d.jpg", frameIdx))
	err = saveJPEGFast(finalImg, outputPath)
	if err != nil {
		return err
	}
	
	// Update counter
	g.framesProcessed.Add(1)
	
	return nil
}

// runGeneratorWithSession runs the generator model with a specific session
func (g *OptimizedGenerator) runGeneratorWithSession(session *ort.DynamicAdvancedSession, imageTensor, audioTensor []float32) ([]float32, error) {
	imageShape := ort.NewShape(1, 6, 320, 320)
	audioShape := ort.NewShape(1, 32, 16, 16)
	outputShape := ort.NewShape(1, 3, 320, 320)
	
	imageTensorONNX, err := ort.NewTensor(imageShape, imageTensor)
	if err != nil {
		return nil, err
	}
	defer imageTensorONNX.Destroy()
	
	audioTensorONNX, err := ort.NewTensor(audioShape, audioTensor)
	if err != nil {
		return nil, err
	}
	defer audioTensorONNX.Destroy()
	
	outputData := make([]float32, 1*3*320*320)
	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()
	
	err = session.Run(
		[]ort.Value{imageTensorONNX, audioTensorONNX},
		[]ort.Value{outputTensor},
	)
	if err != nil {
		return nil, err
	}
	
	result := outputTensor.GetData()
	
	// Scale to 0-255
	for i := range result {
		result[i] *= 255.0
		if result[i] < 0 {
			result[i] = 0
		}
		if result[i] > 255 {
			result[i] = 255
		}
	}
	
	return result, nil
}

// Fast helper functions using direct memory access

func loadImageFast(path string) (*image.RGBA, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	
	img, err := jpeg.Decode(file)
	if err != nil {
		return nil, err
	}
	
	// Convert to RGBA if needed
	rgba, ok := img.(*image.RGBA)
	if !ok {
		bounds := img.Bounds()
		rgba = image.NewRGBA(bounds)
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				rgba.Set(x, y, img.At(x, y))
			}
		}
	}
	
	return rgba, nil
}

func imageToTensorBGR(img *image.RGBA, tensor []float32, normalize bool) {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	
	scale := float32(1.0)
	if normalize {
		scale = 1.0 / 255.0
	}
	
	// Direct pixel buffer access (fast!)
	pix := img.Pix
	
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			pixIdx := (y*width + x) * 4
			tensorIdx := y*width + x
			
			r := float32(pix[pixIdx+0]) * scale
			g := float32(pix[pixIdx+1]) * scale
			b := float32(pix[pixIdx+2]) * scale
			
			// BGR order
			tensor[0*height*width+tensorIdx] = b
			tensor[1*height*width+tensorIdx] = g
			tensor[2*height*width+tensorIdx] = r
		}
	}
}

func tensorToImageBGR(tensor []float32, width, height int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	pix := img.Pix
	
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			pixIdx := (y*width + x) * 4
			tensorIdx := y*width + x
			
			// BGR to RGB
			b := uint8(tensor[0*height*width+tensorIdx])
			g := uint8(tensor[1*height*width+tensorIdx])
			r := uint8(tensor[2*height*width+tensorIdx])
			
			pix[pixIdx+0] = r
			pix[pixIdx+1] = g
			pix[pixIdx+2] = b
			pix[pixIdx+3] = 255
		}
	}
	
	return img
}

func pasteIntoFrameFast(fullFrame, generated *image.RGBA, rect []int) *image.RGBA {
	x1, y1, x2, y2 := rect[0], rect[1], rect[2], rect[3]
	
	output := image.NewRGBA(fullFrame.Bounds())
	copy(output.Pix, fullFrame.Pix) // Fast copy entire buffer
	
	// Resize and paste (simplified nearest neighbor for speed)
	genWidth := generated.Bounds().Dx()
	genHeight := generated.Bounds().Dy()
	targetWidth := x2 - x1
	targetHeight := y2 - y1
	
	genPix := generated.Pix
	outPix := output.Pix
	
	for y := 0; y < targetHeight; y++ {
		for x := 0; x < targetWidth; x++ {
			// Source coordinates
			srcX := (x * genWidth) / targetWidth
			srcY := (y * genHeight) / targetHeight
			
			if srcX < genWidth && srcY < genHeight {
				// Direct pixel copy
				srcIdx := (srcY*genWidth + srcX) * 4
				dstIdx := ((y1+y)*output.Bounds().Dx() + (x1+x)) * 4
				
				outPix[dstIdx+0] = genPix[srcIdx+0]
				outPix[dstIdx+1] = genPix[srcIdx+1]
				outPix[dstIdx+2] = genPix[srcIdx+2]
				outPix[dstIdx+3] = 255
			}
		}
	}
	
	return output
}

func saveJPEGFast(img *image.RGBA, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	return jpeg.Encode(file, img, &jpeg.Options{Quality: 95})
}

func reshapeAudioFeatures(features []float32, output []float32) {
	// Tile 512 features to fill 8192 (32*16*16)
	for i := 0; i < len(output); i++ {
		output[i] = features[i%512]
	}
}

// Close releases resources
func (g *OptimizedGenerator) Close() error {
	if g.audioEncoderPool != nil {
		g.audioEncoderPool.Close()
	}
	if g.generatorPool != nil {
		g.generatorPool.Close()
	}
	return nil
}

