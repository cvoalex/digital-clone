package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"log"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"

	"github.com/go-audio/wav"
	"github.com/mjibson/go-dsp/fft"
	ort "github.com/yalue/onnxruntime_go"
)

// --- Configuration ---

const (
	SampleRate      = 16000
	NFFT            = 800
	HopLength       = 200
	WinLength       = 800
	NMels           = 80
	Fmin            = 55.0
	Fmax            = 7600.0
	PreemphasisCoef = 0.97
	RefLevelDB      = 20.0
	MinLevelDB      = -100.0
	MaxAbsValue     = 4.0
)

// --- Structs ---

type CropRect struct {
	Rect []int `json:"rect"`
}

// --- Main Pipeline ---

func main() {
	startFrame := flag.Int("start", 1, "Start frame")
	endFrame := flag.Int("end", 250, "End frame")
	step := flag.Int("step", 5, "Step size")
	audioFile := flag.String("audio", "", "Audio file (absolute path)")
	sandersDir := flag.String("sanders", "", "Path to sanders dataset (absolute path)")
	outputDir := flag.String("output", "", "Output directory (absolute path)")
	flag.Parse()

	// Convert all paths to absolute
	var err error
	audioPath := *audioFile
	if audioPath == "" {
		audioPath, _ = filepath.Abs("../demo/silence_20s.wav")
	} else if !filepath.IsAbs(audioPath) {
		audioPath, _ = filepath.Abs(audioPath)
	}

	sandersPath := *sandersDir
	if sandersPath == "" {
		sandersPath, _ = filepath.Abs("../model/sanders_full_onnx")
	} else if !filepath.IsAbs(sandersPath) {
		sandersPath, _ = filepath.Abs(sandersPath)
	}

	outPath := *outputDir
	if outPath == "" {
		outPath, _ = filepath.Abs("tests")
	} else if !filepath.IsAbs(outPath) {
		outPath, _ = filepath.Abs(outPath)
	}

	fmt.Printf("Processing Frames: %d to %d (step %d)\n", *startFrame, *endFrame, *step)
	fmt.Printf("Audio: %s\n", audioPath)
	fmt.Printf("Dataset: %s\n", sandersPath)
	fmt.Printf("Output: %s\n", outPath)

	os.MkdirAll(outPath, 0755)

	// 1. Initialize ONNX Runtime
	ort.InitializeEnvironment()
	defer ort.DestroyEnvironment()

	// 2. Load Models
	fmt.Println("Loading models...")
	encoderPath := filepath.Join(sandersPath, "models/audio_encoder.onnx")
	generatorPath := filepath.Join(sandersPath, "models/generator.onnx")

	encoderSession, err := createSession(encoderPath, []string{"mel"}, []string{"emb"})
	if err != nil {
		log.Fatalf("Error loading encoder: %v", err)
	}
	defer encoderSession.Destroy()

	generatorSession, err := createSession(generatorPath, []string{"input", "audio"}, []string{"output"})
	if err != nil {
		log.Fatalf("Error loading generator: %v", err)
	}
	defer generatorSession.Destroy()

	// 3. Process Audio (Full Pipeline) - ONCE
	fmt.Println("Processing audio...")
	melSpec, err := processAudioToMel(audioPath)
	if err != nil {
		log.Fatalf("Audio processing failed: %v", err)
	}

	// Load crop rects - ONCE
	rectsFile, _ := os.Open(filepath.Join(sandersPath, "cache/crop_rectangles.json"))
	defer rectsFile.Close()
	var rects map[string]CropRect
	json.NewDecoder(rectsFile).Decode(&rects)

	// Loop through frames
	for frameNum := *startFrame; frameNum <= *endFrame; frameNum += *step {
		fmt.Printf("Processing frame %d...\n", frameNum)
		
		// Extract window for the specific frame
		frameIdx0 := frameNum - 1 // 0-based index
		startIdx := int(80.0 * (float64(frameIdx0) / 25.0))
		endIdx := startIdx + 16

		if endIdx > len(melSpec[0]) {
			fmt.Printf("  Audio ended at frame %d (need index %d, have %d)\n", frameNum, endIdx, len(melSpec[0]))
			break
		}

		// 4. Run Audio Encoder
		melWindow := make([]float32, 1*1*80*16)
		for melIdx := 0; melIdx < 80; melIdx++ {
			for t := 0; t < 16; t++ {
				melWindow[melIdx*16+t] = float32(melSpec[melIdx][startIdx+t])
			}
		}

		features, err := runAudioEncoder(encoderSession, melWindow)
		if err != nil {
			log.Fatalf("Encoder inference failed: %v", err)
		}

		// Prepare Audio Tensor for Generator (Tile 512 -> 8192 and Reshape)
		audioInput := make([]float32, 1*32*16*16)
		for i := 0; i < len(audioInput); i++ {
			audioInput[i] = features[i%512]
		}

		// 5. Load and Prepare Images
		roiPath := filepath.Join(sandersPath, "rois_320", fmt.Sprintf("%d.jpg", frameNum))
		maskedPath := filepath.Join(sandersPath, "model_inputs", fmt.Sprintf("%d.jpg", frameNum))
		fullBodyPath := filepath.Join(sandersPath, "full_body_img", fmt.Sprintf("%d.jpg", frameNum))

		roiTensor, err := loadImageToTensor(roiPath)
		if err != nil {
			log.Printf("  Error loading images for frame %d: %v", frameNum, err)
			continue
		}
		maskedTensor, err := loadImageToTensor(maskedPath)
		if err != nil {
			log.Printf("  Error loading images for frame %d: %v", frameNum, err)
			continue
		}

		// Concatenate tensors (ROI + Masked) -> [1, 6, 320, 320]
		visualInput := make([]float32, 6*320*320)
		copy(visualInput[:3*320*320], roiTensor)
		copy(visualInput[3*320*320:], maskedTensor)

		// 6. Run Generator
		genOutput, err := runGenerator(generatorSession, visualInput, audioInput)
		if err != nil {
			log.Fatalf("Generator inference failed: %v", err)
		}

		// 7. Post-Processing
		// Convert tensor to image
		mouthImg := tensorToImage(genOutput, 320, 320)

		// Load full body for pasting
		fullBodyFile, _ := os.Open(fullBodyPath)
		fullBodyImg, _, _ := image.Decode(fullBodyFile)
		fullBodyFile.Close()
		fullBodyRGBA := imageToRGBA(fullBodyImg)

		// Get crop rect
		cropRect, ok := rects[fmt.Sprintf("%d", frameIdx0)]
		if !ok {
			log.Printf("  No crop rect for frame %d", frameIdx0)
			continue
		}

		// Paste with Bilinear Interpolation
		finalImg := pasteBilinear(fullBodyRGBA, mouthImg, cropRect.Rect)

		// 8. Save Output
		outFilePath := filepath.Join(outPath, fmt.Sprintf("frame_%05d.jpg", frameNum))
		outFile, err := os.Create(outFilePath)
		if err != nil {
			log.Fatal(err)
		}
		jpeg.Encode(outFile, finalImg, &jpeg.Options{Quality: 95})
		outFile.Close()
	}
	
	fmt.Println("✓ Processing complete!")
}

// --- Helper Functions ---

func createSession(path string, inputs, outputs []string) (*ort.DynamicAdvancedSession, error) {
	options, _ := ort.NewSessionOptions()
	defer options.Destroy()
	return ort.NewDynamicAdvancedSession(path, inputs, outputs, options)
}

// --- Audio Processing ---

func processAudioToMel(path string) ([][]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := wav.NewDecoder(file)
	buf, err := decoder.FullPCMBuffer()
	if err != nil {
		return nil, err
	}

	// Convert to float [-1, 1]
	floats := make([]float64, buf.NumFrames())
	ints := buf.AsIntBuffer().Data
	factor := 1.0 / 32768.0
	for i, v := range ints {
		floats[i] = float64(v) * factor
	}

	// Pre-emphasis
	for i := len(floats) - 1; i > 0; i-- {
		floats[i] = floats[i] - PreemphasisCoef*floats[i-1]
	}

	// STFT
	numFrames := (len(floats)-WinLength)/HopLength + 1
	melSpec := make([][]float64, NMels)
	for i := range melSpec {
		melSpec[i] = make([]float64, numFrames)
	}

	window := make([]float64, WinLength)
	for i := 0; i < WinLength; i++ {
		window[i] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(WinLength-1)))
	}

	// Build Mel Basis (simplified for brevity, would normally calculate)
	// NOTE: For a truly single file without external huge data dumps, we need to calculate this.
	// I will include the basis calculation logic here.
	melBasis := buildMelBasis(SampleRate, NFFT, NMels)

	for t := 0; t < numFrames; t++ {
		start := t * HopLength
		frame := make([]float64, NFFT)
		for i := 0; i < WinLength; i++ {
			if start+i < len(floats) {
				frame[i] = floats[start+i] * window[i]
			}
		}

		fftRes := fft.FFTReal(frame)
		
		// Magnitude & Mel & Log
		for m := 0; m < NMels; m++ {
			var energy float64
			for k := 0; k < NFFT/2+1; k++ {
				mag := cmplx.Abs(fftRes[k])
				energy += mag * melBasis[m][k]
			}
			// Amp to DB
			val := 20.0 * math.Log10(math.Max(1e-5, energy))
			val -= RefLevelDB
			// Normalize
			val = (2.0 * MaxAbsValue) * ((val - MinLevelDB) / -MinLevelDB) - MaxAbsValue
			val = math.Max(-MaxAbsValue, math.Min(MaxAbsValue, val))
			melSpec[m][t] = val
		}
	}

	return melSpec, nil
}

func buildMelBasis(sr, nfft, nMels int) [][]float64 {
	weights := make([][]float64, nMels)
	nFreqs := nfft/2 + 1
	
	fftFreqs := make([]float64, nFreqs)
	for i := 0; i < nFreqs; i++ {
		fftFreqs[i] = float64(i) * float64(sr) / float64(nfft)
	}

	minMel := 2595.0 * math.Log10(1.0+Fmin/700.0)
	maxMel := 2595.0 * math.Log10(1.0+Fmax/700.0)

	melPoints := make([]float64, nMels+2)
	for i := range melPoints {
		melPoints[i] = minMel + (maxMel-minMel)*float64(i)/float64(nMels+1)
	}

	hzPoints := make([]float64, nMels+2)
	for i, m := range melPoints {
		hzPoints[i] = 700.0 * (math.Pow(10.0, m/2595.0) - 1.0)
	}

	for i := 0; i < nMels; i++ {
		weights[i] = make([]float64, nFreqs)
		for j, f := range fftFreqs {
			if f >= hzPoints[i] && f <= hzPoints[i+1] {
				weights[i][j] = (f - hzPoints[i]) / (hzPoints[i+1] - hzPoints[i])
			} else if f > hzPoints[i+1] && f <= hzPoints[i+2] {
				weights[i][j] = (hzPoints[i+2] - f) / (hzPoints[i+2] - hzPoints[i+1])
			}
		}
		// Enorm
		enorm := 2.0 / (hzPoints[i+2] - hzPoints[i])
		for j := range weights[i] {
			weights[i][j] *= enorm
		}
	}
	return weights
}

// --- Tensor / Image ---

func loadImageToTensor(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}

	rgba := imageToRGBA(img)
	bounds := rgba.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	tensor := make([]float32, 3*w*h)
	scale := float32(1.0 / 255.0)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			off := (y*w + x) * 4
			r := float32(rgba.Pix[off+0]) * scale
			g := float32(rgba.Pix[off+1]) * scale
			b := float32(rgba.Pix[off+2]) * scale

			// BGR Planar
			tensor[0*w*h+y*w+x] = b
			tensor[1*w*h+y*w+x] = g
			tensor[2*w*h+y*w+x] = r
		}
	}
	return tensor, nil
}

func tensorToImage(tensor []float32, w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			idx := y*w + x
			// BGR Planar -> RGB Interleaved
			b := uint8(tensor[0*w*h+idx]) // * 255.0 handled in output range? No, usually raw output is 0-1
			g := uint8(tensor[1*w*h+idx])
			r := uint8(tensor[2*w*h+idx])
			
			// Wait, generator output is 0-1 sigmoid. We need to scale.
			// But in the previous code we saw result[i] *= 255.0 in runGenerator.
			// Let's assume input to this func is 0-255 scaled.
			
			img.Set(x, y, color.RGBA{r, g, b, 255})
		}
	}
	return img
}

func imageToRGBA(src image.Image) *image.RGBA {
	if dst, ok := src.(*image.RGBA); ok {
		return dst
	}
	b := src.Bounds()
	dst := image.NewRGBA(b)
	draw.Draw(dst, b, src, b.Min, draw.Src)
	return dst
}

// --- Inference Runners ---

func runAudioEncoder(session *ort.DynamicAdvancedSession, melInput []float32) ([]float32, error) {
	// Input: [1, 1, 80, 16]
	inputShape := ort.NewShape(1, 1, 80, 16)
	inputTensor, _ := ort.NewTensor(inputShape, melInput)
	defer inputTensor.Destroy()

	outputShape := ort.NewShape(1, 512)
	outputData := make([]float32, 512)
	outputTensor, _ := ort.NewTensor(outputShape, outputData)
	defer outputTensor.Destroy()

	err := session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	if err != nil {
		return nil, err
	}
	
	// Copy data out
	res := make([]float32, 512)
	copy(res, outputTensor.GetData())
	return res, nil
}

func runGenerator(session *ort.DynamicAdvancedSession, visual []float32, audio []float32) ([]float32, error) {
	visualShape := ort.NewShape(1, 6, 320, 320)
	visTensor, _ := ort.NewTensor(visualShape, visual)
	defer visTensor.Destroy()

	audioShape := ort.NewShape(1, 32, 16, 16)
	audTensor, _ := ort.NewTensor(audioShape, audio)
	defer audTensor.Destroy()

	outputShape := ort.NewShape(1, 3, 320, 320)
	outData := make([]float32, 3*320*320)
	outTensor, _ := ort.NewTensor(outputShape, outData)
	defer outTensor.Destroy()

	err := session.Run([]ort.Value{visTensor, audTensor}, []ort.Value{outTensor})
	if err != nil {
		return nil, err
	}

	raw := outTensor.GetData()
	res := make([]float32, len(raw))
	for i, v := range raw {
		val := v * 255.0
		if val < 0 { val = 0 }
		if val > 255 { val = 255 }
		res[i] = val
	}
	return res, nil
}

// --- Interpolation ---

func pasteBilinear(full, gen *image.RGBA, rect []int) *image.RGBA {
	out := image.NewRGBA(full.Bounds())
	draw.Draw(out, out.Bounds(), full, full.Bounds().Min, draw.Src)

	x1, y1, x2, y2 := rect[0], rect[1], rect[2], rect[3]
	targetW, targetH := x2-x1, y2-y1
	genW, genH := gen.Bounds().Dx(), gen.Bounds().Dy()

	for y := 0; y < targetH; y++ {
		for x := 0; x < targetW; x++ {
			// Bilinear logic
			srcX := float32(x) * float32(genW-1) / float32(targetW)
			srcY := float32(y) * float32(genH-1) / float32(targetH)

			xL, yT := int(srcX), int(srcY)
			xR, yB := xL+1, yT+1
			if xR >= genW { xR = genW - 1 }
			if yB >= genH { yB = genH - 1 }

			alphaX := srcX - float32(xL)
			alphaY := srcY - float32(yT)

			// Interpolate RGB
			offsetTL := (yT*genW + xL) * 4
			offsetTR := (yT*genW + xR) * 4
			offsetBL := (yB*genW + xL) * 4
			offsetBR := (yB*genW + xR) * 4

			var rgba [4]uint8
			for c := 0; c < 3; c++ {
				vTL := float32(gen.Pix[offsetTL+c])
				vTR := float32(gen.Pix[offsetTR+c])
				vBL := float32(gen.Pix[offsetBL+c])
				vBR := float32(gen.Pix[offsetBR+c])

				top := vTL + (vTR-vTL)*alphaX
				bottom := vBL + (vBR-vBL)*alphaX
				val := top + (bottom-top)*alphaY
				rgba[c] = uint8(val)
			}
			rgba[3] = 255

			// Set in output
			dOff := ((y1+y)*out.Stride) + (x1+x)*4
			out.Pix[dOff+0] = rgba[0]
			out.Pix[dOff+1] = rgba[1]
			out.Pix[dOff+2] = rgba[2]
			out.Pix[dOff+3] = rgba[3]
		}
	}
	return out
}

