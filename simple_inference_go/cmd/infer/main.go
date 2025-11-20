package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/alexanderrusich/simple_inference_go/pkg/compositor"
)

func main() {
	// Command line flags
	sandersDir := flag.String("sanders", "../../model/sanders_full_onnx", "Path to Sanders directory")
	audioFile := flag.String("audio", "", "Path to audio WAV file (if empty, uses sanders/aud.wav)")
	outputDir := flag.String("output", "../comparison_results/go_output/frames", "Output directory for generated frames")
	numFrames := flag.Int("frames", 523, "Number of frames to generate")

	flag.Parse()

	fmt.Println("============================================================")
	fmt.Println("Simple Inference - Sanders Frame Generation")
	fmt.Println("============================================================")
	fmt.Printf("Sanders directory: %s\n", *sandersDir)
	fmt.Printf("Output directory: %s\n", *outputDir)
	fmt.Printf("Number of frames: %d\n", *numFrames)
	fmt.Println("============================================================")

	// Determine audio file to use
	audioPath := *audioFile
	if audioPath == "" {
		audioPath = fmt.Sprintf("%s/aud.wav", *sandersDir)
	}

	// Paths
	modelPath := fmt.Sprintf("%s/models/generator.onnx", *sandersDir)
	audioEncoderPath := fmt.Sprintf("%s/models/audio_encoder.onnx", *sandersDir)
	cropRectsPath := fmt.Sprintf("%s/cache/crop_rectangles.json", *sandersDir)
	roisDir := fmt.Sprintf("%s/rois_320", *sandersDir)
	maskedDir := fmt.Sprintf("%s/model_inputs", *sandersDir)
	fullBodyDir := fmt.Sprintf("%s/full_body_img", *sandersDir)

	// Verify files exist
	requiredFiles := []string{modelPath, audioEncoderPath, cropRectsPath, audioPath}
	for _, file := range requiredFiles {
		if _, err := os.Stat(file); os.IsNotExist(err) {
			log.Fatalf("Required file not found: %s", file)
		}
	}

	// Verify directories exist
	requiredDirs := []string{roisDir, maskedDir, fullBodyDir}
	for _, dir := range requiredDirs {
		if _, err := os.Stat(dir); os.IsNotExist(err) {
			log.Fatalf("Required directory not found: %s", dir)
		}
	}

	fmt.Println("\n[1/4] Loading models...")

	// Create compositor
	comp, err := compositor.NewCompositor(modelPath, audioEncoderPath, cropRectsPath)
	if err != nil {
		log.Fatalf("Failed to create compositor: %v", err)
	}
	defer comp.Close()

	fmt.Println("✓ Models loaded successfully")

	fmt.Println("\n[2/4] Processing audio...")

	// Process audio file into features
	audioFeatures, err := comp.ProcessAudioFile(audioPath)
	if err != nil {
		log.Fatalf("Failed to process audio: %v", err)
	}

	fmt.Printf("✓ Generated %d audio feature frames\n", len(audioFeatures))

	// Limit to requested number of frames
	if *numFrames > len(audioFeatures) {
		*numFrames = len(audioFeatures)
	}

	fmt.Println("\n[3/4] Generating video frames...")

	// Generate frames
	err = comp.GenerateFrames(
		roisDir,
		maskedDir,
		fullBodyDir,
		audioFeatures,
		*outputDir,
		*numFrames,
	)
	if err != nil {
		log.Fatalf("Failed to generate frames: %v", err)
	}

	fmt.Println("\n[4/4] Video assembly...")
	fmt.Println("To create video, run:")
	fmt.Printf("  ffmpeg -framerate 25 -i %s/frame_%%05d.jpg \\\n", *outputDir)
	fmt.Printf("    -i %s \\\n", audioPath)
	fmt.Printf("    -c:v libx264 -c:a aac -crf 20 \\\n")
	fmt.Printf("    output_video.mp4 -y\n")

	fmt.Println("\n============================================================")
	fmt.Println("✓ Frame generation complete!")
	fmt.Println("============================================================")
}

