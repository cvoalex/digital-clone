package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/alexanderrusich/audio_pipeline_go/pkg/pipeline"
)

func main() {
	// Parse command line arguments
	audioPath := flag.String("audio", "", "Path to input audio file (WAV, 16kHz)")
	modelPath := flag.String("model", "models/audio_encoder.onnx", "Path to ONNX model")
	outputDir := flag.String("output", "output", "Output directory for results")
	fps := flag.Int("fps", 25, "Target video frame rate")
	mode := flag.String("mode", "ave", "Audio encoding mode (ave, hubert, wenet)")
	
	flag.Parse()
	
	if *audioPath == "" {
		fmt.Println("Usage: process -audio <audio_file.wav> [options]")
		flag.PrintDefaults()
		os.Exit(1)
	}
	
	// Print banner
	fmt.Println("======================================================================")
	fmt.Println("Audio Pipeline - Go Implementation")
	fmt.Println("======================================================================")
	fmt.Printf("Audio: %s\n", *audioPath)
	fmt.Printf("Model: %s\n", *modelPath)
	fmt.Printf("Output: %s\n", *outputDir)
	fmt.Printf("FPS: %d\n", *fps)
	fmt.Printf("Mode: %s\n", *mode)
	fmt.Println("======================================================================")
	fmt.Println()
	
	// Create output directory
	err := os.MkdirAll(*outputDir, 0755)
	if err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}
	
	// Create pipeline
	fmt.Println("Initializing pipeline...")
	pipe, err := pipeline.New(*modelPath, *fps, *mode)
	if err != nil {
		log.Fatalf("Failed to create pipeline: %v", err)
	}
	defer pipe.Close()
	
	fmt.Println("✓ Pipeline initialized")
	fmt.Println()
	
	// Process audio
	fmt.Println("Processing audio file...")
	result, err := pipe.ProcessAudioFile(*audioPath)
	if err != nil {
		log.Fatalf("Failed to process audio: %v", err)
	}
	
	fmt.Println("✓ Processing complete!")
	fmt.Println()
	
	// Print statistics
	fmt.Println("======================================================================")
	fmt.Println("Results:")
	fmt.Println("======================================================================")
	stats := result.GetStats()
	for key, value := range stats {
		fmt.Printf("  %s: %v\n", key, value)
	}
	fmt.Println()
	
	// Save metadata
	metadataPath := filepath.Join(*outputDir, "metadata.json")
	metadata := map[string]interface{}{
		"audio_path": *audioPath,
		"mode":       *mode,
		"fps":        *fps,
		"stats":      stats,
	}
	
	metadataJSON, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		log.Printf("Warning: Failed to create metadata: %v", err)
	} else {
		err = os.WriteFile(metadataPath, metadataJSON, 0644)
		if err != nil {
			log.Printf("Warning: Failed to save metadata: %v", err)
		} else {
			fmt.Printf("✓ Saved metadata to: %s\n", metadataPath)
		}
	}
	
	// Process and save per-frame features
	fmt.Println()
	fmt.Println("Generating per-frame features...")
	
	framesDir := filepath.Join(*outputDir, "frames")
	err = os.MkdirAll(framesDir, 0755)
	if err != nil {
		log.Fatalf("Failed to create frames directory: %v", err)
	}
	
	// Save first, middle, and last frame as examples
	framesToSave := []int{0, result.NFrames / 2, result.NFrames - 1}
	
	for _, frameIdx := range framesToSave {
		if frameIdx >= result.NFrames {
			continue
		}
		
		// Get frame features
		features, err := pipe.GetFrameFeatures(result.AudioFeatures, frameIdx)
		if err != nil {
			log.Printf("Warning: Failed to get features for frame %d: %v", frameIdx, err)
			continue
		}
		
		// Reshape for model
		reshaped, err := pipe.ReshapeForModel(features)
		if err != nil {
			log.Printf("Warning: Failed to reshape frame %d: %v", frameIdx, err)
			continue
		}
		
		// Save as binary file
		framePath := filepath.Join(framesDir, fmt.Sprintf("frame_%05d.bin", frameIdx))
		err = saveFloat32Array(framePath, reshaped)
		if err != nil {
			log.Printf("Warning: Failed to save frame %d: %v", frameIdx, err)
			continue
		}
		
		fmt.Printf("  ✓ Saved frame %d\n", frameIdx)
	}
	
	fmt.Println()
	fmt.Println("======================================================================")
	fmt.Println("✅ Complete!")
	fmt.Println("======================================================================")
	fmt.Printf("Output directory: %s\n", *outputDir)
	fmt.Println()
	fmt.Println("Generated files:")
	fmt.Println("  - metadata.json")
	fmt.Println("  - frames/frame_XXXXX.bin")
	fmt.Println()
}

// saveFloat32Array saves a float32 array to a binary file
func saveFloat32Array(filename string, data []float32) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	// Write length
	length := uint32(len(data))
	err = writeBinary(file, &length)
	if err != nil {
		return err
	}
	
	// Write data
	for _, val := range data {
		err = writeBinary(file, &val)
		if err != nil {
			return err
		}
	}
	
	return nil
}

func writeBinary(file *os.File, data interface{}) error {
	return json.NewEncoder(file).Encode(data)
}

