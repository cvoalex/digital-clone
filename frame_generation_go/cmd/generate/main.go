package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/alexanderrusich/digital-clone/frame_generation_go/pkg/generator"
	"gocv.io/x/gocv"
)

func main() {
	// Command line flags
	modelPath := flag.String("model", "./models/unet_328.onnx", "Path to ONNX model")
	audioFeatures := flag.String("audio", "", "Path to audio features (binary format)")
	templateDir := flag.String("template", "", "Path to template directory")
	outputDir := flag.String("output", "./output/frames", "Output directory for frames")
	mode := flag.String("mode", "ave", "Audio feature mode (ave, hubert, wenet)")
	startFrame := flag.Int("start", 0, "Starting frame index")
	saveVideo := flag.Bool("video", false, "Create video from frames (requires ffmpeg)")
	videoPath := flag.String("video-path", "./output/result.mp4", "Output video path")
	audioPath := flag.String("audio-file", "", "Audio file for video")
	fps := flag.Int("fps", 25, "Frames per second")

	flag.Parse()

	// Validate inputs
	if *audioFeatures == "" || *templateDir == "" {
		fmt.Println("Usage: generate --audio <audio_features> --template <template_dir>")
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Create frame generator
	fmt.Println("Initializing frame generator...")
	gen, err := generator.NewFrameGenerator(generator.Config{
		ModelPath: *modelPath,
		Mode:      *mode,
	})
	if err != nil {
		log.Fatalf("Failed to create generator: %v", err)
	}
	defer gen.Close()

	// Load audio features
	fmt.Printf("Loading audio features from %s...\n", *audioFeatures)
	features, err := loadBinaryFeatures(*audioFeatures)
	if err != nil {
		log.Fatalf("Failed to load audio features: %v", err)
	}
	fmt.Printf("Loaded %d frames of audio features\n", len(features))

	// Set up template directories
	imgDir := filepath.Join(*templateDir, "full_body_img")
	lmsDir := filepath.Join(*templateDir, "landmarks")

	// Validate directories
	if _, err := os.Stat(imgDir); os.IsNotExist(err) {
		log.Fatalf("Image directory not found: %s", imgDir)
	}
	if _, err := os.Stat(lmsDir); os.IsNotExist(err) {
		log.Fatalf("Landmarks directory not found: %s", lmsDir)
	}

	// Generate frames
	fmt.Println("Generating frames...")
	frames, err := gen.GenerateFramesFromSequence(imgDir, lmsDir, features, *startFrame)
	if err != nil {
		log.Fatalf("Failed to generate frames: %v", err)
	}

	// Save frames
	fmt.Printf("Saving %d frames to %s...\n", len(frames), *outputDir)
	err = gen.SaveFrames(frames, *outputDir, "frame")
	if err != nil {
		log.Fatalf("Failed to save frames: %v", err)
	}

	// Create video if requested
	if *saveVideo {
		if *audioPath == "" {
			log.Fatal("Audio file required for video creation (--audio-file)")
		}

		fmt.Println("Creating video...")
		err = createVideo(frames, *videoPath, *audioPath, *fps)
		if err != nil {
			log.Fatalf("Failed to create video: %v", err)
		}
		fmt.Printf("Video saved to %s\n", *videoPath)
	}

	// Clean up frames
	for _, frame := range frames {
		frame.Close()
	}

	fmt.Println("Done!")
}

// loadBinaryFeatures loads audio features from binary format
// Expected format: JSON metadata + binary float32 data
func loadBinaryFeatures(path string) ([][]float32, error) {
	// Read metadata
	metadataPath := path + ".json"
	metadataFile, err := os.Open(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open metadata: %w", err)
	}
	defer metadataFile.Close()

	var metadata struct {
		NumFrames   int   `json:"num_frames"`
		FeatureSize int   `json:"feature_size"`
		Shape       []int `json:"shape"`
	}

	err = json.NewDecoder(metadataFile).Decode(&metadata)
	if err != nil {
		return nil, fmt.Errorf("failed to decode metadata: %w", err)
	}

	// Read binary data
	dataFile, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open data file: %w", err)
	}
	defer dataFile.Close()

	// Read all float32 values
	totalSize := metadata.NumFrames * metadata.FeatureSize
	data := make([]float32, totalSize)

	err = binary.Read(dataFile, binary.LittleEndian, data)
	if err != nil {
		return nil, fmt.Errorf("failed to read binary data: %w", err)
	}

	// Reshape to [num_frames][feature_size]
	features := make([][]float32, metadata.NumFrames)
	for i := 0; i < metadata.NumFrames; i++ {
		features[i] = data[i*metadata.FeatureSize : (i+1)*metadata.FeatureSize]
	}

	return features, nil
}

// createVideo creates a video from frames using OpenCV and ffmpeg
func createVideo(frames []gocv.Mat, outputPath string, audioPath string, fps int) error {
	if len(frames) == 0 {
		return fmt.Errorf("no frames to write")
	}

	// Create temporary video without audio
	tempPath := outputPath + ".temp.avi"

	// Get frame dimensions
	height := frames[0].Rows()
	width := frames[0].Cols()

	fmt.Printf("Creating video: %dx%d @ %d fps\n", width, height, fps)

	// Create video writer with MJPEG codec
	writer, err := gocv.VideoWriterFile(
		tempPath,
		"MJPG",
		float64(fps),
		width,
		height,
		true,
	)
	if err != nil {
		return fmt.Errorf("failed to create video writer: %w", err)
	}

	// Write frames
	fmt.Println("Writing frames to video...")
	for i, frame := range frames {
		err = writer.Write(frame)
		if err != nil {
			writer.Close()
			return fmt.Errorf("failed to write frame %d: %w", i, err)
		}
		if (i+1)%100 == 0 {
			fmt.Printf("Wrote %d/%d frames\n", i+1, len(frames))
		}
	}
	writer.Close()
	fmt.Printf("Wrote all %d frames to temporary video\n", len(frames))

	// Merge with audio using ffmpeg
	fmt.Println("Merging video with audio using ffmpeg...")
	
	cmd := exec.Command(
		"ffmpeg",
		"-i", tempPath,
		"-i", audioPath,
		"-c:v", "libx264",
		"-c:a", "aac",
		"-crf", "20",
		"-y",
		outputPath,
	)
	
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("ffmpeg output: %s\n", string(output))
		return fmt.Errorf("ffmpeg failed: %w", err)
	}

	// Clean up temporary file
	os.Remove(tempPath)
	
	fmt.Printf("Video saved to: %s\n", outputPath)

	return nil
}

