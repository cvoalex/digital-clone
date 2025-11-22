package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/alexanderrusich/go_optimized/pkg/parallel"
)

func main() {
	// Flags
	sandersDir := flag.String("sanders", "../../model/sanders_full_onnx", "Sanders directory")
	audioFile := flag.String("audio", "", "Audio WAV file (default: sanders/aud.wav)")
	outputDir := flag.String("output", "../../comparison_results/go_optimized_output/frames", "Output directory")
	numFrames := flag.Int("frames", 250, "Number of frames")
	batchSize := flag.Int("batch", 10, "Batch size for parallel processing")
	
	flag.Parse()
	
	// Set audio path
	audioPath := *audioFile
	if audioPath == "" {
		audioPath = fmt.Sprintf("%s/aud.wav", *sandersDir)
	}
	
	// Set GOMAXPROCS to use all cores
	numCPU := runtime.NumCPU()
	runtime.GOMAXPROCS(numCPU)
	
	fmt.Println("============================================================")
	fmt.Println("Optimized Go Inference - Parallel + Memory Pools")
	fmt.Println("============================================================")
	fmt.Printf("Sanders: %s\n", *sandersDir)
	fmt.Printf("Audio: %s\n", audioPath)
	fmt.Printf("Output: %s\n", *outputDir)
	fmt.Printf("Frames: %d\n", *numFrames)
	fmt.Printf("Batch size: %d\n", *batchSize)
	fmt.Printf("CPU cores: %d\n", numCPU)
	fmt.Println("============================================================")
	fmt.Println("Optimizations:")
	fmt.Println("  ✓ Parallel processing with goroutines")
	fmt.Println("  ✓ Memory pooling (zero allocation)")
	fmt.Println("  ✓ Batch processing")
	fmt.Println("  ✓ Direct pixel buffer access")
	fmt.Println("  ✓ Multi-threaded ONNX Runtime")
	fmt.Println("============================================================")
	
	totalStart := time.Now()
	
	// Create optimized generator
	fmt.Println("\n[1/3] Initializing (parallel workers + memory pools)...")
	gen, err := parallel.NewOptimizedGenerator(*sandersDir, *batchSize)
	if err != nil {
		log.Fatalf("Failed to create generator: %v", err)
	}
	defer gen.Close()
	
	fmt.Println("✓ Optimized generator ready")
	
	// Process audio
	fmt.Println("\n[2/3] Processing audio...")
	audioStart := time.Now()
	audioFeatures, err := gen.ProcessAudioParallel(audioPath)
	if err != nil {
		log.Fatalf("Failed to process audio: %v", err)
	}
	audioDuration := time.Since(audioStart)
	fmt.Printf("✓ Audio processed in %.2fs\n", audioDuration.Seconds())
	
	// Limit frames
	if *numFrames > len(audioFeatures) {
		*numFrames = len(audioFeatures)
	}
	
	// Generate frames
	fmt.Println("\n[3/3] Generating frames (parallel + optimized)...")
	genStart := time.Now()
	err = gen.GenerateFramesOptimized(audioFeatures, *numFrames, *outputDir)
	if err != nil {
		log.Fatalf("Failed to generate frames: %v", err)
	}
	genDuration := time.Since(genStart)
	
	totalDuration := time.Since(totalStart)
	
	fmt.Println("\n============================================================")
	fmt.Println("Performance Results")
	fmt.Println("============================================================")
	fmt.Printf("Audio processing: %.2fs\n", audioDuration.Seconds())
	fmt.Printf("Frame generation: %.2fs\n", genDuration.Seconds())
	fmt.Printf("Total time: %.2fs\n", totalDuration.Seconds())
	fmt.Printf("Frames per second: %.1f FPS\n", float64(*numFrames)/genDuration.Seconds())
	fmt.Printf("Overall FPS: %.1f FPS\n", float64(*numFrames)/totalDuration.Seconds())
	fmt.Println("============================================================")
	fmt.Println("\nOptimizations used:")
	fmt.Printf("  • %d parallel workers\n", numCPU)
	fmt.Printf("  • Batch size: %d\n", *batchSize)
	fmt.Println("  • Memory pooling (zero allocation)")
	fmt.Println("  • Direct pixel buffer access")
	fmt.Println("============================================================")
	fmt.Println("\nTo create video:")
	fmt.Printf("  ffmpeg -framerate 25 -i %s/frame_%%05d.jpg \\\n", *outputDir)
	fmt.Printf("    -i %s \\\n", audioPath)
	fmt.Printf("    -vframes %d -shortest \\\n", *numFrames)
	fmt.Printf("    -c:v libx264 -c:a aac -crf 20 \\\n")
	fmt.Printf("    go_optimized.mp4 -y\n")
	fmt.Println("\n✓ Complete!")
}
