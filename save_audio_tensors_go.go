package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
)

// Quick program to save Go's audio tensors for first 5 frames for debugging

func main() {
	// This would need to be integrated into the Go code
	// For now, let's modify simple_inference_go to save tensors
	
	fmt.Println("Add this to simple_inference_go/pkg/compositor/compositor.go:")
	fmt.Println(`
// After reshaping audio, save for debugging
if frameIdx <= 5 {
	debugPath := filepath.Join(outputDir, "..", fmt.Sprintf("debug_audio_go_frame%d.bin", frameIdx))
	file, _ := os.Create(debugPath)
	binary.Write(file, binary.LittleEndian, audioTensor)
	file.Close()
	fmt.Printf("  DEBUG: Saved audio tensor for frame %d\n", frameIdx)
}
`)
}

