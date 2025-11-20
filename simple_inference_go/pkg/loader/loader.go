package loader

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	_ "image/png"
	"os"
)

// CropRect represents a crop rectangle
type CropRect struct {
	Rect []int `json:"rect"` // [x1, y1, x2, y2]
}

// LoadImage loads a JPEG image
func LoadImage(path string) (image.Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open image: %w", err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	return img, nil
}

// SaveImage saves an image as JPEG
func SaveImage(path string, img image.Image) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	err = jpeg.Encode(file, img, &jpeg.Options{Quality: 95})
	if err != nil {
		return fmt.Errorf("failed to encode image: %w", err)
	}

	return nil
}

// ImageToTensor converts an image to a float32 tensor in CHW format (BGR)
// Input: image.Image (loaded from JPEG - in RGB)
// Output: []float32 with shape (3, height, width) in BGR format, normalized to [0, 1]
func ImageToTensor(img image.Image, normalize bool) []float32 {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	tensor := make([]float32, 3*height*width)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()

			// Convert from uint32 (0-65535) to float32 (0-255)
			rVal := float32(r >> 8)
			gVal := float32(g >> 8)
			bVal := float32(b >> 8)

			if normalize {
				rVal /= 255.0
				gVal /= 255.0
				bVal /= 255.0
			}

			// CHW format in BGR order (swap R and B)
			tensor[0*height*width+y*width+x] = bVal  // B
			tensor[1*height*width+y*width+x] = gVal  // G
			tensor[2*height*width+y*width+x] = rVal  // R
		}
	}

	return tensor
}

// TensorToImage converts a tensor back to an image
// Input: []float32 with shape (3, height, width) in BGR format, values 0-255
// Output: image.Image in RGB format
func TensorToImage(tensor []float32, width, height int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// Tensor is in BGR format, convert back to RGB
			b := uint8(tensor[0*height*width+y*width+x])  // B
			g := uint8(tensor[1*height*width+y*width+x])  // G
			r := uint8(tensor[2*height*width+y*width+x])  // R

			img.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	return img
}

// LoadCropRectangles loads the crop rectangles JSON
func LoadCropRectangles(path string) (map[string]CropRect, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open crop rectangles: %w", err)
	}
	defer file.Close()

	var rects map[string]CropRect
	err = json.NewDecoder(file).Decode(&rects)
	if err != nil {
		return nil, fmt.Errorf("failed to decode crop rectangles: %w", err)
	}

	return rects, nil
}

// PasteIntoFrame pastes a generated region into a full frame
func PasteIntoFrame(fullFrame image.Image, generated image.Image, rect []int) image.Image {
	// rect is [x1, y1, x2, y2]
	x1, y1, x2, y2 := rect[0], rect[1], rect[2], rect[3]

	// Create a new RGBA image for the output
	bounds := fullFrame.Bounds()
	output := image.NewRGBA(bounds)

	// Copy the full frame first
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			output.Set(x, y, fullFrame.At(x, y))
		}
	}

	// Paste the generated region
	genBounds := generated.Bounds()
	genWidth := genBounds.Dx()
	genHeight := genBounds.Dy()

	// Calculate scaling factors
	targetWidth := x2 - x1
	targetHeight := y2 - y1

	for y := 0; y < targetHeight; y++ {
		for x := 0; x < targetWidth; x++ {
			// Map to source coordinates (simple nearest neighbor)
			srcX := (x * genWidth) / targetWidth
			srcY := (y * genHeight) / targetHeight

			if srcX < genWidth && srcY < genHeight {
				color := generated.At(srcX+genBounds.Min.X, srcY+genBounds.Min.Y)
				output.Set(x1+x, y1+y, color)
			}
		}
	}

	return output
}
