package imageproc

import (
	"fmt"
	"image"
	"image/color"
	"os"

	"github.com/disintegration/imaging"
	"gocv.io/x/gocv"
)

// Landmark represents a facial landmark point
type Landmark struct {
	X int
	Y int
}

// CropCoords represents the coordinates of a crop region
type CropCoords struct {
	XMin int
	YMin int
	XMax int
	YMax int
}

// ImageProcessor handles all image processing operations
type ImageProcessor struct{}

// NewImageProcessor creates a new image processor
func NewImageProcessor() *ImageProcessor {
	return &ImageProcessor{}
}

// LoadImage loads an image from disk
func (p *ImageProcessor) LoadImage(path string) (gocv.Mat, error) {
	img := gocv.IMRead(path, gocv.IMReadColor)
	if img.Empty() {
		return gocv.Mat{}, fmt.Errorf("failed to load image: %s", path)
	}
	return img, nil
}

// LoadLandmarks loads facial landmarks from a .lms file
func (p *ImageProcessor) LoadLandmarks(path string) ([]Landmark, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var landmarks []Landmark
	for {
		var x, y int
		_, err := fmt.Fscanf(file, "%d %d\n", &x, &y)
		if err != nil {
			break
		}
		landmarks = append(landmarks, Landmark{X: x, Y: y})
	}

	if len(landmarks) == 0 {
		return nil, fmt.Errorf("no landmarks found in file: %s", path)
	}

	return landmarks, nil
}

// GetCropRegion calculates the crop region based on facial landmarks
// Uses landmarks 1, 31, and 52 (0-indexed)
func (p *ImageProcessor) GetCropRegion(landmarks []Landmark) CropCoords {
	if len(landmarks) < 53 {
		panic("not enough landmarks")
	}

	xmin := landmarks[1].X
	ymin := landmarks[52].Y
	xmax := landmarks[31].X
	width := xmax - xmin
	ymax := ymin + width

	return CropCoords{
		XMin: xmin,
		YMin: ymin,
		XMax: xmax,
		YMax: ymax,
	}
}

// CropFaceRegion crops the face region from an image based on landmarks
func (p *ImageProcessor) CropFaceRegion(img gocv.Mat, landmarks []Landmark) (gocv.Mat, CropCoords) {
	coords := p.GetCropRegion(landmarks)

	// Create rectangle for cropping
	rect := image.Rect(coords.XMin, coords.YMin, coords.XMax, coords.YMax)

	// Crop the region
	cropped := img.Region(rect)

	return cropped, coords
}

// ResizeImage resizes an image using cubic interpolation (matches cv2.INTER_CUBIC)
func (p *ImageProcessor) ResizeImage(img gocv.Mat, width, height int) gocv.Mat {
	resized := gocv.NewMat()
	gocv.Resize(img, &resized, image.Point{X: width, Y: height}, 0, 0, gocv.InterpolationCubic)
	return resized
}

// CreateMaskedRegion creates a masked version with lower face blacked out
func (p *ImageProcessor) CreateMaskedRegion(img gocv.Mat) gocv.Mat {
	masked := img.Clone()

	// Draw black rectangle on lower face region
	// Rectangle coordinates: (5, 5) to (310, 305)
	rect := image.Rect(5, 5, 310, 305)
	gocv.Rectangle(&masked, rect, color.RGBA{0, 0, 0, 255}, -1)

	return masked
}

// PrepareInputTensors prepares input tensors for the U-Net model
// Returns a 6-channel concatenated tensor (original + masked)
func (p *ImageProcessor) PrepareInputTensors(img gocv.Mat) ([]float32, error) {
	// Create masked version
	masked := p.CreateMaskedRegion(img)
	defer masked.Close()

	// Convert to float32 and normalize
	imgFloat := gocv.NewMat()
	defer imgFloat.Close()
	img.ConvertTo(&imgFloat, gocv.MatTypeCV32F)
	imgFloat.DivideFloat(255.0)

	maskedFloat := gocv.NewMat()
	defer maskedFloat.Close()
	masked.ConvertTo(&maskedFloat, gocv.MatTypeCV32F)
	maskedFloat.DivideFloat(255.0)

	// Convert to CHW format and concatenate
	// Shape: (6, 320, 320)
	height := img.Rows()
	width := img.Cols()
	channels := 3

	tensor := make([]float32, 6*height*width)

	// Copy original image (BGR -> RGB and HWC -> CHW)
	for c := 0; c < channels; c++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				// BGR to RGB conversion (reverse channel order)
				srcChannel := 2 - c
				val := imgFloat.GetVecfAt(y, x)[srcChannel]
				tensor[c*height*width+y*width+x] = val
			}
		}
	}

	// Copy masked image
	for c := 0; c < channels; c++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				srcChannel := 2 - c
				val := maskedFloat.GetVecfAt(y, x)[srcChannel]
				tensor[(c+3)*height*width+y*width+x] = val
			}
		}
	}

	return tensor, nil
}

// PasteGeneratedRegion pastes the generated face region back into the full frame
func (p *ImageProcessor) PasteGeneratedRegion(
	fullFrame gocv.Mat,
	generatedRegion gocv.Mat,
	coords CropCoords,
	originalCropHeight, originalCropWidth int,
) gocv.Mat {
	// Create 328x328 canvas
	canvas := gocv.NewMatWithSize(328, 328, gocv.MatTypeCV8UC3)
	defer canvas.Close()
	canvas.SetTo(gocv.NewScalar(0, 0, 0, 0))

	// Paste generated region in center [4:324, 4:324]
	roi := canvas.Region(image.Rect(4, 4, 324, 324))
	generatedRegion.CopyTo(&roi)
	roi.Close()

	// Resize back to original crop size
	resized := gocv.NewMat()
	gocv.Resize(canvas, &resized, image.Point{X: originalCropWidth, Y: originalCropHeight}, 0, 0, gocv.InterpolationCubic)
	defer resized.Close()

	// Create output frame
	outputFrame := fullFrame.Clone()

	// Paste back into full frame
	roi2 := outputFrame.Region(image.Rect(coords.XMin, coords.YMin, coords.XMax, coords.YMax))
	resized.CopyTo(&roi2)
	roi2.Close()

	return outputFrame
}

// TensorToMat converts a float32 tensor to a gocv.Mat
// Tensor shape: (3, 320, 320) in RGB format
// Output: Mat in BGR format (320x320x3)
func (p *ImageProcessor) TensorToMat(tensor []float32, height, width int) gocv.Mat {
	mat := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8UC3)

	// Convert from CHW RGB to HWC BGR
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// RGB to BGR conversion
			r := uint8(tensor[0*height*width+y*width+x])
			g := uint8(tensor[1*height*width+y*width+x])
			b := uint8(tensor[2*height*width+y*width+x])

			// Set BGR value
			mat.SetUCharAt(y, x*3+0, b)
			mat.SetUCharAt(y, x*3+1, g)
			mat.SetUCharAt(y, x*3+2, r)
		}
	}

	return mat
}

