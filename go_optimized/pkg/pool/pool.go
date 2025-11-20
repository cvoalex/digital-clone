package pool

import (
	"image"
	"sync"
)

// TensorPool manages reusable float32 slices for tensors
type TensorPool struct {
	pool sync.Pool
	size int
}

// NewTensorPool creates a new tensor pool
func NewTensorPool(size int) *TensorPool {
	return &TensorPool{
		pool: sync.Pool{
			New: func() interface{} {
				return make([]float32, size)
			},
		},
		size: size,
	}
}

// Get retrieves a tensor from the pool
func (p *TensorPool) Get() []float32 {
	return p.pool.Get().([]float32)
}

// Put returns a tensor to the pool
func (p *TensorPool) Put(tensor []float32) {
	// Clear the tensor before returning
	for i := range tensor {
		tensor[i] = 0
	}
	p.pool.Put(tensor)
}

// ImagePool manages reusable RGBA images
type ImagePool struct {
	pool   sync.Pool
	width  int
	height int
}

// NewImagePool creates a new image pool
func NewImagePool(width, height int) *ImagePool {
	return &ImagePool{
		pool: sync.Pool{
			New: func() interface{} {
				return image.NewRGBA(image.Rect(0, 0, width, height))
			},
		},
		width:  width,
		height: height,
	}
}

// Get retrieves an image from the pool
func (p *ImagePool) Get() *image.RGBA {
	return p.pool.Get().(*image.RGBA)
}

// Put returns an image to the pool
func (p *ImagePool) Put(img *image.RGBA) {
	// Clear the image before returning (optional, costs performance)
	// for i := range img.Pix {
	// 	img.Pix[i] = 0
	// }
	p.pool.Put(img)
}

// BytePool manages reusable byte slices
type BytePool struct {
	pool sync.Pool
	size int
}

// NewBytePool creates a new byte pool
func NewBytePool(size int) *BytePool {
	return &BytePool{
		pool: sync.Pool{
			New: func() interface{} {
				return make([]byte, size)
			},
		},
		size: size,
	}
}

// Get retrieves a byte slice from the pool
func (p *BytePool) Get() []byte {
	return p.pool.Get().([]byte)
}

// Put returns a byte slice to the pool
func (p *BytePool) Put(buf []byte) {
	p.pool.Put(buf)
}

