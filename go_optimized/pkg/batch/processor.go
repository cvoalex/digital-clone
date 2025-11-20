package batch

import (
	"fmt"
	"image"
	"sync"

	"github.com/alexanderrusich/go_optimized/pkg/pool"
)

// FrameBatch represents a batch of frames to process
type FrameBatch struct {
	StartIdx int
	EndIdx   int
	Frames   []int
}

// BatchProcessor processes frames in batches with memory pooling
type BatchProcessor struct {
	// Memory pools
	tensor6Pool   *pool.TensorPool  // 6-channel input (1*6*320*320)
	tensor3Pool   *pool.TensorPool  // 3-channel output (1*3*320*320)
	audioPool     *pool.TensorPool  // Audio features (1*32*16*16)
	image320Pool  *pool.ImagePool   // 320x320 images
	image1280Pool *pool.ImagePool   // 1280x720 images
	
	// Worker pool
	workerPool sync.Pool
	
	// Configuration
	batchSize   int
	numWorkers  int
}

// NewBatchProcessor creates a new batch processor with memory pools
func NewBatchProcessor(batchSize, numWorkers int) *BatchProcessor {
	return &BatchProcessor{
		// Pre-allocate memory pools
		tensor6Pool:   pool.NewTensorPool(1 * 6 * 320 * 320),
		tensor3Pool:   pool.NewTensorPool(1 * 3 * 320 * 320),
		audioPool:     pool.NewTensorPool(1 * 32 * 16 * 16),
		image320Pool:  pool.NewImagePool(320, 320),
		image1280Pool: pool.NewImagePool(1280, 720),
		
		batchSize:  batchSize,
		numWorkers: numWorkers,
	}
}

// GetTensor6 gets a 6-channel tensor from pool
func (bp *BatchProcessor) GetTensor6() []float32 {
	return bp.tensor6Pool.Get()
}

// PutTensor6 returns a 6-channel tensor to pool
func (bp *BatchProcessor) PutTensor6(t []float32) {
	bp.tensor6Pool.Put(t)
}

// GetTensor3 gets a 3-channel tensor from pool
func (bp *BatchProcessor) GetTensor3() []float32 {
	return bp.tensor3Pool.Get()
}

// PutTensor3 returns a 3-channel tensor to pool
func (bp *BatchProcessor) PutTensor3(t []float32) {
	bp.tensor3Pool.Put(t)
}

// GetAudioTensor gets an audio tensor from pool
func (bp *BatchProcessor) GetAudioTensor() []float32 {
	return bp.audioPool.Get()
}

// PutAudioTensor returns an audio tensor to pool
func (bp *BatchProcessor) PutAudioTensor(t []float32) {
	bp.audioPool.Put(t)
}

// GetImage320 gets a 320x320 image from pool
func (bp *BatchProcessor) GetImage320() *image.RGBA {
	return bp.image320Pool.Get()
}

// PutImage320 returns a 320x320 image to pool
func (bp *BatchProcessor) PutImage320(img *image.RGBA) {
	bp.image320Pool.Put(img)
}

// GetImage1280 gets a 1280x720 image from pool
func (bp *BatchProcessor) GetImage1280() *image.RGBA {
	return bp.image1280Pool.Get()
}

// PutImage1280 returns a 1280x720 image to pool
func (bp *BatchProcessor) PutImage1280(img *image.RGBA) {
	bp.image1280Pool.Put(img)
}

// CreateBatches splits frame indices into batches
func (bp *BatchProcessor) CreateBatches(totalFrames int) []FrameBatch {
	var batches []FrameBatch
	
	for start := 0; start < totalFrames; start += bp.batchSize {
		end := start + bp.batchSize
		if end > totalFrames {
			end = totalFrames
		}
		
		frames := make([]int, end-start)
		for i := start; i < end; i++ {
			frames[i-start] = i + 1 // Frame indices are 1-based
		}
		
		batches = append(batches, FrameBatch{
			StartIdx: start,
			EndIdx:   end,
			Frames:   frames,
		})
	}
	
	return batches
}

// ProcessBatchParallel processes a batch of frames in parallel
func (bp *BatchProcessor) ProcessBatchParallel(
	batch FrameBatch,
	processFn func(frameIdx int, tensor6 []float32, tensor3 []float32, audioTensor []float32) error,
) error {
	var wg sync.WaitGroup
	errChan := make(chan error, len(batch.Frames))
	
	// Semaphore to limit concurrent workers
	sem := make(chan struct{}, bp.numWorkers)
	
	for _, frameIdx := range batch.Frames {
		wg.Add(1)
		
		go func(idx int) {
			defer wg.Done()
			
			// Acquire semaphore
			sem <- struct{}{}
			defer func() { <-sem }()
			
			// Get tensors from pool
			tensor6 := bp.GetTensor6()
			tensor3 := bp.GetTensor3()
			audioTensor := bp.GetAudioTensor()
			
			// Process frame
			err := processFn(idx, tensor6, tensor3, audioTensor)
			
			// Return tensors to pool
			bp.PutTensor6(tensor6)
			bp.PutTensor3(tensor3)
			bp.PutAudioTensor(audioTensor)
			
			if err != nil {
				errChan <- err
			}
		}(frameIdx)
	}
	
	wg.Wait()
	close(errChan)
	
	// Check for errors
	for err := range errChan {
		if err != nil {
			return err
		}
	}
	
	return nil
}

// Stats returns pool statistics
func (bp *BatchProcessor) Stats() string {
	return fmt.Sprintf("BatchProcessor: batch_size=%d, num_workers=%d", 
		bp.batchSize, bp.numWorkers)
}

