package cache

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// TensorCache caches converted image tensors to disk
type TensorCache struct {
	cacheDir string
	mu       sync.RWMutex
	hits     int
	misses   int
}

// NewTensorCache creates a tensor cache
func NewTensorCache(cacheDir string) (*TensorCache, error) {
	err := os.MkdirAll(cacheDir, 0755)
	if err != nil {
		return nil, err
	}
	
	return &TensorCache{
		cacheDir: cacheDir,
	}, nil
}

// Get retrieves or converts an image tensor
func (tc *TensorCache) Get(imagePath string, converter func() ([]float32, error)) ([]float32, error) {
	// Generate cache key from file path
	cacheKey := filepath.Base(imagePath)
	cachePath := filepath.Join(tc.cacheDir, cacheKey+".tensor")
	
	// Try to load from cache
	tc.mu.RLock()
	if data, err := tc.loadFromDisk(cachePath); err == nil {
		tc.mu.RUnlock()
		tc.mu.Lock()
		tc.hits++
		if tc.hits%100 == 0 {
			total := tc.hits + tc.misses
			fmt.Printf("  Cache: %d hits, %d misses (%.0f%% hit rate)\n", 
				tc.hits, tc.misses, float64(tc.hits)*100/float64(total))
		}
		tc.mu.Unlock()
		return data, nil
	}
	tc.mu.RUnlock()
	
	// Convert and cache
	tc.mu.Lock()
	tc.misses++
	tc.mu.Unlock()
	
	tensor, err := converter()
	if err != nil {
		return nil, err
	}
	
	// Save to disk
	tc.saveToDisk(cachePath, tensor)
	
	return tensor, nil
}

func (tc *TensorCache) loadFromDisk(path string) ([]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	
	stat, err := file.Stat()
	if err != nil {
		return nil, err
	}
	
	size := int(stat.Size()) / 4
	data := make([]float32, size)
	err = binary.Read(file, binary.LittleEndian, data)
	if err != nil {
		return nil, err
	}
	
	return data, nil
}

func (tc *TensorCache) saveToDisk(path string, data []float32) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	return binary.Write(file, binary.LittleEndian, data)
}

// Stats returns cache statistics
func (tc *TensorCache) Stats() (hits, misses int) {
	tc.mu.RLock()
	defer tc.mu.RUnlock()
	return tc.hits, tc.misses
}

