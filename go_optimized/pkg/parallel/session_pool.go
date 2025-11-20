package parallel

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// SessionPool manages multiple ONNX Runtime sessions for parallel inference
type SessionPool struct {
	sessions []*ort.DynamicAdvancedSession
	pool     chan *ort.DynamicAdvancedSession
	size     int
}

// NewSessionPool creates a pool of ONNX sessions
func NewSessionPool(modelPath string, inputNames, outputNames []string, poolSize int) (*SessionPool, error) {
	fmt.Printf("Creating session pool: %d sessions for %s\n", poolSize, modelPath)
	
	sessions := make([]*ort.DynamicAdvancedSession, poolSize)
	pool := make(chan *ort.DynamicAdvancedSession, poolSize)
	
	// Create multiple sessions (one per worker)
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	
	// Set threads per session
	options.SetIntraOpNumThreads(1) // Each session uses 1 thread
	
	for i := 0; i < poolSize; i++ {
		session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, options)
		if err != nil {
			// Clean up already created sessions
			for j := 0; j < i; j++ {
				sessions[j].Destroy()
			}
			options.Destroy()
			return nil, fmt.Errorf("failed to create session %d: %w", i, err)
		}
		sessions[i] = session
		pool <- session
	}
	
	options.Destroy()
	
	fmt.Printf("  ✓ Created %d parallel sessions (TRUE parallel inference!)\n", poolSize)
	
	return &SessionPool{
		sessions: sessions,
		pool:     pool,
		size:     poolSize,
	}, nil
}

// Get retrieves a session from the pool (blocks if all busy)
func (sp *SessionPool) Get() *ort.DynamicAdvancedSession {
	return <-sp.pool
}

// Put returns a session to the pool
func (sp *SessionPool) Put(session *ort.DynamicAdvancedSession) {
	sp.pool <- session
}

// Close destroys all sessions
func (sp *SessionPool) Close() error {
	// Drain the pool
	for i := 0; i < sp.size; i++ {
		session := <-sp.pool
		session.Destroy()
	}
	close(sp.pool)
	return nil
}

// Size returns the pool size
func (sp *SessionPool) Size() int {
	return sp.size
}

