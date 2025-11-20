package mel

import (
	"fmt"
	"math"
	"math/cmplx"
	"os"

	"github.com/go-audio/wav"
	"github.com/mjibson/go-dsp/fft"
)

// Processor handles mel spectrogram generation
type Processor struct {
	SampleRate       int
	NFFT             int
	HopLength        int
	WinLength        int
	NMels            int
	Fmin             float64
	Fmax             float64
	PreemphasisCoef  float64
	RefLevelDB       float64
	MinLevelDB       float64
	MaxAbsValue      float64
	melBasis         [][]float64
}

// NewProcessor creates a new mel spectrogram processor with SyncTalk_2D parameters
func NewProcessor() *Processor {
	p := &Processor{
		SampleRate:      16000,
		NFFT:            800,
		HopLength:       200,
		WinLength:       800,
		NMels:           80,
		Fmin:            55.0,
		Fmax:            7600.0,
		PreemphasisCoef: 0.97,
		RefLevelDB:      20.0,
		MinLevelDB:      -100.0,
		MaxAbsValue:     4.0,
	}
	
	p.melBasis = p.buildMelBasis()
	
	return p
}

// LoadWAV loads a WAV file and returns the audio samples
func (p *Processor) LoadWAV(filename string) ([]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()
	
	decoder := wav.NewDecoder(file)
	if !decoder.IsValidFile() {
		return nil, fmt.Errorf("invalid WAV file")
	}
	
	// Read entire buffer
	buf, err := decoder.FullPCMBuffer()
	if err != nil {
		return nil, fmt.Errorf("failed to read PCM data: %w", err)
	}
	
	// Convert to float64 samples normalized to [-1, 1]
	numFrames := buf.NumFrames()
	samples := make([]float64, numFrames)
	intData := buf.AsIntBuffer().Data
	
	// Determine bit depth for normalization
	bitDepth := decoder.BitDepth
	var maxVal float64
	switch bitDepth {
	case 16:
		maxVal = 32768.0
	case 24:
		maxVal = 8388608.0
	case 32:
		maxVal = 2147483648.0
	default:
		maxVal = 32768.0
	}
	
	// Handle both mono and stereo - take first channel if stereo
	numChannels := int(decoder.NumChans)
	for i := 0; i < numFrames; i++ {
		dataIdx := i * numChannels  // Skip to first channel of this frame
		if dataIdx < len(intData) {
			samples[i] = float64(intData[dataIdx]) / maxVal
		}
	}
	
	// Note: Resampling to 16kHz should be done externally if needed
	// The Python implementation also expects 16kHz input
	
	return samples, nil
}

// PreEmphasis applies pre-emphasis filter to audio
func (p *Processor) PreEmphasis(audio []float64) []float64 {
	output := make([]float64, len(audio))
	output[0] = audio[0]
	
	for i := 1; i < len(audio); i++ {
		output[i] = audio[i] - p.PreemphasisCoef*audio[i-1]
	}
	
	return output
}

// STFT computes Short-Time Fourier Transform
func (p *Processor) STFT(audio []float64) [][]complex128 {
	numFrames := (len(audio)-p.WinLength)/p.HopLength + 1
	fftSize := p.NFFT / 2 + 1
	
	result := make([][]complex128, fftSize)
	for i := range result {
		result[i] = make([]complex128, numFrames)
	}
	
	// Hann window
	window := p.hannWindow(p.WinLength)
	
	for frameIdx := 0; frameIdx < numFrames; frameIdx++ {
		start := frameIdx * p.HopLength
		end := start + p.WinLength
		
		if end > len(audio) {
			break
		}
		
		// Apply window
		frame := make([]float64, p.NFFT)
		for i := 0; i < p.WinLength; i++ {
			frame[i] = audio[start+i] * window[i]
		}
		
		// FFT
		fftResult := fft.FFTReal(frame)
		
		// Store only first half (positive frequencies)
		for i := 0; i < fftSize; i++ {
			result[i][frameIdx] = fftResult[i]
		}
	}
	
	return result
}

// hannWindow creates a Hann window
func (p *Processor) hannWindow(size int) []float64 {
	window := make([]float64, size)
	for i := 0; i < size; i++ {
		window[i] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(size-1)))
	}
	return window
}

// LinearToMel converts linear spectrogram to mel scale
func (p *Processor) LinearToMel(spectrogram [][]float64) [][]float64 {
	numFrames := len(spectrogram[0])
	melSpec := make([][]float64, p.NMels)
	
	for i := range melSpec {
		melSpec[i] = make([]float64, numFrames)
	}
	
	// Matrix multiplication: melBasis × spectrogram
	for melIdx := 0; melIdx < p.NMels; melIdx++ {
		for frameIdx := 0; frameIdx < numFrames; frameIdx++ {
			sum := 0.0
			for freqIdx := 0; freqIdx < len(spectrogram); freqIdx++ {
				sum += p.melBasis[melIdx][freqIdx] * spectrogram[freqIdx][frameIdx]
			}
			melSpec[melIdx][frameIdx] = sum
		}
	}
	
	return melSpec
}

// AmpToDB converts amplitude to decibels
func (p *Processor) AmpToDB(spec [][]float64) [][]float64 {
	minLevel := math.Exp(-5.0 * math.Log(10.0))
	
	result := make([][]float64, len(spec))
	for i := range spec {
		result[i] = make([]float64, len(spec[i]))
		for j := range spec[i] {
			amp := math.Max(minLevel, spec[i][j])
			result[i][j] = 20.0 * math.Log10(amp)
		}
	}
	
	return result
}

// Normalize normalizes the spectrogram to [-4, 4]
func (p *Processor) Normalize(spec [][]float64) [][]float64 {
	result := make([][]float64, len(spec))
	
	for i := range spec {
		result[i] = make([]float64, len(spec[i]))
		for j := range spec[i] {
			val := (2.0*p.MaxAbsValue)*((spec[i][j]-p.MinLevelDB)/(-p.MinLevelDB)) - p.MaxAbsValue
			result[i][j] = math.Max(-p.MaxAbsValue, math.Min(p.MaxAbsValue, val))
		}
	}
	
	return result
}

// Process converts audio to mel spectrogram
func (p *Processor) Process(audio []float64) ([][]float64, error) {
	// 1. Pre-emphasis
	preEmphasized := p.PreEmphasis(audio)
	
	// 2. STFT
	stftResult := p.STFT(preEmphasized)
	
	// 3. Magnitude
	magnitude := make([][]float64, len(stftResult))
	for i := range stftResult {
		magnitude[i] = make([]float64, len(stftResult[i]))
		for j := range stftResult[i] {
			magnitude[i][j] = cmplx.Abs(stftResult[i][j])
		}
	}
	
	// 4. Linear to Mel
	melSpec := p.LinearToMel(magnitude)
	
	// 5. Amplitude to dB
	melDB := p.AmpToDB(melSpec)
	
	// 6. Apply reference level
	for i := range melDB {
		for j := range melDB[i] {
			melDB[i][j] -= p.RefLevelDB
		}
	}
	
	// 7. Normalize
	normalized := p.Normalize(melDB)
	
	return normalized, nil
}

// buildMelBasis builds the mel filterbank matrix
func (p *Processor) buildMelBasis() [][]float64 {
	nFreqs := p.NFFT/2 + 1
	
	// Create mel filterbank
	fftFreqs := make([]float64, nFreqs)
	for i := 0; i < nFreqs; i++ {
		fftFreqs[i] = float64(i) * float64(p.SampleRate) / float64(p.NFFT)
	}
	
	// Mel scale conversion
	melMin := p.freqToMel(p.Fmin)
	melMax := p.freqToMel(p.Fmax)
	
	melPoints := make([]float64, p.NMels+2)
	for i := range melPoints {
		melPoints[i] = melMin + (melMax-melMin)*float64(i)/float64(p.NMels+1)
	}
	
	freqPoints := make([]float64, len(melPoints))
	for i, mel := range melPoints {
		freqPoints[i] = p.melToFreq(mel)
	}
	
	// Build filterbank
	filterbank := make([][]float64, p.NMels)
	for i := range filterbank {
		filterbank[i] = make([]float64, nFreqs)
	}
	
	for melIdx := 0; melIdx < p.NMels; melIdx++ {
		leftFreq := freqPoints[melIdx]
		centerFreq := freqPoints[melIdx+1]
		rightFreq := freqPoints[melIdx+2]
		
		for freqIdx := 0; freqIdx < nFreqs; freqIdx++ {
			freq := fftFreqs[freqIdx]
			
			if freq >= leftFreq && freq <= centerFreq {
				filterbank[melIdx][freqIdx] = (freq - leftFreq) / (centerFreq - leftFreq)
			} else if freq >= centerFreq && freq <= rightFreq {
				filterbank[melIdx][freqIdx] = (rightFreq - freq) / (rightFreq - centerFreq)
			}
		}
	}
	
	// Normalize
	for i := range filterbank {
		enorm := 2.0 / (freqPoints[i+2] - freqPoints[i])
		for j := range filterbank[i] {
			filterbank[i][j] *= enorm
		}
	}
	
	return filterbank
}

// freqToMel converts frequency to mel scale
func (p *Processor) freqToMel(freq float64) float64 {
	return 2595.0 * math.Log10(1.0+freq/700.0)
}

// melToFreq converts mel scale to frequency
func (p *Processor) melToFreq(mel float64) float64 {
	return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
}

// CropAudioWindow extracts a 16-frame window for a specific video frame
func (p *Processor) CropAudioWindow(melSpec [][]float64, frameIdx int, fps int) ([][]float64, error) {
	startIdx := int(80.0 * float64(frameIdx) / float64(fps))
	endIdx := startIdx + 16
	
	nFrames := len(melSpec[0])
	
	if endIdx > nFrames {
		endIdx = nFrames
		startIdx = endIdx - 16
	}
	
	if startIdx < 0 {
		return nil, fmt.Errorf("frame index out of range")
	}
	
	// Extract window and transpose to (16, 80)
	window := make([][]float64, 16)
	for i := 0; i < 16; i++ {
		window[i] = make([]float64, p.NMels)
		for j := 0; j < p.NMels; j++ {
			window[i][j] = melSpec[j][startIdx+i]
		}
	}
	
	return window, nil
}

// GetFrameCount calculates the number of video frames for a mel spectrogram
func (p *Processor) GetFrameCount(melSpec [][]float64, fps int) int {
	nMelFrames := len(melSpec[0])
	return int((float64(nMelFrames)-16.0)/80.0*float64(fps)) + 2
}

