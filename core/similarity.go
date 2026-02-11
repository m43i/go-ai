package core

import (
	"errors"
	"fmt"
	"math"
)

// CosineSimilarity returns the cosine similarity between two vectors.
//
// Both vectors must be non-empty, have equal length, and be non-zero vectors.
func CosineSimilarity(a, b []float64) (float64, error) {
	if len(a) == 0 || len(b) == 0 {
		return 0, errors.New("cosine similarity requires non-empty vectors")
	}
	if len(a) != len(b) {
		return 0, fmt.Errorf("cosine similarity requires vectors with equal dimensions, got %d and %d", len(a), len(b))
	}

	dot := 0.0
	aNorm := 0.0
	bNorm := 0.0

	for i := range a {
		dot += a[i] * b[i]
		aNorm += a[i] * a[i]
		bNorm += b[i] * b[i]
	}

	if aNorm == 0 || bNorm == 0 {
		return 0, errors.New("cosine similarity is undefined for zero vectors")
	}

	return dot / (math.Sqrt(aNorm) * math.Sqrt(bNorm)), nil
}
