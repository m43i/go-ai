package core

type EmbedParams struct {
	Input      string
	Dimensions *int64
}

type EmbedResult struct {
	Embedding []float64
	Usage     *Usage
}

type EmbedManyParams struct {
	Inputs     []string
	Dimensions *int64
}

type EmbedManyResult struct {
	Embeddings [][]float64
	Usage      *Usage
}
