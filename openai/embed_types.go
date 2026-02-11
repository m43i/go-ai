package openai

type embeddingRequest struct {
	Model      string `json:"model"`
	Input      any    `json:"input"`
	Dimensions *int64 `json:"dimensions,omitempty"`
}

type embeddingResponse struct {
	Data  []embeddingVector `json:"data"`
	Usage *embeddingUsage   `json:"usage,omitempty"`
}

type embeddingVector struct {
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

type embeddingUsage struct {
	PromptTokens int64 `json:"prompt_tokens"`
	TotalTokens  int64 `json:"total_tokens"`
}
