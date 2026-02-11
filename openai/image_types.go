package openai

type imageGenerationResponse struct {
	ID      string                `json:"id,omitempty"`
	Model   string                `json:"model,omitempty"`
	Created int64                 `json:"created,omitempty"`
	Data    []imageData           `json:"data"`
	Usage   *imageGenerationUsage `json:"usage,omitempty"`
}

type imageData struct {
	B64JSON       string `json:"b64_json,omitempty"`
	URL           string `json:"url,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}

type imageGenerationUsage struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
	TotalTokens  int64 `json:"total_tokens"`
}
