package core

type ImageParams struct {
	Prompt         string
	NumberOfImages *int64
	Size           string
	ModelOptions   map[string]any
}

type GeneratedImage struct {
	B64JSON       string
	URL           string
	RevisedPrompt string
}

type ImageUsage struct {
	InputTokens  int64
	OutputTokens int64
	TotalTokens  int64
}

type ImageResult struct {
	ID     string
	Model  string
	Images []GeneratedImage
	Usage  *ImageUsage
}
