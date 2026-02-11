package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/m43i/go-ai/core"
)

var imageGenerationCounter uint64

var imageRequestReservedKeys = map[string]struct{}{
	"model":  {},
	"prompt": {},
	"n":      {},
	"size":   {},
}

// GenerateImage creates images with the configured OpenAI image model.
func (a *Adapter) GenerateImage(ctx context.Context, params *core.ImageParams) (*core.ImageResult, error) {
	if err := a.validate(); err != nil {
		return nil, err
	}

	request, err := imageGenerationRequest(a.Model, params)
	if err != nil {
		return nil, err
	}

	response, err := a.postImageGeneration(ctx, request)
	if err != nil {
		return nil, err
	}

	if len(response.Data) == 0 {
		return nil, errors.New("openai: image generation response did not include any images")
	}

	images := make([]core.GeneratedImage, 0, len(response.Data))
	for _, image := range response.Data {
		images = append(images, core.GeneratedImage{
			B64JSON:       strings.TrimSpace(image.B64JSON),
			URL:           strings.TrimSpace(image.URL),
			RevisedPrompt: strings.TrimSpace(image.RevisedPrompt),
		})
	}

	resultModel := strings.TrimSpace(response.Model)
	if resultModel == "" {
		resultModel = strings.TrimSpace(a.Model)
	}

	return &core.ImageResult{
		ID:     imageGenerationID(response),
		Model:  resultModel,
		Images: images,
		Usage:  toCoreImageUsage(response.Usage),
	}, nil
}

func imageGenerationRequest(model string, params *core.ImageParams) (map[string]any, error) {
	if params == nil {
		return nil, errors.New("openai: image params are required")
	}

	model = strings.TrimSpace(model)
	if model == "" {
		return nil, errors.New("openai: model is required")
	}

	prompt := strings.TrimSpace(params.Prompt)
	if prompt == "" {
		return nil, errors.New("openai: image prompt is required")
	}

	numberOfImages := int64(1)
	if params.NumberOfImages != nil {
		numberOfImages = *params.NumberOfImages
	}
	if numberOfImages < 1 {
		return nil, fmt.Errorf("openai: number of images must be at least 1; requested: %d", numberOfImages)
	}

	modelOptions, err := normalizedImageModelOptions(params.ModelOptions)
	if err != nil {
		return nil, err
	}

	request := map[string]any{
		"model":  model,
		"prompt": prompt,
		"n":      numberOfImages,
	}

	size := strings.TrimSpace(params.Size)
	if size != "" {
		request["size"] = size
	}

	for key, value := range modelOptions {
		request[key] = value
	}

	return request, nil
}

func normalizedImageModelOptions(modelOptions map[string]any) (map[string]any, error) {
	if len(modelOptions) == 0 {
		return nil, nil
	}

	out := make(map[string]any, len(modelOptions))
	for key, value := range modelOptions {
		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}

		normalizedKey := normalizeImageModelOptionKey(key)
		if _, exists := imageRequestReservedKeys[normalizedKey]; exists {
			return nil, fmt.Errorf("openai: model option %q conflicts with top-level image parameters", key)
		}
		if _, exists := out[normalizedKey]; exists {
			return nil, fmt.Errorf("openai: duplicate image model option key %q", normalizedKey)
		}

		out[normalizedKey] = value
	}

	return out, nil
}

func normalizeImageModelOptionKey(key string) string {
	switch key {
	case "outputFormat":
		return "output_format"
	case "outputCompression":
		return "output_compression"
	case "responseFormat":
		return "response_format"
	case "partialImages":
		return "partial_images"
	default:
		return key
	}
}

func (a *Adapter) postImageGeneration(ctx context.Context, request map[string]any) (*imageGenerationResponse, error) {
	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal image generation request: %w", err)
	}

	url := strings.TrimRight(a.baseURL(), "/") + "/images/generations"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: build image generation request: %w", err)
	}

	httpReq.Header.Set("Authorization", "Bearer "+a.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := a.client().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: image generation request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode >= http.StatusBadRequest {
		return nil, decodeAPIError(httpResp)
	}

	var response imageGenerationResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("openai: decode image generation response: %w", err)
	}

	return &response, nil
}

func imageGenerationID(response *imageGenerationResponse) string {
	if response != nil {
		if id := strings.TrimSpace(response.ID); id != "" {
			return id
		}
		if response.Created > 0 {
			return fmt.Sprintf("img_%d", response.Created)
		}
	}

	counter := atomic.AddUint64(&imageGenerationCounter, 1)
	return fmt.Sprintf("img_%d_%d", time.Now().UnixNano(), counter)
}

func toCoreImageUsage(in *imageGenerationUsage) *core.ImageUsage {
	if in == nil {
		return nil
	}

	totalTokens := in.TotalTokens
	if totalTokens == 0 {
		totalTokens = in.InputTokens + in.OutputTokens
	}

	return &core.ImageUsage{
		InputTokens:  in.InputTokens,
		OutputTokens: in.OutputTokens,
		TotalTokens:  totalTokens,
	}
}
