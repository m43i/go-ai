package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/m43i/go-ai/core"
)

// Embed creates one embedding vector for params.Input.
func (a *Adapter) Embed(ctx context.Context, params *core.EmbedParams) (*core.EmbedResult, error) {
	if err := a.validate(); err != nil {
		return nil, err
	}

	request, expectedCount, err := embeddingRequestFromSingle(a.Model, params)
	if err != nil {
		return nil, err
	}

	response, err := a.postEmbed(ctx, &request)
	if err != nil {
		return nil, err
	}

	vectors, err := orderedEmbeddingVectors(response.Embeddings, expectedCount)
	if err != nil {
		return nil, err
	}

	return &core.EmbedResult{
		Embedding: vectors[0],
		Usage:     toCoreEmbedUsage(response),
	}, nil
}

// EmbedMany creates embedding vectors for params.Inputs.
func (a *Adapter) EmbedMany(ctx context.Context, params *core.EmbedManyParams) (*core.EmbedManyResult, error) {
	if err := a.validate(); err != nil {
		return nil, err
	}

	request, expectedCount, err := embeddingRequestFromMany(a.Model, params)
	if err != nil {
		return nil, err
	}

	response, err := a.postEmbed(ctx, &request)
	if err != nil {
		return nil, err
	}

	vectors, err := orderedEmbeddingVectors(response.Embeddings, expectedCount)
	if err != nil {
		return nil, err
	}

	return &core.EmbedManyResult{
		Embeddings: vectors,
		Usage:      toCoreEmbedUsage(response),
	}, nil
}

func embeddingRequestFromSingle(model string, params *core.EmbedParams) (embedRequest, int, error) {
	if params == nil {
		return embedRequest{}, 0, errors.New("ollama: embed params are required")
	}

	input := strings.TrimSpace(params.Input)
	if input == "" {
		return embedRequest{}, 0, errors.New("ollama: embed input is required")
	}

	if params.Dimensions != nil && *params.Dimensions <= 0 {
		return embedRequest{}, 0, errors.New("ollama: embed dimensions must be greater than zero")
	}

	return embedRequest{
		Model:      model,
		Input:      input,
		Dimensions: params.Dimensions,
	}, 1, nil
}

func embeddingRequestFromMany(model string, params *core.EmbedManyParams) (embedRequest, int, error) {
	if params == nil {
		return embedRequest{}, 0, errors.New("ollama: embed many params are required")
	}
	if len(params.Inputs) == 0 {
		return embedRequest{}, 0, errors.New("ollama: embed many inputs are required")
	}

	inputs := make([]string, 0, len(params.Inputs))
	for i, input := range params.Inputs {
		trimmed := strings.TrimSpace(input)
		if trimmed == "" {
			return embedRequest{}, 0, fmt.Errorf("ollama: embed many input at index %d is empty", i)
		}
		inputs = append(inputs, trimmed)
	}

	if params.Dimensions != nil && *params.Dimensions <= 0 {
		return embedRequest{}, 0, errors.New("ollama: embed many dimensions must be greater than zero")
	}

	return embedRequest{
		Model:      model,
		Input:      inputs,
		Dimensions: params.Dimensions,
	}, len(inputs), nil
}

func (a *Adapter) postEmbed(ctx context.Context, request *embedRequest) (*embedResponse, error) {
	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("ollama: marshal embed request: %w", err)
	}

	url := strings.TrimRight(a.baseURL(), "/") + "/api/embed"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama: build embed request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	if strings.TrimSpace(a.APIKey) != "" {
		httpReq.Header.Set("Authorization", "Bearer "+strings.TrimSpace(a.APIKey))
	}

	httpResp, err := a.client().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama: embed request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode >= http.StatusBadRequest {
		return nil, decodeAPIError(httpResp)
	}

	var response embedResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("ollama: decode embed response: %w", err)
	}

	return &response, nil
}

func orderedEmbeddingVectors(data [][]float64, expectedCount int) ([][]float64, error) {
	if expectedCount <= 0 {
		return nil, errors.New("ollama: expected embedding count must be greater than zero")
	}
	if len(data) == 0 {
		return nil, errors.New("ollama: embeddings response did not include any vectors")
	}
	if len(data) != expectedCount {
		return nil, fmt.Errorf("ollama: embeddings response count mismatch: expected %d, got %d", expectedCount, len(data))
	}

	out := make([][]float64, expectedCount)
	for i := range data {
		out[i] = append([]float64(nil), data[i]...)
	}

	return out, nil
}
