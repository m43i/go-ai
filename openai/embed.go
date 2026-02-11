package openai

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

	response, err := a.postEmbeddings(ctx, &request)
	if err != nil {
		return nil, err
	}

	vectors, err := orderedEmbeddingVectors(response.Data, expectedCount)
	if err != nil {
		return nil, err
	}

	return &core.EmbedResult{
		Embedding: vectors[0],
		Usage:     toCoreEmbeddingUsage(response.Usage),
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

	response, err := a.postEmbeddings(ctx, &request)
	if err != nil {
		return nil, err
	}

	vectors, err := orderedEmbeddingVectors(response.Data, expectedCount)
	if err != nil {
		return nil, err
	}

	return &core.EmbedManyResult{
		Embeddings: vectors,
		Usage:      toCoreEmbeddingUsage(response.Usage),
	}, nil
}

func embeddingRequestFromSingle(model string, params *core.EmbedParams) (embeddingRequest, int, error) {
	if params == nil {
		return embeddingRequest{}, 0, errors.New("openai: embed params are required")
	}

	input := strings.TrimSpace(params.Input)
	if input == "" {
		return embeddingRequest{}, 0, errors.New("openai: embed input is required")
	}

	if params.Dimensions != nil && *params.Dimensions <= 0 {
		return embeddingRequest{}, 0, errors.New("openai: embed dimensions must be greater than zero")
	}

	return embeddingRequest{
		Model:      model,
		Input:      input,
		Dimensions: params.Dimensions,
	}, 1, nil
}

func embeddingRequestFromMany(model string, params *core.EmbedManyParams) (embeddingRequest, int, error) {
	if params == nil {
		return embeddingRequest{}, 0, errors.New("openai: embed many params are required")
	}
	if len(params.Inputs) == 0 {
		return embeddingRequest{}, 0, errors.New("openai: embed many inputs are required")
	}

	inputs := make([]string, 0, len(params.Inputs))
	for i, input := range params.Inputs {
		trimmed := strings.TrimSpace(input)
		if trimmed == "" {
			return embeddingRequest{}, 0, fmt.Errorf("openai: embed many input at index %d is empty", i)
		}
		inputs = append(inputs, trimmed)
	}

	if params.Dimensions != nil && *params.Dimensions <= 0 {
		return embeddingRequest{}, 0, errors.New("openai: embed many dimensions must be greater than zero")
	}

	return embeddingRequest{
		Model:      model,
		Input:      inputs,
		Dimensions: params.Dimensions,
	}, len(inputs), nil
}

func (a *Adapter) postEmbeddings(ctx context.Context, request *embeddingRequest) (*embeddingResponse, error) {
	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal embeddings request: %w", err)
	}

	url := strings.TrimRight(a.baseURL(), "/") + "/embeddings"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: build embeddings request: %w", err)
	}

	httpReq.Header.Set("Authorization", "Bearer "+a.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := a.client().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: embeddings request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode >= http.StatusBadRequest {
		return nil, decodeAPIError(httpResp)
	}

	var response embeddingResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("openai: decode embeddings response: %w", err)
	}

	return &response, nil
}

func orderedEmbeddingVectors(data []embeddingVector, expectedCount int) ([][]float64, error) {
	if expectedCount <= 0 {
		return nil, errors.New("openai: expected embedding count must be greater than zero")
	}
	if len(data) == 0 {
		return nil, errors.New("openai: embeddings response did not include any vectors")
	}

	out := make([][]float64, expectedCount)
	seen := make([]bool, expectedCount)

	for _, vector := range data {
		if vector.Index < 0 || vector.Index >= expectedCount {
			return nil, fmt.Errorf("openai: embeddings response index %d out of range", vector.Index)
		}
		if seen[vector.Index] {
			return nil, fmt.Errorf("openai: embeddings response contains duplicate index %d", vector.Index)
		}

		seen[vector.Index] = true
		out[vector.Index] = append([]float64(nil), vector.Embedding...)
	}

	for i, ok := range seen {
		if !ok {
			return nil, fmt.Errorf("openai: embeddings response is missing index %d", i)
		}
	}

	return out, nil
}

func toCoreEmbeddingUsage(in *embeddingUsage) *core.Usage {
	if in == nil {
		return nil
	}

	totalTokens := in.TotalTokens
	if totalTokens == 0 {
		totalTokens = in.PromptTokens
	}

	return &core.Usage{
		PromptTokens:     in.PromptTokens,
		CompletionTokens: 0,
		TotalTokens:      totalTokens,
	}
}
