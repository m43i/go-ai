package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"mime/multipart"
	"net/http"
	"strings"

	"github.com/m43i/go-ai/core"
)

var transcriptionReservedKeys = map[string]struct{}{
	"model":    {},
	"file":     {},
	"language": {},
}

// Transcribe converts audio to text using the configured OpenAI model.
//
// The OpenAI transcription API requires multipart/form-data. Audio bytes and
// filename are sent as the "file" field; all other parameters are sent as
// form fields alongside it.
func (a *Adapter) Transcribe(ctx context.Context, params *core.TranscriptionParams) (*core.TranscriptionResult, error) {
	if err := a.validate(); err != nil {
		return nil, err
	}

	body, contentType, err := buildTranscriptionForm(a.Model, params)
	if err != nil {
		return nil, err
	}

	response, err := a.postTranscription(ctx, body, contentType)
	if err != nil {
		return nil, err
	}

	return toCoreTranscriptionResult(response), nil
}

func buildTranscriptionForm(model string, params *core.TranscriptionParams) (*bytes.Buffer, string, error) {
	if params == nil {
		return nil, "", errors.New("openai: transcription params are required")
	}
	if len(params.Audio) == 0 {
		return nil, "", errors.New("openai: transcription audio data is required")
	}

	filename := strings.TrimSpace(params.Filename)
	if filename == "" {
		return nil, "", errors.New("openai: transcription filename is required")
	}

	model = strings.TrimSpace(model)
	if model == "" {
		return nil, "", errors.New("openai: model is required")
	}

	modelOptions, err := normalizedTranscriptionModelOptions(params.ModelOptions)
	if err != nil {
		return nil, "", err
	}

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	if err := writer.WriteField("model", model); err != nil {
		return nil, "", fmt.Errorf("openai: write model field: %w", err)
	}

	language := strings.TrimSpace(params.Language)
	if language != "" {
		if err := writer.WriteField("language", language); err != nil {
			return nil, "", fmt.Errorf("openai: write language field: %w", err)
		}
	}

	for key, value := range modelOptions {
		stringValue, err := modelOptionToString(value)
		if err != nil {
			return nil, "", fmt.Errorf("openai: model option %q: %w", key, err)
		}
		if err := writer.WriteField(key, stringValue); err != nil {
			return nil, "", fmt.Errorf("openai: write model option %q: %w", key, err)
		}
	}

	filePart, err := writer.CreateFormFile("file", filename)
	if err != nil {
		return nil, "", fmt.Errorf("openai: create file form field: %w", err)
	}
	if _, err := filePart.Write(params.Audio); err != nil {
		return nil, "", fmt.Errorf("openai: write audio data: %w", err)
	}

	if err := writer.Close(); err != nil {
		return nil, "", fmt.Errorf("openai: close multipart writer: %w", err)
	}

	return &buf, writer.FormDataContentType(), nil
}

func normalizedTranscriptionModelOptions(modelOptions map[string]any) (map[string]any, error) {
	if len(modelOptions) == 0 {
		return nil, nil
	}

	out := make(map[string]any, len(modelOptions))
	for key, value := range modelOptions {
		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}

		normalizedKey := normalizeTranscriptionModelOptionKey(key)
		if _, exists := transcriptionReservedKeys[normalizedKey]; exists {
			return nil, fmt.Errorf("openai: model option %q conflicts with top-level transcription parameters", key)
		}
		if _, exists := out[normalizedKey]; exists {
			return nil, fmt.Errorf("openai: duplicate transcription model option key %q", normalizedKey)
		}

		out[normalizedKey] = value
	}

	return out, nil
}

func normalizeTranscriptionModelOptionKey(key string) string {
	switch key {
	case "responseFormat":
		return "response_format"
	case "timestampGranularities":
		return "timestamp_granularities[]"
	default:
		return key
	}
}

func modelOptionToString(value any) (string, error) {
	switch v := value.(type) {
	case string:
		return v, nil
	case float64:
		return fmt.Sprintf("%g", v), nil
	case float32:
		return fmt.Sprintf("%g", v), nil
	case int:
		return fmt.Sprintf("%d", v), nil
	case int64:
		return fmt.Sprintf("%d", v), nil
	case bool:
		if v {
			return "true", nil
		}
		return "false", nil
	case []string:
		// For array-type fields like include/timestamp_granularities,
		// caller should add multiple fields. Here we join as comma-separated.
		return strings.Join(v, ","), nil
	default:
		b, err := json.Marshal(value)
		if err != nil {
			return "", fmt.Errorf("cannot convert value of type %T to string: %w", value, err)
		}
		return string(b), nil
	}
}

func (a *Adapter) postTranscription(ctx context.Context, body *bytes.Buffer, contentType string) (*transcriptionResponse, error) {
	url := strings.TrimRight(a.baseURL(), "/") + "/audio/transcriptions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
	if err != nil {
		return nil, fmt.Errorf("openai: build transcription request: %w", err)
	}

	httpReq.Header.Set("Authorization", "Bearer "+a.APIKey)
	httpReq.Header.Set("Content-Type", contentType)

	httpResp, err := a.client().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: transcription request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode >= http.StatusBadRequest {
		return nil, decodeAPIError(httpResp)
	}

	var response transcriptionResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("openai: decode transcription response: %w", err)
	}

	return &response, nil
}

func toCoreTranscriptionResult(resp *transcriptionResponse) *core.TranscriptionResult {
	if resp == nil {
		return &core.TranscriptionResult{}
	}

	result := &core.TranscriptionResult{
		Text:     resp.Text,
		Language: resp.Language,
		Duration: resp.Duration,
	}

	if len(resp.Segments) > 0 {
		result.Segments = make([]core.TranscriptionSegment, 0, len(resp.Segments))
		for _, seg := range resp.Segments {
			coreSegment := core.TranscriptionSegment{
				Start: seg.Start,
				End:   seg.End,
				Text:  seg.Text,
			}

			if len(seg.Words) > 0 {
				coreSegment.Words = toCoreTranscriptionWords(seg.Words)
			}

			result.Segments = append(result.Segments, coreSegment)
		}
	}

	// Some response formats return words at the top level instead of nested in segments.
	if len(resp.Words) > 0 && len(result.Segments) == 0 {
		result.Segments = []core.TranscriptionSegment{
			{
				Start: 0,
				End:   resp.Duration,
				Text:  resp.Text,
				Words: toCoreTranscriptionWords(resp.Words),
			},
		}
	}

	return result
}

func toCoreTranscriptionWords(words []transcriptionWord) []core.TranscriptionWord {
	out := make([]core.TranscriptionWord, 0, len(words))
	for _, w := range words {
		out = append(out, core.TranscriptionWord{
			Word:  w.Word,
			Start: w.Start,
			End:   w.End,
		})
	}
	return out
}
