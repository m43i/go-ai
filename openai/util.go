package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

func parseToolArguments(raw string) (any, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return map[string]any{}, nil
	}

	var out any
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return nil, err
	}

	return out, nil
}

func parseAssistantContent(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return "", nil
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil
	}

	var parts []map[string]any
	if err := json.Unmarshal(raw, &parts); err == nil {
		return extractTextFromParts(parts), nil
	}

	var part map[string]any
	if err := json.Unmarshal(raw, &part); err == nil {
		return extractTextFromPart(part), nil
	}

	return "", fmt.Errorf("openai: unsupported assistant content payload: %s", string(raw))
}

func parseAssistantMessage(message chatResponseMessage) (string, error) {
	text, err := parseAssistantContent(message.Content)
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(text) != "" {
		return text, nil
	}

	if strings.TrimSpace(message.Refusal) != "" {
		return message.Refusal, nil
	}

	return "", nil
}

func parseAssistantChoice(choice chatChoice) (string, error) {
	text, err := parseAssistantMessage(choice.Message)
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(text) != "" {
		return text, nil
	}

	if strings.TrimSpace(choice.Text) != "" {
		return choice.Text, nil
	}
	if strings.TrimSpace(choice.DeltaText) != "" {
		return choice.DeltaText, nil
	}

	return "", nil
}

func parseAssistantChoiceRaw(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}

	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", err
	}

	return extractTextFromAny(value), nil
}

func parseAssistantMessageReasoning(message chatResponseMessage) string {
	if strings.TrimSpace(message.ReasoningContent) != "" {
		return strings.TrimSpace(message.ReasoningContent)
	}
	if len(message.Content) == 0 {
		return ""
	}

	var value any
	if err := json.Unmarshal(message.Content, &value); err != nil {
		return ""
	}

	return extractReasoningFromAny(value)
}

func parseAssistantChoiceReasoning(choice chatChoice) string {
	if reasoning := parseAssistantMessageReasoning(choice.Message); reasoning != "" {
		return reasoning
	}
	return strings.TrimSpace(choice.Reasoning)
}

func parseAssistantChoiceRawReasoning(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}

	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", err
	}

	return extractReasoningFromAny(value), nil
}

func parseStreamDeltaText(delta streamDelta) (string, error) {
	text, err := parseAssistantContent(delta.Content)
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(text) != "" {
		return text, nil
	}

	if strings.TrimSpace(delta.Refusal) != "" {
		return delta.Refusal, nil
	}
	if strings.TrimSpace(delta.Text) != "" {
		return delta.Text, nil
	}

	return "", nil
}

func parseStreamChoiceText(choice streamChoice) (string, error) {
	text, err := parseStreamDeltaText(choice.Delta)
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(text) != "" {
		return text, nil
	}

	if strings.TrimSpace(choice.Text) != "" {
		return choice.Text, nil
	}

	return "", nil
}

func parseStreamChoiceRaw(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}

	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", err
	}

	return extractTextFromAny(value), nil
}

func parseStreamDeltaReasoning(delta streamDelta) string {
	if strings.TrimSpace(delta.ReasoningContent) != "" {
		return strings.TrimSpace(delta.ReasoningContent)
	}
	if len(delta.Content) == 0 {
		return ""
	}

	var value any
	if err := json.Unmarshal(delta.Content, &value); err != nil {
		return ""
	}

	return extractReasoningFromAny(value)
}

func parseStreamChoiceReasoning(choice streamChoice) string {
	if reasoning := parseStreamDeltaReasoning(choice.Delta); reasoning != "" {
		return reasoning
	}
	return strings.TrimSpace(choice.Reasoning)
}

func parseStreamChoiceRawReasoning(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}

	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", err
	}

	return extractReasoningFromAny(value), nil
}

func extractTextFromParts(parts []map[string]any) string {
	if len(parts) == 0 {
		return ""
	}

	var builder strings.Builder
	for _, part := range parts {
		builder.WriteString(extractTextFromPart(part))
	}

	return builder.String()
}

func extractTextFromPart(part map[string]any) string {
	if len(part) == 0 {
		return ""
	}

	partType := strings.ToLower(strings.TrimSpace(stringValue(part["type"])))

	if partType == "text" || partType == "output_text" || partType == "refusal" || partType == "reasoning" {
		if text := stringValue(part["text"]); text != "" {
			return text
		}
		if text := stringValue(part["refusal"]); text != "" {
			return text
		}
		if text := stringValue(part["reasoning_content"]); text != "" {
			return text
		}
		if text := stringValue(part["content"]); text != "" {
			return text
		}
	}

	if text := stringValue(part["text"]); text != "" {
		return text
	}
	if text := stringValue(part["content"]); text != "" {
		return text
	}

	if nested, ok := part["content"].([]any); ok {
		var builder strings.Builder
		for _, item := range nested {
			if nestedPart, ok := item.(map[string]any); ok {
				builder.WriteString(extractTextFromPart(nestedPart))
			}
		}
		return builder.String()
	}

	return ""
}

func stringValue(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	case map[string]any:
		if text := stringValue(typed["text"]); text != "" {
			return text
		}
		if value := stringValue(typed["value"]); value != "" {
			return value
		}
	}

	return ""
}

func extractTextFromAny(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		if strings.TrimSpace(typed) == "" {
			return ""
		}
		return typed
	case map[string]any:
		priorityKeys := []string{
			"text",
			"output_text",
			"refusal",
			"content",
			"message",
			"delta",
			"value",
			"choices",
		}

		for _, key := range priorityKeys {
			if candidate, ok := typed[key]; ok {
				if text := extractTextFromAny(candidate); text != "" {
					return text
				}
			}
		}

		ignoredKeys := map[string]struct{}{
			"role":              {},
			"type":              {},
			"id":                {},
			"index":             {},
			"finish_reason":     {},
			"logprobs":          {},
			"model":             {},
			"object":            {},
			"created":           {},
			"tool_calls":        {},
			"function":          {},
			"name":              {},
			"reasoning":         {},
			"reasoning_content": {},
			"thinking":          {},
			"reasoning_text":    {},
			"thinking_text":     {},
		}

		for key, candidate := range typed {
			if _, ignored := ignoredKeys[key]; ignored {
				continue
			}
			if text := extractTextFromAny(candidate); text != "" {
				return text
			}
		}

	case []any:
		var builder strings.Builder
		for _, item := range typed {
			builder.WriteString(extractTextFromAny(item))
		}
		return builder.String()
	}

	return ""
}

func extractReasoningFromAny(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""

	case map[string]any:
		keys := []string{"reasoning_content", "reasoning", "thinking", "reasoning_text", "thinking_text"}
		for _, key := range keys {
			if raw, ok := typed[key]; ok {
				if reasoning := extractTextFromAny(raw); reasoning != "" {
					return strings.TrimSpace(reasoning)
				}
			}
		}

		for _, key := range []string{"message", "delta", "content", "choices"} {
			if raw, ok := typed[key]; ok {
				if reasoning := extractReasoningFromAny(raw); reasoning != "" {
					return strings.TrimSpace(reasoning)
				}
			}
		}

	case []any:
		parts := make([]string, 0)
		for _, item := range typed {
			if reasoning := extractReasoningFromAny(item); reasoning != "" {
				parts = append(parts, reasoning)
			}
		}
		if len(parts) > 0 {
			return strings.TrimSpace(strings.Join(parts, "\n"))
		}
	}

	return ""
}

func decodeAPIError(resp *http.Response) error {
	body, readErr := io.ReadAll(io.LimitReader(resp.Body, 2*1024*1024))
	if readErr != nil {
		return fmt.Errorf("openai: API status %d and failed to read error body: %w", resp.StatusCode, readErr)
	}

	var envelope struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    any    `json:"code"`
		} `json:"error"`
	}

	if err := json.Unmarshal(body, &envelope); err == nil && envelope.Error.Message != "" {
		if envelope.Error.Type != "" || envelope.Error.Code != nil {
			return fmt.Errorf("openai: API error (%s, %v): %s", envelope.Error.Type, envelope.Error.Code, envelope.Error.Message)
		}
		return fmt.Errorf("openai: API error: %s", envelope.Error.Message)
	}

	text := strings.TrimSpace(string(body))
	if text == "" {
		text = http.StatusText(resp.StatusCode)
	}

	return fmt.Errorf("openai: API status %d: %s", resp.StatusCode, text)
}
