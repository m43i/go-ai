package claude

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"unicode"
)

func marshalMessageRequest(request *messageRequest) ([]byte, error) {
	body, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}
	if request == nil || len(request.ModelOptions) == 0 {
		return body, nil
	}

	var envelope map[string]any
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, err
	}
	for key, value := range request.ModelOptions {
		key = strings.TrimSpace(key)
		if key != "" && value != nil {
			envelope[jsonKey(key)] = value
		}
	}

	return json.Marshal(envelope)
}

func jsonKey(key string) string {
	switch key {
	case "maxTokens":
		return "max_tokens"
	case "topP":
		return "top_p"
	case "topK":
		return "top_k"
	case "toolChoice":
		return "tool_choice"
	case "stopSequences":
		return "stop_sequences"
	case "serviceTier":
		return "service_tier"
	case "outputConfig":
		return "output_config"
	case "cacheControl":
		return "cache_control"
	}

	if strings.Contains(key, "_") {
		return key
	}
	return camelToSnake(key)
}

func camelToSnake(value string) string {
	var builder strings.Builder
	for i, r := range value {
		if unicode.IsUpper(r) {
			if i > 0 {
				builder.WriteByte('_')
			}
			builder.WriteRune(unicode.ToLower(r))
			continue
		}
		builder.WriteRune(r)
	}
	return builder.String()
}

func extractText(content []contentBlock) string {
	var builder strings.Builder
	for _, block := range content {
		if block.Type == "text" {
			builder.WriteString(block.Text)
		}
	}
	return builder.String()
}

func extractReasoning(content []contentBlock) string {
	parts := make([]string, 0)
	for _, block := range content {
		switch block.Type {
		case "thinking", "reasoning":
			if strings.TrimSpace(block.Thinking) != "" {
				parts = append(parts, strings.TrimSpace(block.Thinking))
				continue
			}
			if strings.TrimSpace(block.Text) != "" {
				parts = append(parts, strings.TrimSpace(block.Text))
			}
		}
	}

	if len(parts) == 0 {
		return ""
	}

	return strings.TrimSpace(strings.Join(parts, "\n"))
}

func extractToolUses(content []contentBlock) []contentBlock {
	out := make([]contentBlock, 0)
	for _, block := range content {
		if block.Type == "tool_use" {
			out = append(out, block)
		}
	}
	return out
}

func toolResultBlock(toolUseID, result string) contentBlock {
	return contentBlock{
		Type:      "tool_result",
		ToolUseID: toolUseID,
		Content:   result,
	}
}

func decodeAPIError(resp *http.Response) error {
	body, readErr := io.ReadAll(io.LimitReader(resp.Body, 2*1024*1024))
	if readErr != nil {
		return fmt.Errorf("claude: API status %d and failed to read error body: %w", resp.StatusCode, readErr)
	}

	var envelope struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.Unmarshal(body, &envelope); err == nil && envelope.Error.Message != "" {
		if envelope.Error.Type != "" {
			return fmt.Errorf("claude: API error (%s): %s", envelope.Error.Type, envelope.Error.Message)
		}
		return fmt.Errorf("claude: API error: %s", envelope.Error.Message)
	}

	text := strings.TrimSpace(string(body))
	if text == "" {
		text = http.StatusText(resp.StatusCode)
	}

	return fmt.Errorf("claude: API status %d: %s", resp.StatusCode, text)
}
