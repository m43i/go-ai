package claude

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/m43i/go-ai/core"
)

func applyOutputInstruction(system string, output *core.Schema) string {
	instruction := outputInstruction(output)
	if instruction == "" {
		return system
	}

	if strings.TrimSpace(system) == "" {
		return instruction
	}

	return strings.TrimSpace(system) + "\n\n" + instruction
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
