package ollama

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/m43i/go-ai/core"
)

func decodeAPIError(resp *http.Response) error {
	body, readErr := io.ReadAll(io.LimitReader(resp.Body, 2*1024*1024))
	if readErr != nil {
		return fmt.Errorf("ollama: API status %d and failed to read error body: %w", resp.StatusCode, readErr)
	}

	var envelope struct {
		Error string `json:"error"`
	}

	if err := json.Unmarshal(body, &envelope); err == nil && strings.TrimSpace(envelope.Error) != "" {
		return fmt.Errorf("ollama: API error: %s", strings.TrimSpace(envelope.Error))
	}

	text := strings.TrimSpace(string(body))
	if text == "" {
		text = http.StatusText(resp.StatusCode)
	}

	return fmt.Errorf("ollama: API status %d: %s", resp.StatusCode, text)
}

func toCoreChatUsage(in *chatResponse) *core.Usage {
	if in == nil {
		return nil
	}

	return toCoreUsageWithMetrics(
		in.PromptEvalCount,
		in.EvalCount,
		in.TotalDuration,
		in.LoadDuration,
		in.PromptEvalDuration,
		in.EvalDuration,
	)
}

func toCoreEmbedUsage(in *embedResponse) *core.Usage {
	if in == nil {
		return nil
	}

	return toCoreUsageWithMetrics(
		in.PromptEvalCount,
		0,
		in.TotalDuration,
		in.LoadDuration,
		0,
		0,
	)
}

func toCoreUsageWithMetrics(promptEvalCount, evalCount, totalDuration, loadDuration, promptEvalDuration, evalDuration int64) *core.Usage {
	if promptEvalCount <= 0 && evalCount <= 0 && totalDuration <= 0 && loadDuration <= 0 && promptEvalDuration <= 0 && evalDuration <= 0 {
		return nil
	}

	totalTokens := promptEvalCount + evalCount
	if totalTokens == 0 {
		totalTokens = promptEvalCount
	}

	var details map[string]int64
	addDetail := func(key string, value int64) {
		if value <= 0 {
			return
		}
		if details == nil {
			details = make(map[string]int64)
		}
		details[key] = value
	}

	addDetail("total_duration_ns", totalDuration)
	addDetail("load_duration_ns", loadDuration)
	addDetail("prompt_eval_duration_ns", promptEvalDuration)
	addDetail("eval_duration_ns", evalDuration)

	return &core.Usage{
		PromptTokens:     promptEvalCount,
		CompletionTokens: evalCount,
		TotalTokens:      totalTokens,
		Details:          details,
	}
}

func appendReasoningPart(parts []string, reasoning string) []string {
	reasoning = strings.TrimSpace(reasoning)
	if reasoning == "" {
		return parts
	}
	if len(parts) > 0 && parts[len(parts)-1] == reasoning {
		return parts
	}
	return append(parts, reasoning)
}

func joinReasoningParts(parts []string) string {
	if len(parts) == 0 {
		return ""
	}
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

func defaultFinishReason(result *core.ChatResult) string {
	if result != nil && len(result.ToolCalls) > 0 {
		return "tool_calls"
	}
	return "stop"
}

func nonEmpty(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}

func appendStreamSegment(current, incoming string) (next string, delta string) {
	if incoming == "" {
		return current, ""
	}

	if strings.HasPrefix(incoming, current) {
		return incoming, incoming[len(current):]
	}
	if strings.HasPrefix(current, incoming) {
		return current, ""
	}

	return current + incoming, incoming
}

func emitChunksFromResult(out chan<- core.StreamChunk, params *core.ChatParams, result *core.ChatResult) {
	if result == nil {
		return
	}

	if strings.TrimSpace(result.Reasoning) != "" {
		reasoning := strings.TrimSpace(result.Reasoning)
		out <- core.StreamChunk{
			Type:      core.StreamChunkReasoning,
			Role:      core.RoleAssistant,
			Delta:     reasoning,
			Reasoning: reasoning,
		}
	}

	start := 0
	if params != nil {
		start = len(params.Messages)
	}
	if start < 0 || start > len(result.Messages) {
		start = 0
	}

	for _, message := range result.Messages[start:] {
		switch m := message.(type) {
		case core.TextMessagePart:
			if m.Role == core.RoleAssistant {
				out <- core.StreamChunk{Type: core.StreamChunkContent, Role: core.RoleAssistant, Delta: m.Content, Content: m.Content}
			}
		case *core.TextMessagePart:
			if m != nil && m.Role == core.RoleAssistant {
				out <- core.StreamChunk{Type: core.StreamChunkContent, Role: core.RoleAssistant, Delta: m.Content, Content: m.Content}
			}

		case core.ToolCallMessagePart:
			for _, call := range m.ToolCalls {
				c := call
				out <- core.StreamChunk{Type: core.StreamChunkToolCall, ToolCall: &c}
			}
		case *core.ToolCallMessagePart:
			if m != nil {
				for _, call := range m.ToolCalls {
					c := call
					out <- core.StreamChunk{Type: core.StreamChunkToolCall, ToolCall: &c}
				}
			}

		case core.ToolResultMessagePart:
			out <- core.StreamChunk{Type: core.StreamChunkToolResult, ToolCallID: m.ToolCallID, Content: m.Content}
		case *core.ToolResultMessagePart:
			if m != nil {
				out <- core.StreamChunk{Type: core.StreamChunkToolResult, ToolCallID: m.ToolCallID, Content: m.Content}
			}
		}
	}
}
