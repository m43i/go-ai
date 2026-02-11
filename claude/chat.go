package claude

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/m43i/go-ai/core"
)

// Chat sends a non-streaming messages request to Claude.
//
// It supports tool calls, optional structured output schemas, and reasoning metadata.
func (a *Adapter) Chat(ctx context.Context, params *core.ChatParams) (*core.ChatResult, error) {
	if err := a.validate(); err != nil {
		return nil, err
	}

	requestTemplate, messages, serverTools, clientTools, maxLoopCount, err := a.buildRequestTemplate(params)
	if err != nil {
		return nil, err
	}

	conversation := cloneCoreMessages(params)
	reasoningParts := make([]string, 0, 4)

	for i := 0; i < maxLoopCount; i++ {
		request := requestTemplate
		request.Messages = messages

		response, err := a.postMessages(ctx, &request)
		if err != nil {
			return nil, err
		}

		reasoningParts = appendReasoningPart(reasoningParts, extractReasoning(response.Content))

		toolUses := extractToolUses(response.Content)
		if len(toolUses) == 0 {
			text := extractText(response.Content)
			conversation = append(conversation, core.TextMessagePart{Role: core.RoleAssistant, Content: text})
			return &core.ChatResult{
				Text:         text,
				Reasoning:    joinReasoningParts(reasoningParts),
				Messages:     append([]core.MessageUnion(nil), conversation...),
				ToolCalls:    nil,
				FinishReason: nonEmpty(response.StopReason, "stop"),
				Usage:        toCoreUsage(response.Usage),
			}, nil
		}

		messages = append(messages, message{Role: "assistant", Content: response.Content})

		coreCalls := toCoreToolCalls(toolUses)
		conversation = append(conversation, core.ToolCallMessagePart{Role: core.RoleToolCall, ToolCalls: coreCalls})

		resultBlocks := make([]contentBlock, 0, len(toolUses))
		pendingClientCalls := make([]core.ToolCall, 0)

		for idx, use := range toolUses {
			if serverTool, ok := serverTools[use.Name]; ok {
				result, callErr := serverTool.Handler(coreCalls[idx].Arguments)
				if callErr != nil {
					result = "tool_error: " + callErr.Error()
				}

				resultBlocks = append(resultBlocks, toolResultBlock(use.ID, result))
				conversation = append(conversation, core.ToolResultMessagePart{
					Role:       core.RoleToolResult,
					ToolCallID: use.ID,
					Name:       use.Name,
					Content:    result,
				})
				continue
			}

			if _, ok := clientTools[use.Name]; ok {
				pendingClientCalls = append(pendingClientCalls, coreCalls[idx])
				continue
			}

			return nil, fmt.Errorf("claude: tool %q was requested but not registered", use.Name)
		}

		if len(pendingClientCalls) > 0 {
			return &core.ChatResult{
				Text:         "",
				Reasoning:    joinReasoningParts(reasoningParts),
				Messages:     append([]core.MessageUnion(nil), conversation...),
				ToolCalls:    pendingClientCalls,
				FinishReason: "tool_calls",
				Usage:        toCoreUsage(response.Usage),
			}, nil
		}

		if len(resultBlocks) > 0 {
			messages = append(messages, message{Role: "user", Content: resultBlocks})
		}
	}

	return nil, fmt.Errorf("claude: reached max tool loop count (%d)", maxLoopCount)
}

// ChatStream sends a streaming messages request to Claude.
//
// When tools or structured output are configured, ChatStream emits chunks derived
// from a non-streaming Chat call to preserve consistent behavior.
func (a *Adapter) ChatStream(ctx context.Context, params *core.ChatParams) (<-chan core.StreamChunk, error) {
	if err := a.validate(); err != nil {
		return nil, err
	}

	request, messages, serverTools, clientTools, _, err := a.buildRequestTemplate(params)
	if err != nil {
		return nil, err
	}

	out := make(chan core.StreamChunk, 64)

	go func() {
		defer close(out)

		if len(serverTools) > 0 || len(clientTools) > 0 || (params != nil && params.Output != nil) {
			result, err := a.Chat(ctx, params)
			if err != nil {
				out <- core.StreamChunk{Type: core.StreamChunkError, Error: err.Error()}
				return
			}

			emitChunksFromResult(out, params, result)
			out <- core.StreamChunk{
				Type:         core.StreamChunkDone,
				FinishReason: nonEmpty(result.FinishReason, defaultFinishReason(result)),
				Reasoning:    result.Reasoning,
				Usage:        result.Usage,
			}
			return
		}

		request.Messages = messages
		request.Stream = true

		url := strings.TrimRight(a.baseURL(), "/") + "/messages"
		body, err := json.Marshal(request)
		if err != nil {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("claude: marshal stream request: %v", err)}
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("claude: build stream request: %v", err)}
			return
		}

		httpReq.Header.Set("x-api-key", a.APIKey)
		if version := a.version(); version != "" {
			httpReq.Header.Set("anthropic-version", version)
		}
		httpReq.Header.Set("content-type", "application/json")

		httpResp, err := a.client().Do(httpReq)
		if err != nil {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("claude: stream request failed: %v", err)}
			return
		}
		defer httpResp.Body.Close()

		if httpResp.StatusCode >= http.StatusBadRequest {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: decodeAPIError(httpResp).Error()}
			return
		}

		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

		content := ""
		reasoningParts := make([]string, 0, 4)
		var usage *core.Usage

		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || strings.HasPrefix(line, ":") || !strings.HasPrefix(line, "data:") {
				continue
			}

			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if payload == "" || payload == "[DONE]" {
				continue
			}

			var event streamEvent
			if err := json.Unmarshal([]byte(payload), &event); err != nil {
				out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("claude: decode stream event: %v", err)}
				return
			}

			if event.Usage != nil {
				usage = toCoreUsage(event.Usage)
			}

			if event.Type == "error" && event.Error != nil {
				out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("claude: stream error (%s): %s", event.Error.Type, event.Error.Message)}
				return
			}

			if event.Type == "content_block_delta" && event.Delta != nil {
				if event.Delta.Type == "text_delta" && event.Delta.Text != "" {
					content += event.Delta.Text
					out <- core.StreamChunk{
						Type:    core.StreamChunkContent,
						Role:    core.RoleAssistant,
						Delta:   event.Delta.Text,
						Content: content,
					}
				} else if event.Delta.Type == "thinking_delta" {
					reasoningDelta := strings.TrimSpace(nonEmpty(event.Delta.Thinking, event.Delta.Text))
					before := len(reasoningParts)
					reasoningParts = appendReasoningPart(reasoningParts, reasoningDelta)
					if len(reasoningParts) > before {
						out <- core.StreamChunk{
							Type:      core.StreamChunkReasoning,
							Role:      core.RoleAssistant,
							Delta:     reasoningDelta,
							Reasoning: joinReasoningParts(reasoningParts),
						}
					}
				}
			}

			if event.Type == "message_stop" {
				out <- core.StreamChunk{Type: core.StreamChunkDone, FinishReason: "stop", Reasoning: joinReasoningParts(reasoningParts), Usage: usage}
				return
			}
		}

		if err := scanner.Err(); err != nil {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("claude: stream read failed: %v", err)}
			return
		}

		out <- core.StreamChunk{Type: core.StreamChunkDone, FinishReason: "stop", Reasoning: joinReasoningParts(reasoningParts), Usage: usage}
	}()

	return out, nil
}

func (a *Adapter) buildRequestTemplate(params *core.ChatParams) (messageRequest, []message, map[string]core.ServerTool, map[string]struct{}, int, error) {
	messages, system, err := toMessagesAndSystem(params)
	if err != nil {
		return messageRequest{}, nil, nil, nil, 0, err
	}

	tools, serverTools, clientTools, err := toTools(params)
	if err != nil {
		return messageRequest{}, nil, nil, nil, 0, err
	}

	request := messageRequest{
		Model:       a.Model,
		System:      applyOutputInstruction(system, paramsOutput(params)),
		Tools:       tools,
		MaxTokens:   maxTokens(params),
		Temperature: temperature(params),
	}

	if len(tools) > 0 {
		request.ToolChoice = &toolChoice{Type: "auto"}
	}

	return request, messages, serverTools, clientTools, maxLoops(params, len(serverTools) > 0), nil
}

func (a *Adapter) postMessages(ctx context.Context, request *messageRequest) (*messageResponse, error) {
	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("claude: marshal request: %w", err)
	}

	url := strings.TrimRight(a.baseURL(), "/") + "/messages"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("claude: build request: %w", err)
	}

	httpReq.Header.Set("x-api-key", a.APIKey)
	if version := a.version(); version != "" {
		httpReq.Header.Set("anthropic-version", version)
	}
	httpReq.Header.Set("content-type", "application/json")

	httpResp, err := a.client().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("claude: request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode >= http.StatusBadRequest {
		return nil, decodeAPIError(httpResp)
	}

	var response messageResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("claude: decode response: %w", err)
	}

	return &response, nil
}

func paramsOutput(params *core.ChatParams) *core.Schema {
	if params == nil {
		return nil
	}
	return params.Output
}

func cloneCoreMessages(params *core.ChatParams) []core.MessageUnion {
	if params == nil || len(params.Messages) == 0 {
		return nil
	}

	out := make([]core.MessageUnion, 0, len(params.Messages)+8)
	out = append(out, params.Messages...)
	return out
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

func toCoreUsage(in *usage) *core.Usage {
	if in == nil {
		return nil
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

	addDetail("cache_creation_input_tokens", in.CacheCreationInputTokens)
	addDetail("cache_read_input_tokens", in.CacheReadInputTokens)

	return &core.Usage{
		PromptTokens:     in.InputTokens,
		CompletionTokens: in.OutputTokens,
		TotalTokens:      in.InputTokens + in.OutputTokens,
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
