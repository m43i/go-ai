package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/m43i/go-ai/core"
)

// Chat sends a non-streaming chat completion request to OpenAI.
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

	for range maxLoopCount {
		request := requestTemplate
		request.Messages = messages

		response, err := a.postChatCompletions(ctx, &request)
		if err != nil {
			return nil, err
		}

		if len(response.Choices) == 0 {
			return nil, errors.New("openai: empty response choices")
		}

		choice := response.Choices[0]
		assistant := choice.Message

		reasoning := parseAssistantChoiceReasoning(choice)
		if reasoning == "" && len(response.RawChoices) > 0 {
			rawReasoning, rawErr := parseAssistantChoiceRawReasoning(response.RawChoices[0])
			if rawErr != nil {
				return nil, fmt.Errorf("openai: decode raw choice reasoning: %w", rawErr)
			}
			reasoning = rawReasoning
		}
		reasoningParts = appendReasoningPart(reasoningParts, reasoning)

		if len(assistant.ToolCalls) == 0 {
			text, err := parseAssistantChoice(choice)
			if err != nil {
				return nil, err
			}
			if strings.TrimSpace(text) == "" && len(response.RawChoices) > 0 {
				rawText, rawErr := parseAssistantChoiceRaw(response.RawChoices[0])
				if rawErr != nil {
					return nil, fmt.Errorf("openai: decode raw choice: %w", rawErr)
				}
				text = rawText
			}

			conversation = append(conversation, core.TextMessagePart{Role: core.RoleAssistant, Content: text})
			return &core.ChatResult{
				Text:         text,
				Reasoning:    joinReasoningParts(reasoningParts),
				Messages:     append([]core.MessageUnion(nil), conversation...),
				ToolCalls:    nil,
				FinishReason: nonEmpty(choice.FinishReason, "stop"),
				Usage:        toCoreUsage(response.Usage),
			}, nil
		}

		messages = append(messages, chatMessage{Role: "assistant", ToolCalls: assistant.ToolCalls})

		coreCalls, err := toCoreToolCalls(assistant.ToolCalls)
		if err != nil {
			return nil, err
		}
		conversation = append(conversation, core.ToolCallMessagePart{Role: core.RoleToolCall, ToolCalls: coreCalls})

		pendingClientCalls := make([]core.ToolCall, 0)

		for idx, call := range assistant.ToolCalls {
			if serverTool, ok := serverTools[call.Function.Name]; ok {
				result, callErr := serverTool.Handler(coreCalls[idx].Arguments)
				if callErr != nil {
					result = "tool_error: " + callErr.Error()
				}

				messages = append(messages, chatMessage{
					Role:       "tool",
					ToolCallID: call.ID,
					Content:    result,
				})
				conversation = append(conversation, core.ToolResultMessagePart{
					Role:       core.RoleToolResult,
					ToolCallID: call.ID,
					Name:       call.Function.Name,
					Content:    result,
				})
				continue
			}

			if _, ok := clientTools[call.Function.Name]; ok {
				pendingClientCalls = append(pendingClientCalls, coreCalls[idx])
				continue
			}

			return nil, fmt.Errorf("openai: tool %q was requested but not registered", call.Function.Name)
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
	}

	return nil, fmt.Errorf("openai: reached max tool loop count (%d)", maxLoopCount)
}

// ChatStream sends a streaming chat completion request to OpenAI.
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

		url := strings.TrimRight(a.baseURL(), "/") + "/chat/completions"
		body, err := json.Marshal(request)
		if err != nil {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("openai: marshal stream request: %v", err)}
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("openai: build stream request: %v", err)}
			return
		}

		httpReq.Header.Set("Authorization", "Bearer "+a.APIKey)
		httpReq.Header.Set("Content-Type", "application/json")

		httpResp, err := a.client().Do(httpReq)
		if err != nil {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("openai: stream request failed: %v", err)}
			return
		}
		defer httpResp.Body.Close()

		if httpResp.StatusCode >= http.StatusBadRequest {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: decodeAPIError(httpResp).Error()}
			return
		}

		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

		var content strings.Builder
		reasoningParts := make([]string, 0, 4)
		finishReason := ""
		var usage *core.Usage

		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || strings.HasPrefix(line, ":") || !strings.HasPrefix(line, "data:") {
				continue
			}

			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if payload == "[DONE]" {
				out <- core.StreamChunk{
					Type:         core.StreamChunkDone,
					FinishReason: nonEmpty(finishReason, "stop"),
					Reasoning:    joinReasoningParts(reasoningParts),
					Usage:        usage,
				}
				return
			}

			var event streamEvent
			if err := json.Unmarshal([]byte(payload), &event); err != nil {
				out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("openai: decode stream event: %v", err)}
				return
			}

			var rawEvent struct {
				Choices []json.RawMessage `json:"choices"`
			}
			_ = json.Unmarshal([]byte(payload), &rawEvent)

			if event.Usage != nil {
				usage = toCoreUsage(event.Usage)
			}

			for idx, choice := range event.Choices {
				if choice.FinishReason != "" {
					finishReason = choice.FinishReason
				}

				reasoning := parseStreamChoiceReasoning(choice)
				if reasoning == "" && idx < len(rawEvent.Choices) {
					rawReasoning, rawErr := parseStreamChoiceRawReasoning(rawEvent.Choices[idx])
					if rawErr != nil {
						out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("openai: decode raw stream choice reasoning: %v", rawErr)}
						return
					}
					reasoning = rawReasoning
				}
				before := len(reasoningParts)
				reasoningParts = appendReasoningPart(reasoningParts, reasoning)
				if len(reasoningParts) > before {
					out <- core.StreamChunk{
						Type:      core.StreamChunkReasoning,
						Role:      core.RoleAssistant,
						Delta:     strings.TrimSpace(reasoning),
						Reasoning: joinReasoningParts(reasoningParts),
					}
				}

				deltaText, err := parseStreamChoiceText(choice)
				if err != nil {
					out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("openai: decode stream delta: %v", err)}
					return
				}
				if deltaText == "" && idx < len(rawEvent.Choices) {
					rawText, rawErr := parseStreamChoiceRaw(rawEvent.Choices[idx])
					if rawErr != nil {
						out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("openai: decode raw stream choice: %v", rawErr)}
						return
					}
					deltaText = rawText
				}

				if deltaText == "" {
					continue
				}

				content.WriteString(deltaText)
				out <- core.StreamChunk{
					Type:    core.StreamChunkContent,
					Role:    core.RoleAssistant,
					Delta:   deltaText,
					Content: content.String(),
				}
			}
		}

		if err := scanner.Err(); err != nil {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("openai: stream read failed: %v", err)}
			return
		}

		out <- core.StreamChunk{
			Type:         core.StreamChunkDone,
			FinishReason: nonEmpty(finishReason, "stop"),
			Reasoning:    joinReasoningParts(reasoningParts),
			Usage:        usage,
		}
	}()

	return out, nil
}

func (a *Adapter) buildRequestTemplate(params *core.ChatParams) (chatCompletionRequest, []chatMessage, map[string]core.ServerTool, map[string]struct{}, int, error) {
	messages, err := toChatMessages(params)
	if err != nil {
		return chatCompletionRequest{}, nil, nil, nil, 0, err
	}

	tools, serverTools, clientTools, err := toChatTools(params)
	if err != nil {
		return chatCompletionRequest{}, nil, nil, nil, 0, err
	}

	request := chatCompletionRequest{
		Model:               a.Model,
		Tools:               tools,
		MaxCompletionTokens: maxTokens(params),
		Temperature:         temperature(params),
		ReasoningEffort:     reasoningEffort(params),
	}

	if len(tools) > 0 {
		request.ToolChoice = "auto"
	}

	if params != nil && params.Output != nil {
		request.ResponseFormat = params.Output
	}

	return request, messages, serverTools, clientTools, maxLoops(params, len(serverTools) > 0), nil
}

func (a *Adapter) postChatCompletions(ctx context.Context, request *chatCompletionRequest) (*chatCompletionResponse, error) {
	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	url := strings.TrimRight(a.baseURL(), "/") + "/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: build request: %w", err)
	}

	httpReq.Header.Set("Authorization", "Bearer "+a.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := a.client().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode >= http.StatusBadRequest {
		return nil, decodeAPIError(httpResp)
	}

	bodyBytes, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: read response body: %w", err)
	}

	var response chatCompletionResponse
	if err := json.Unmarshal(bodyBytes, &response); err != nil {
		return nil, fmt.Errorf("openai: decode response: %w", err)
	}

	var rawEnvelope struct {
		Choices []json.RawMessage `json:"choices"`
	}
	if err := json.Unmarshal(bodyBytes, &rawEnvelope); err == nil {
		response.RawChoices = rawEnvelope.Choices
	}

	return &response, nil
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

	reasoningTokens := int64(0)
	if in.CompletionTokensDetails != nil {
		reasoningTokens = in.CompletionTokensDetails.ReasoningTokens
		addDetail("completion_audio_tokens", in.CompletionTokensDetails.AudioTokens)
		addDetail("accepted_prediction_tokens", in.CompletionTokensDetails.AcceptedPredictionTokens)
		addDetail("rejected_prediction_tokens", in.CompletionTokensDetails.RejectedPredictionTokens)
	}

	if in.PromptTokensDetails != nil {
		addDetail("cached_prompt_tokens", in.PromptTokensDetails.CachedTokens)
		addDetail("prompt_audio_tokens", in.PromptTokensDetails.AudioTokens)
	}

	return &core.Usage{
		PromptTokens:     in.PromptTokens,
		CompletionTokens: in.CompletionTokens,
		TotalTokens:      in.TotalTokens,
		ReasoningTokens:  reasoningTokens,
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
