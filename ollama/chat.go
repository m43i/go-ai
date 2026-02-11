package ollama

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

// Chat sends a non-streaming chat request to Ollama.
//
// It supports tool calls, optional structured output schemas, and thinking metadata.
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
		stream := false
		request.Stream = &stream

		response, err := a.postChat(ctx, &request)
		if err != nil {
			return nil, err
		}

		reasoningParts = appendReasoningPart(reasoningParts, response.Message.Thinking)
		assistantText := response.Message.Content

		if len(response.Message.ToolCalls) == 0 {
			conversation = append(conversation, core.TextMessagePart{Role: core.RoleAssistant, Content: assistantText})
			return &core.ChatResult{
				Text:         assistantText,
				Reasoning:    joinReasoningParts(reasoningParts),
				Messages:     append([]core.MessageUnion(nil), conversation...),
				ToolCalls:    nil,
				FinishReason: nonEmpty(response.DoneReason, "stop"),
				Usage:        toCoreChatUsage(response),
			}, nil
		}

		messages = append(messages, response.Message)

		coreCalls, err := toCoreToolCalls(response.Message.ToolCalls)
		if err != nil {
			return nil, err
		}

		if strings.TrimSpace(assistantText) != "" {
			conversation = append(conversation, core.TextMessagePart{Role: core.RoleAssistant, Content: assistantText})
		}
		conversation = append(conversation, core.ToolCallMessagePart{Role: core.RoleToolCall, ToolCalls: coreCalls})

		pendingClientCalls := make([]core.ToolCall, 0)

		for _, call := range coreCalls {
			if serverTool, ok := serverTools[call.Name]; ok {
				result, callErr := serverTool.Handler(call.Arguments)
				if callErr != nil {
					result = "tool_error: " + callErr.Error()
				}

				messages = append(messages, message{
					Role:       "tool",
					ToolCallID: call.ID,
					ToolName:   call.Name,
					Content:    result,
				})
				conversation = append(conversation, core.ToolResultMessagePart{
					Role:       core.RoleToolResult,
					ToolCallID: call.ID,
					Name:       call.Name,
					Content:    result,
				})
				continue
			}

			if _, ok := clientTools[call.Name]; ok {
				pendingClientCalls = append(pendingClientCalls, call)
				continue
			}

			return nil, fmt.Errorf("ollama: tool %q was requested but not registered", call.Name)
		}

		if len(pendingClientCalls) > 0 {
			return &core.ChatResult{
				Text:         "",
				Reasoning:    joinReasoningParts(reasoningParts),
				Messages:     append([]core.MessageUnion(nil), conversation...),
				ToolCalls:    pendingClientCalls,
				FinishReason: "tool_calls",
				Usage:        toCoreChatUsage(response),
			}, nil
		}
	}

	return nil, fmt.Errorf("ollama: reached max tool loop count (%d)", maxLoopCount)
}

// ChatStream sends a streaming chat request to Ollama.
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
		stream := true
		request.Stream = &stream

		url := strings.TrimRight(a.baseURL(), "/") + "/api/chat"
		body, err := json.Marshal(request)
		if err != nil {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("ollama: marshal stream request: %v", err)}
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("ollama: build stream request: %v", err)}
			return
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "application/x-ndjson")
		if strings.TrimSpace(a.APIKey) != "" {
			httpReq.Header.Set("Authorization", "Bearer "+strings.TrimSpace(a.APIKey))
		}

		httpResp, err := a.client().Do(httpReq)
		if err != nil {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("ollama: stream request failed: %v", err)}
			return
		}
		defer httpResp.Body.Close()

		if httpResp.StatusCode >= http.StatusBadRequest {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: decodeAPIError(httpResp).Error()}
			return
		}

		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)

		content := ""
		reasoning := ""
		finishReason := ""
		var usage *core.Usage

		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}

			var event chatResponse
			if err := json.Unmarshal([]byte(line), &event); err != nil {
				out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("ollama: decode stream event: %v", err)}
				return
			}

			usage = toCoreChatUsage(&event)

			nextReasoning, reasoningDelta := appendStreamSegment(reasoning, strings.TrimSpace(event.Message.Thinking))
			reasoning = nextReasoning
			if reasoningDelta != "" {
				out <- core.StreamChunk{
					Type:      core.StreamChunkReasoning,
					Role:      core.RoleAssistant,
					Delta:     reasoningDelta,
					Reasoning: strings.TrimSpace(reasoning),
				}
			}

			nextContent, delta := appendStreamSegment(content, event.Message.Content)
			content = nextContent
			if delta != "" {
				out <- core.StreamChunk{
					Type:    core.StreamChunkContent,
					Role:    core.RoleAssistant,
					Delta:   delta,
					Content: content,
				}
			}

			if event.Done {
				finishReason = nonEmpty(event.DoneReason, "stop")
				out <- core.StreamChunk{
					Type:         core.StreamChunkDone,
					FinishReason: finishReason,
					Reasoning:    strings.TrimSpace(reasoning),
					Usage:        usage,
				}
				return
			}
		}

		if err := scanner.Err(); err != nil {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: fmt.Sprintf("ollama: stream read failed: %v", err)}
			return
		}

		out <- core.StreamChunk{
			Type:         core.StreamChunkDone,
			FinishReason: nonEmpty(finishReason, "stop"),
			Reasoning:    strings.TrimSpace(reasoning),
			Usage:        usage,
		}
	}()

	return out, nil
}

func (a *Adapter) buildRequestTemplate(params *core.ChatParams) (chatRequest, []message, map[string]core.ServerTool, map[string]struct{}, int, error) {
	messages, err := toMessages(params)
	if err != nil {
		return chatRequest{}, nil, nil, nil, 0, err
	}

	tools, serverTools, clientTools, err := toTools(params)
	if err != nil {
		return chatRequest{}, nil, nil, nil, 0, err
	}

	format, err := formatFromOutput(paramsOutput(params))
	if err != nil {
		return chatRequest{}, nil, nil, nil, 0, err
	}

	request := chatRequest{
		Model:   a.Model,
		Tools:   tools,
		Options: requestOptions(params),
		Think:   thinkValue(params),
	}
	if len(format) > 0 {
		request.Format = format
	}

	return request, messages, serverTools, clientTools, maxLoops(params, len(serverTools) > 0), nil
}

func (a *Adapter) postChat(ctx context.Context, request *chatRequest) (*chatResponse, error) {
	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("ollama: marshal request: %w", err)
	}

	url := strings.TrimRight(a.baseURL(), "/") + "/api/chat"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama: build request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	if strings.TrimSpace(a.APIKey) != "" {
		httpReq.Header.Set("Authorization", "Bearer "+strings.TrimSpace(a.APIKey))
	}

	httpResp, err := a.client().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama: request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode >= http.StatusBadRequest {
		return nil, decodeAPIError(httpResp)
	}

	var response chatResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("ollama: decode response: %w", err)
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
