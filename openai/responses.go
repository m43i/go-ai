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

func (a *Adapter) chatResponses(ctx context.Context, params *core.ChatParams) (*core.ChatResult, error) {
	requestTemplate, input, serverTools, clientTools, maxLoopCount, err := a.buildResponsesRequestTemplate(params)
	if err != nil {
		return nil, err
	}

	conversation := cloneCoreMessages(params)
	reasoningParts := make([]string, 0, 4)

	for range maxLoopCount {
		request := requestTemplate
		request.Input = input

		response, err := a.postResponses(ctx, &request)
		if err != nil {
			return nil, err
		}

		text := responseText(response)
		reasoningParts = appendReasoningPart(reasoningParts, responseReasoning(response))
		toolCalls, err := responseToolCalls(response)
		if err != nil {
			return nil, err
		}

		if len(toolCalls) == 0 {
			conversation = append(conversation, core.TextMessagePart{Role: core.RoleAssistant, Content: text})
			return &core.ChatResult{
				Text:         text,
				Reasoning:    joinReasoningParts(reasoningParts),
				Messages:     append([]core.MessageUnion(nil), conversation...),
				FinishReason: responseFinishReason(response),
				Usage:        toCoreResponsesUsage(response.Usage),
			}, nil
		}

		input = append(input, responseFunctionCallInput(toolCalls)...)
		conversation = append(conversation, core.ToolCallMessagePart{Role: core.RoleToolCall, ToolCalls: toolCalls})

		pendingClientCalls := make([]core.ToolCall, 0)
		for _, call := range toolCalls {
			if serverTool, ok := serverTools[call.Name]; ok {
				result, callErr := serverTool.Handler(call.Arguments)
				if callErr != nil {
					result = "tool_error: " + callErr.Error()
				}

				input = append(input, responseInputItem{Type: "function_call_output", CallID: call.ID, Output: result})
				conversation = append(conversation, core.ToolResultMessagePart{Role: core.RoleToolResult, ToolCallID: call.ID, Name: call.Name, Content: result})
				continue
			}

			if _, ok := clientTools[call.Name]; ok {
				pendingClientCalls = append(pendingClientCalls, call)
				continue
			}

			return nil, fmt.Errorf("openai: tool %q was requested but not registered", call.Name)
		}

		if len(pendingClientCalls) > 0 {
			return &core.ChatResult{
				Reasoning:    joinReasoningParts(reasoningParts),
				Messages:     append([]core.MessageUnion(nil), conversation...),
				ToolCalls:    pendingClientCalls,
				FinishReason: "tool_calls",
				Usage:        toCoreResponsesUsage(response.Usage),
			}, nil
		}
	}

	return nil, fmt.Errorf("openai: reached max tool loop count (%d)", maxLoopCount)
}

func (a *Adapter) chatResponsesStream(ctx context.Context, params *core.ChatParams) (<-chan core.StreamChunk, error) {
	request, input, serverTools, clientTools, _, err := a.buildResponsesRequestTemplate(params)
	if err != nil {
		return nil, err
	}

	out := make(chan core.StreamChunk, 64)
	go func() {
		defer close(out)

		if len(serverTools) > 0 || len(clientTools) > 0 || (params != nil && params.Output != nil) {
			result, err := a.chatResponses(ctx, params)
			if err != nil {
				out <- core.StreamChunk{Type: core.StreamChunkError, Error: err.Error()}
				return
			}
			emitChunksFromResult(out, params, result)
			out <- core.StreamChunk{Type: core.StreamChunkDone, FinishReason: nonEmpty(result.FinishReason, defaultFinishReason(result)), Reasoning: result.Reasoning, Usage: result.Usage}
			return
		}

		request.Input = input
		request.Stream = true
		if err := a.streamResponses(ctx, &request, out); err != nil {
			out <- core.StreamChunk{Type: core.StreamChunkError, Error: err.Error()}
		}
	}()

	return out, nil
}

func (a *Adapter) buildResponsesRequestTemplate(params *core.ChatParams) (responsesRequest, []responseInputItem, map[string]core.ServerTool, map[string]struct{}, int, error) {
	input, instructions, err := toResponseInput(params)
	if err != nil {
		return responsesRequest{}, nil, nil, nil, 0, err
	}

	tools, serverTools, clientTools, err := toChatTools(params)
	if err != nil {
		return responsesRequest{}, nil, nil, nil, 0, err
	}

	request := responsesRequest{
		Model:           a.Model,
		Instructions:    instructions,
		Tools:           tools,
		MaxOutputTokens: maxTokens(params),
		Temperature:     temperature(params),
		TopP:            topP(params),
		Metadata:        metadata(params),
		ModelOptions:    modelOptions(params),
	}
	if len(tools) > 0 {
		request.ToolChoice = "auto"
	}
	if params != nil && params.Output != nil {
		request.Text = responseTextFormat(params.Output)
	}
	if effort := reasoningEffort(params); effort != "" {
		request.Reasoning = map[string]any{"effort": effort}
	}

	return request, input, serverTools, clientTools, maxLoops(params, len(serverTools) > 0), nil
}

func responseTextFormat(schema *core.Schema) map[string]any {
	if schema == nil || schema.Schema == nil {
		return nil
	}
	return map[string]any{
		"format": map[string]any{
			"type":   "json_schema",
			"name":   schema.Name,
			"strict": schema.Strict,
			"schema": schema.Schema,
		},
	}
}

func (a *Adapter) postResponses(ctx context.Context, request *responsesRequest) (*responsesResponse, error) {
	body, err := marshalWithModelOptions(request, request.ModelOptions)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal responses request: %w", err)
	}

	url := strings.TrimRight(a.baseURL(), "/") + "/responses"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: build responses request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+a.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := a.client().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: responses request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode >= http.StatusBadRequest {
		return nil, decodeAPIError(httpResp)
	}

	bodyBytes, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: read responses body: %w", err)
	}

	var response responsesResponse
	if err := json.Unmarshal(bodyBytes, &response); err != nil {
		return nil, fmt.Errorf("openai: decode responses response: %w", err)
	}
	var rawEnvelope struct {
		Output []json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(bodyBytes, &rawEnvelope); err == nil {
		response.RawOutput = rawEnvelope.Output
	}

	return &response, nil
}

func (a *Adapter) streamResponses(ctx context.Context, request *responsesRequest, out chan<- core.StreamChunk) error {
	body, err := marshalWithModelOptions(request, request.ModelOptions)
	if err != nil {
		return fmt.Errorf("openai: marshal responses stream request: %w", err)
	}

	url := strings.TrimRight(a.baseURL(), "/") + "/responses"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("openai: build responses stream request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+a.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := a.client().Do(httpReq)
	if err != nil {
		return fmt.Errorf("openai: responses stream request failed: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode >= http.StatusBadRequest {
		return decodeAPIError(httpResp)
	}

	scanner := bufio.NewScanner(httpResp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var content strings.Builder
	var reasoning strings.Builder
	var finalUsage *core.Usage
	finishReason := "stop"

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, ":") || !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			out <- core.StreamChunk{Type: core.StreamChunkDone, FinishReason: finishReason, Reasoning: reasoning.String(), Usage: finalUsage}
			return nil
		}

		var event responsesStreamEvent
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			return fmt.Errorf("openai: decode responses stream event: %w", err)
		}
		switch event.Type {
		case "response.output_text.delta":
			if event.Delta == "" {
				continue
			}
			content.WriteString(event.Delta)
			out <- core.StreamChunk{Type: core.StreamChunkContent, Role: core.RoleAssistant, Delta: event.Delta, Content: content.String()}
		case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
			if event.Delta == "" {
				continue
			}
			reasoning.WriteString(event.Delta)
			out <- core.StreamChunk{Type: core.StreamChunkReasoning, Role: core.RoleAssistant, Delta: event.Delta, Reasoning: reasoning.String()}
		case "response.completed":
			if event.Response != nil {
				finalUsage = toCoreResponsesUsage(event.Response.Usage)
				finishReason = responseFinishReason(event.Response)
			}
			out <- core.StreamChunk{Type: core.StreamChunkDone, FinishReason: finishReason, Reasoning: reasoning.String(), Usage: finalUsage}
			return nil
		case "response.failed", "response.incomplete":
			if event.Response != nil {
				return errors.New("openai: responses stream ended with status " + event.Response.Status)
			}
			return errors.New("openai: responses stream ended with " + event.Type)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("openai: responses stream read failed: %w", err)
	}
	out <- core.StreamChunk{Type: core.StreamChunkDone, FinishReason: finishReason, Reasoning: reasoning.String(), Usage: finalUsage}
	return nil
}

func responseText(response *responsesResponse) string {
	if response == nil {
		return ""
	}
	if strings.TrimSpace(response.OutputText) != "" {
		return response.OutputText
	}
	var builder strings.Builder
	for _, item := range response.Output {
		for _, part := range item.Content {
			builder.WriteString(extractTextFromPart(part))
		}
	}
	return builder.String()
}

func responseReasoning(response *responsesResponse) string {
	if response == nil {
		return ""
	}
	parts := make([]string, 0)
	for _, item := range response.Output {
		if item.Type == "reasoning" {
			if reasoning := extractReasoningFromAny(item.Content); reasoning != "" {
				parts = append(parts, reasoning)
			}
		}
		for _, part := range item.Content {
			if reasoning := extractReasoningFromAny(part); reasoning != "" {
				parts = append(parts, reasoning)
			}
		}
	}
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

func responseToolCalls(response *responsesResponse) ([]core.ToolCall, error) {
	if response == nil {
		return nil, nil
	}
	out := make([]core.ToolCall, 0)
	for _, item := range response.Output {
		if item.Type != "function_call" {
			continue
		}
		arguments, err := parseToolArguments(item.Arguments)
		if err != nil {
			return nil, fmt.Errorf("openai: invalid arguments for tool %q: %w", item.Name, err)
		}
		out = append(out, core.ToolCall{ID: item.CallID, Name: item.Name, Arguments: arguments})
	}
	return out, nil
}

func responseFunctionCallInput(calls []core.ToolCall) []responseInputItem {
	out := make([]responseInputItem, 0, len(calls))
	for _, call := range calls {
		arguments, err := stringifyToolArguments(call.Arguments)
		if err != nil {
			arguments = "{}"
		}
		out = append(out, responseInputItem{Type: "function_call", CallID: call.ID, Name: call.Name, Arguments: arguments})
	}
	return out
}

func responseFinishReason(response *responsesResponse) string {
	if response == nil {
		return "stop"
	}
	if response.IncompleteDetails != nil && strings.TrimSpace(response.IncompleteDetails.Reason) != "" {
		return strings.TrimSpace(response.IncompleteDetails.Reason)
	}
	if strings.TrimSpace(response.Status) != "" && response.Status != "completed" {
		return response.Status
	}
	return "stop"
}

func toCoreResponsesUsage(in *responsesUsage) *core.Usage {
	if in == nil {
		return nil
	}
	reasoningTokens := in.ReasoningTokens
	if in.OutputTokensDetails != nil && in.OutputTokensDetails.ReasoningTokens > 0 {
		reasoningTokens = in.OutputTokensDetails.ReasoningTokens
	}
	return &core.Usage{
		PromptTokens:     in.InputTokens,
		CompletionTokens: in.OutputTokens,
		TotalTokens:      in.TotalTokens,
		ReasoningTokens:  reasoningTokens,
	}
}
