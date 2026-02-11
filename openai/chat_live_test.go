package openai

import (
	"context"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/m43i/go-ai/core"
)

const liveTimeout = 120 * time.Second

func TestPublicChatLive(t *testing.T) {
	adapter := requireLiveAdapter(t)

	t.Run("NormalTextResponse", func(t *testing.T) {
		result := runChat(t, adapter, liveParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "Reply with exactly LIVE_TEXT_OK."},
		}, nil, nil))

		text, err := core.LastAssistantText(result)
		if err != nil {
			t.Fatalf("expected assistant text, got error: %v", err)
		}
		if strings.TrimSpace(text) == "" {
			t.Fatal("assistant text is empty")
		}
	})

	t.Run("ContinuingConversationByPassingMessages", func(t *testing.T) {
		first := runChat(t, adapter, liveParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "Remember token CONTINUE_CODE_731 and reply READY."},
		}, nil, nil))

		conversation := append([]core.MessageUnion(nil), first.Messages...)
		conversation = append(conversation, core.TextMessagePart{
			Role:    core.RoleUser,
			Content: "What token did I ask you to remember?",
		})

		second := runChat(t, adapter, liveParams(conversation, nil, nil))
		text, err := core.LastAssistantText(second)
		if err != nil {
			t.Fatalf("expected assistant text in continued conversation, got error: %v", err)
		}
		if strings.TrimSpace(text) == "" {
			t.Fatal("continued conversation returned empty assistant text")
		}
		if len(second.Messages) <= len(conversation) {
			t.Fatalf("expected conversation to grow, before=%d after=%d", len(conversation), len(second.Messages))
		}
	})

	t.Run("ToolCallsServerOnly", func(t *testing.T) {
		called := 0
		tools := []core.ToolUnion{
			core.ServerTool{
				Name:        "server_echo",
				Description: "Returns a fixed server value.",
				Parameters:  toolTextParameters(),
				Handler: func(_ any) (string, error) {
					called++
					return "SERVER_TOOL_RESULT_321", nil
				},
			},
		}

		messages := []core.MessageUnion{
			core.TextMessagePart{Role: core.RoleSystem, Content: "When explicitly asked to use a tool, use the tool."},
			core.TextMessagePart{Role: core.RoleUser, Content: "Call server_echo exactly once with text=SERVER_TOOL_RESULT_321, then return the tool result."},
		}

		result := runChat(t, adapter, liveParams(messages, tools, nil))
		if called == 0 {
			t.Fatal("expected server tool handler to be called")
		}
		if len(result.ToolCalls) != 0 {
			t.Fatalf("expected no pending client tool calls, got %#v", result.ToolCalls)
		}
	})

	t.Run("ToolCallsClientOnly", func(t *testing.T) {
		tools := []core.ToolUnion{
			core.ClientTool{
				Name:        "client_ping",
				Description: "Client-side ping tool.",
				Parameters:  toolTextParameters(),
			},
		}

		messages := []core.MessageUnion{
			core.TextMessagePart{Role: core.RoleSystem, Content: "When explicitly asked to use a tool, use the tool."},
			core.TextMessagePart{Role: core.RoleUser, Content: "Call client_ping with text=ping and stop."},
		}

		result := runChat(t, adapter, liveParams(messages, tools, nil))
		if len(result.ToolCalls) == 0 {
			t.Fatal("expected pending client tool call")
		}
		if !containsToolCall(result.ToolCalls, "client_ping") {
			t.Fatalf("expected pending tool call named client_ping, got %#v", result.ToolCalls)
		}
	})

	t.Run("ToolCallsMixedServerAndClient", func(t *testing.T) {
		called := 0
		tools := []core.ToolUnion{
			core.ServerTool{
				Name:        "server_echo",
				Description: "Server-side tool.",
				Parameters:  toolTextParameters(),
				Handler: func(_ any) (string, error) {
					called++
					return "SERVER_MIX_RESULT_9", nil
				},
			},
			core.ClientTool{
				Name:        "client_ping",
				Description: "Client-side tool.",
				Parameters:  toolTextParameters(),
			},
		}

		messages := []core.MessageUnion{
			core.TextMessagePart{Role: core.RoleSystem, Content: "Call both tools when asked."},
			core.TextMessagePart{Role: core.RoleUser, Content: "Call server_echo and client_ping, then stop after issuing the client call."},
		}

		params := liveParams(messages, tools, nil)
		params.MaxAgenticLoops = 4

		result := runChat(t, adapter, params)
		if called == 0 {
			t.Fatal("expected server tool to run in mixed flow")
		}
		if len(result.ToolCalls) == 0 {
			t.Fatal("expected pending client tool call in mixed flow")
		}
		if !containsToolCall(result.ToolCalls, "client_ping") {
			t.Fatalf("expected pending client tool call named client_ping, got %#v", result.ToolCalls)
		}
	})

	t.Run("OutputFormat", func(t *testing.T) {
		type outputPayload struct {
			Status string `json:"status"`
			Count  int    `json:"count"`
		}

		schema, err := core.New("openai_live_output", outputPayload{})
		if err != nil {
			t.Fatalf("failed to build output schema: %v", err)
		}

		result := runChat(t, adapter, liveParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "Return JSON with status=ok and count=7."},
		}, nil, &schema))

		decoded, err := core.DecodeLast[outputPayload](result)
		if err != nil {
			t.Fatalf("failed to decode structured output: %v", err)
		}
		if strings.ToLower(strings.TrimSpace(decoded.Status)) != "ok" {
			t.Fatalf("expected status=ok, got %q", decoded.Status)
		}
		if decoded.Count != 7 {
			t.Fatalf("expected count=7, got %d", decoded.Count)
		}
	})

	t.Run("ReasoningContent", func(t *testing.T) {
		reasoningAdapter := requireReasoningLiveAdapter(t)

		result := runChat(t, reasoningAdapter, liveParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "Briefly reason about why 9.11 is greater than 9.9, then give the answer."},
		}, nil, nil))

		if strings.TrimSpace(result.Reasoning) == "" {
			t.Fatal("expected non-empty reasoning content")
		}
	})
}

func TestPublicChatStreamLive(t *testing.T) {
	adapter := requireLiveAdapter(t)

	t.Run("NormalTextResponse", func(t *testing.T) {
		summary := runStream(t, adapter, liveParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "Reply with exactly STREAM_TEXT_OK."},
		}, nil, nil))
		if strings.TrimSpace(summary.Content) == "" {
			t.Fatal("expected non-empty stream content")
		}
	})

	t.Run("ContinuingConversationByPassingMessages", func(t *testing.T) {
		first := runChat(t, adapter, liveParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "Remember token STREAM_MEMORY_208 and reply READY."},
		}, nil, nil))

		conversation := append([]core.MessageUnion(nil), first.Messages...)
		conversation = append(conversation, core.TextMessagePart{
			Role:    core.RoleUser,
			Content: "What token did I ask you to remember?",
		})

		summary := runStream(t, adapter, liveParams(conversation, nil, nil))
		if strings.TrimSpace(summary.Content) == "" {
			t.Fatal("expected non-empty stream content for continued conversation")
		}
	})

	t.Run("ToolCallsServerOnly", func(t *testing.T) {
		called := 0
		tools := []core.ToolUnion{
			core.ServerTool{
				Name:        "server_echo",
				Description: "Server-side tool.",
				Parameters:  toolTextParameters(),
				Handler: func(_ any) (string, error) {
					called++
					return "STREAM_SERVER_RESULT_555", nil
				},
			},
		}

		summary := runStream(t, adapter, liveParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleSystem, Content: "When explicitly asked to use a tool, use the tool."},
			core.TextMessagePart{Role: core.RoleUser, Content: "Call server_echo with text=STREAM_SERVER_RESULT_555 and return the result."},
		}, tools, nil))

		if called == 0 {
			t.Fatal("expected server tool to run in stream flow")
		}
		if !summary.SawToolCall {
			t.Fatal("expected stream tool_call chunk")
		}
		if !summary.SawToolResult {
			t.Fatal("expected stream tool_result chunk")
		}
	})

	t.Run("ToolCallsClientOnly", func(t *testing.T) {
		summary := runStream(t, adapter, liveParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleSystem, Content: "When explicitly asked to use a tool, use the tool."},
			core.TextMessagePart{Role: core.RoleUser, Content: "Call client_ping with text=ping and stop."},
		}, []core.ToolUnion{
			core.ClientTool{Name: "client_ping", Description: "Client-side tool.", Parameters: toolTextParameters()},
		}, nil))

		if !summary.SawToolCall {
			t.Fatal("expected stream tool_call chunk for client tool")
		}
		if summary.SawToolResult {
			t.Fatal("did not expect stream tool_result chunk for client-only flow")
		}
	})

	t.Run("ToolCallsMixedServerAndClient", func(t *testing.T) {
		called := 0
		summary := runStream(t, adapter, liveParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleSystem, Content: "Call both tools when asked."},
			core.TextMessagePart{Role: core.RoleUser, Content: "Call server_echo and client_ping, then stop after issuing the client call."},
		}, []core.ToolUnion{
			core.ServerTool{Name: "server_echo", Description: "Server-side tool.", Parameters: toolTextParameters(), Handler: func(_ any) (string, error) {
				called++
				return "STREAM_MIX_SERVER_RESULT_22", nil
			}},
			core.ClientTool{Name: "client_ping", Description: "Client-side tool.", Parameters: toolTextParameters()},
		}, nil))

		if called == 0 {
			t.Fatal("expected server tool to run in mixed stream flow")
		}
		if !summary.SawToolCall {
			t.Fatal("expected stream tool_call chunk in mixed flow")
		}
		if !summary.SawToolResult {
			t.Fatal("expected stream tool_result chunk in mixed flow")
		}
	})

	t.Run("OutputFormat", func(t *testing.T) {
		type outputPayload struct {
			Status string `json:"status"`
			Count  int    `json:"count"`
		}

		schema, err := core.New("openai_live_stream_output", outputPayload{})
		if err != nil {
			t.Fatalf("failed to build output schema: %v", err)
		}

		summary := runStream(t, adapter, liveParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "Return JSON with status=ok and count=9."},
		}, nil, &schema))

		var decoded outputPayload
		if err := jsonUnmarshalStrict(summary.Content, &decoded); err != nil {
			t.Fatalf("failed to decode stream structured output %q: %v", summary.Content, err)
		}
		if strings.ToLower(strings.TrimSpace(decoded.Status)) != "ok" {
			t.Fatalf("expected status=ok, got %q", decoded.Status)
		}
		if decoded.Count != 9 {
			t.Fatalf("expected count=9, got %d", decoded.Count)
		}
	})

	t.Run("ReasoningContent", func(t *testing.T) {
		reasoningAdapter := requireReasoningLiveAdapter(t)

		summary := runStream(t, reasoningAdapter, liveParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "Briefly reason about why 9.11 is greater than 9.9, then give the answer."},
		}, nil, nil))

		if !summary.SawReasoning {
			t.Fatal("expected at least one reasoning chunk")
		}
		if strings.TrimSpace(summary.Reasoning) == "" {
			t.Fatal("expected non-empty reasoning content in stream")
		}
	})
}

type streamSummary struct {
	Content       string
	Reasoning     string
	FinishReason  string
	SawDone       bool
	SawReasoning  bool
	SawToolCall   bool
	SawToolResult bool
}

func runChat(t *testing.T, adapter *Adapter, params *core.ChatParams) *core.ChatResult {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), liveTimeout)
	defer cancel()

	result, err := core.Chat(ctx, adapter, params)
	if err != nil {
		t.Fatalf("chat failed: %v", err)
	}
	if result == nil {
		t.Fatal("chat returned nil result")
	}

	return result
}

func runStream(t *testing.T, adapter *Adapter, params *core.ChatParams) streamSummary {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), liveTimeout)
	defer cancel()

	stream, err := core.ChatStream(ctx, adapter, params)
	if err != nil {
		t.Fatalf("chat stream failed: %v", err)
	}

	var summary streamSummary

	for {
		select {
		case <-ctx.Done():
			t.Fatalf("chat stream timed out: %v", ctx.Err())
		case chunk, ok := <-stream:
			if !ok {
				if !summary.SawDone {
					t.Fatal("chat stream closed without done chunk")
				}
				return summary
			}

			switch chunk.Type {
			case core.StreamChunkError:
				t.Fatalf("chat stream returned error chunk: %s", chunk.Error)
			case core.StreamChunkReasoning:
				summary.SawReasoning = true
				if chunk.Reasoning != "" {
					summary.Reasoning = chunk.Reasoning
				} else {
					summary.Reasoning += chunk.Delta
				}
			case core.StreamChunkContent:
				if chunk.Content != "" {
					summary.Content = chunk.Content
				} else {
					summary.Content += chunk.Delta
				}
			case core.StreamChunkToolCall:
				summary.SawToolCall = true
			case core.StreamChunkToolResult:
				summary.SawToolResult = true
			case core.StreamChunkDone:
				summary.SawDone = true
				summary.FinishReason = chunk.FinishReason
				if strings.TrimSpace(summary.Reasoning) == "" {
					summary.Reasoning = chunk.Reasoning
				}
			}
		}
	}
}

func liveParams(messages []core.MessageUnion, tools []core.ToolUnion, output *core.Schema) *core.ChatParams {
	maxOutputTokens := liveMaxOutputTokens()
	temperature := 0.0

	return &core.ChatParams{
		Messages:        messages,
		Tools:           tools,
		Output:          output,
		MaxOutputTokens: &maxOutputTokens,
		Temperature:     &temperature,
		ReasoningEffort: liveReasoningEffort(),
	}
}

func containsToolCall(calls []core.ToolCall, name string) bool {
	for _, call := range calls {
		if strings.TrimSpace(call.Name) == name {
			return true
		}
	}
	return false
}

func toolTextParameters() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"text": map[string]any{"type": "string"},
		},
		"required":             []string{"text"},
		"additionalProperties": false,
	}
}

func requireLiveAdapter(t *testing.T) *Adapter {
	t.Helper()

	apiKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set; skipping live OpenAI tests")
	}

	model := strings.TrimSpace(os.Getenv("GOAI_OPENAI_CHAT_MODEL"))
	if model == "" {
		model = "gpt-4o-mini"
	}

	baseURL := strings.TrimSpace(os.Getenv("GOAI_OPENAI_BASE_URL"))
	if baseURL == "" {
		baseURL = strings.TrimSpace(os.Getenv("OPENAI_BASE_URL"))
	}

	if baseURL == "" {
		return New(apiKey, model)
	}

	return New(apiKey, model, WithBaseURL(baseURL))
}

func requireReasoningLiveAdapter(t *testing.T) *Adapter {
	t.Helper()

	apiKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set; skipping live OpenAI reasoning tests")
	}

	model := strings.TrimSpace(os.Getenv("GOAI_OPENAI_REASONING_MODEL"))
	if model == "" {
		t.Skip("GOAI_OPENAI_REASONING_MODEL not set; skipping reasoning-specific live tests")
	}

	baseURL := strings.TrimSpace(os.Getenv("GOAI_OPENAI_BASE_URL"))
	if baseURL == "" {
		baseURL = strings.TrimSpace(os.Getenv("OPENAI_BASE_URL"))
	}

	if baseURL == "" {
		return New(apiKey, model)
	}

	return New(apiKey, model, WithBaseURL(baseURL))
}

func liveMaxOutputTokens() int64 {
	raw := strings.TrimSpace(os.Getenv("GOAI_OPENAI_LIVE_MAX_OUTPUT_TOKENS"))
	if raw == "" {
		return 512
	}

	v, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || v <= 0 {
		return 512
	}

	return v
}

func liveReasoningEffort() string {
	v := strings.TrimSpace(os.Getenv("GOAI_OPENAI_LIVE_REASONING_EFFORT"))
	if v == "" {
		return "low"
	}
	return v
}

func jsonUnmarshalStrict(text string, out any) error {
	payload := strings.TrimSpace(text)
	if strings.HasPrefix(payload, "```") {
		payload = strings.TrimPrefix(payload, "```json")
		payload = strings.TrimPrefix(payload, "```")
		payload = strings.TrimSuffix(payload, "```")
		payload = strings.TrimSpace(payload)
	}

	return json.Unmarshal([]byte(payload), out)
}
