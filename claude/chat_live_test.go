package claude

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/m43i/go-ai/core"
)

const liveTestTimeout = 120 * time.Second

func TestChatLiveScenarios(t *testing.T) {
	apiKey, model, baseURL, version := requireClaudeLiveConfig(t)
	adapter := newClaudeLiveTestAdapter(apiKey, model, baseURL, version)

	t.Run("NormalTextResponse", func(t *testing.T) {
		result := runClaudeChat(t, adapter, baseChatParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "Reply with exactly: LIVE_TEXT_OK"},
		}, nil, nil))

		text := mustAssistantText(t, result)
		if !strings.Contains(strings.ToUpper(text), "LIVE_TEXT_OK") {
			t.Fatalf("expected response to contain LIVE_TEXT_OK, got %q", text)
		}
	})

	t.Run("ContinuingConversationByPassingMessages", func(t *testing.T) {
		const memoryToken = "CONTINUE_CODE_731"

		first := runClaudeChat(t, adapter, baseChatParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "Remember this code: CONTINUE_CODE_731. Reply only READY."},
		}, nil, nil))

		conversation := append([]core.MessageUnion(nil), first.Messages...)
		conversation = append(conversation, core.TextMessagePart{
			Role:    core.RoleUser,
			Content: "What code did I ask you to remember? Reply only with the code.",
		})

		second := runClaudeChat(t, adapter, baseChatParams(conversation, nil, nil))
		text := mustAssistantText(t, second)
		if !strings.Contains(strings.ToUpper(text), memoryToken) {
			t.Fatalf("expected continuation response to contain %q, got %q", memoryToken, text)
		}
	})

	t.Run("ToolCallsServerOnly", func(t *testing.T) {
		called := 0
		tools := []core.ToolUnion{
			core.ServerTool{
				Name:        "server_echo",
				Description: "Echoes back a fixed server-side value.",
				Parameters:  toolTextParameters(),
				Handler: func(_ any) (string, error) {
					called++
					return "SERVER_TOOL_RESULT_321", nil
				},
			},
		}

		messages := []core.MessageUnion{
			core.TextMessagePart{Role: core.RoleSystem, Content: "When told to use tools, you must use them."},
			core.TextMessagePart{Role: core.RoleUser, Content: "Use the server_echo tool exactly once, then answer with its result."},
		}

		result := runClaudeChat(t, adapter, baseChatParams(messages, tools, nil))

		if called == 0 {
			t.Fatal("expected server tool handler to be called at least once")
		}
		if len(result.ToolCalls) != 0 {
			t.Fatalf("expected no pending client tool calls, got %#v", result.ToolCalls)
		}
		if !hasToolCallMessage(result.Messages) {
			t.Fatal("expected tool call message in conversation")
		}
		if !hasToolResultMessage(result.Messages) {
			t.Fatal("expected tool result message in conversation")
		}

		text := mustAssistantText(t, result)
		if !strings.Contains(strings.ToUpper(text), "SERVER_TOOL_RESULT_321") {
			t.Fatalf("expected assistant response to include server tool result, got %q", text)
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
			core.TextMessagePart{Role: core.RoleSystem, Content: "When a tool is requested, call the tool and stop."},
			core.TextMessagePart{Role: core.RoleUser, Content: "Call the client_ping tool with text=ping and stop."},
		}

		result := runClaudeChat(t, adapter, baseChatParams(messages, tools, nil))

		if len(result.ToolCalls) == 0 {
			t.Fatal("expected pending client tool call, got none")
		}
		if !hasToolCallNamed(result.ToolCalls, "client_ping") {
			t.Fatalf("expected pending client tool call named client_ping, got %#v", result.ToolCalls)
		}
		if !hasToolCallMessage(result.Messages) {
			t.Fatal("expected tool call message in conversation")
		}
		if hasToolResultMessage(result.Messages) {
			t.Fatal("did not expect tool result message for client-only tool call handoff")
		}
		if result.FinishReason != "tool_calls" {
			t.Fatalf("expected finish reason tool_calls, got %q", result.FinishReason)
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
			core.TextMessagePart{Role: core.RoleSystem, Content: "You must call both tools, first server_echo then client_ping."},
			core.TextMessagePart{Role: core.RoleUser, Content: "Call both tools now and stop after issuing the client tool call."},
		}

		params := baseChatParams(messages, tools, nil)
		params.MaxAgenticLoops = 4

		result := runClaudeChat(t, adapter, params)

		if called == 0 {
			t.Fatal("expected server tool handler to be called in mixed flow")
		}
		if len(result.ToolCalls) == 0 {
			t.Fatal("expected pending client tool call in mixed flow")
		}
		if !hasToolCallNamed(result.ToolCalls, "client_ping") {
			t.Fatalf("expected pending client call named client_ping, got %#v", result.ToolCalls)
		}
		if !hasToolResultMessage(result.Messages) {
			t.Fatal("expected server tool result message in mixed flow")
		}
		if result.FinishReason != "tool_calls" {
			t.Fatalf("expected finish reason tool_calls, got %q", result.FinishReason)
		}
	})

	t.Run("OutputFormat", func(t *testing.T) {
		type outputPayload struct {
			Status string `json:"status"`
			Count  int    `json:"count"`
		}

		schema, err := core.New("live_claude_output", outputPayload{})
		if err != nil {
			t.Fatalf("failed to build schema: %v", err)
		}

		result := runClaudeChat(t, adapter, baseChatParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "Return JSON with status set to \"ok\" and count set to 7."},
		}, nil, &schema))

		text := mustAssistantText(t, result)
		payload := normalizeJSONPayload(text)

		var decoded outputPayload
		if err := json.Unmarshal([]byte(payload), &decoded); err != nil {
			t.Fatalf("failed to decode structured output %q: %v", text, err)
		}
		if strings.ToLower(strings.TrimSpace(decoded.Status)) != "ok" {
			t.Fatalf("expected status=ok, got %#v", decoded.Status)
		}
		if decoded.Count != 7 {
			t.Fatalf("expected count=7, got %d", decoded.Count)
		}
	})
}

func TestChatStreamLiveScenarios(t *testing.T) {
	apiKey, model, baseURL, version := requireClaudeLiveConfig(t)
	adapter := newClaudeLiveTestAdapter(apiKey, model, baseURL, version)

	t.Run("NormalTextResponse", func(t *testing.T) {
		stream, err := adapter.ChatStream(context.Background(), baseChatParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "Reply with exactly: STREAM_TEXT_OK"},
		}, nil, nil))
		if err != nil {
			t.Fatalf("claude live chat stream failed: %v", err)
		}

		snapshot := readClaudeStream(t, stream)
		if !strings.Contains(strings.ToUpper(snapshot.Content), "STREAM_TEXT_OK") {
			t.Fatalf("expected stream content to contain STREAM_TEXT_OK, got %q", snapshot.Content)
		}
	})

	t.Run("ContinuingConversationByPassingMessages", func(t *testing.T) {
		const memoryToken = "STREAM_MEMORY_208"

		first := runClaudeChat(t, adapter, baseChatParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "Remember this code: STREAM_MEMORY_208. Reply only READY."},
		}, nil, nil))

		conversation := append([]core.MessageUnion(nil), first.Messages...)
		conversation = append(conversation, core.TextMessagePart{
			Role:    core.RoleUser,
			Content: "What code did I ask you to remember? Reply only with the code.",
		})

		stream, err := adapter.ChatStream(context.Background(), baseChatParams(conversation, nil, nil))
		if err != nil {
			t.Fatalf("claude live continuation stream failed: %v", err)
		}

		snapshot := readClaudeStream(t, stream)
		if !strings.Contains(strings.ToUpper(snapshot.Content), memoryToken) {
			t.Fatalf("expected continuation stream response to contain %q, got %q", memoryToken, snapshot.Content)
		}
	})

	t.Run("ToolCallsServerOnly", func(t *testing.T) {
		called := 0
		tools := []core.ToolUnion{
			core.ServerTool{
				Name:        "server_echo",
				Description: "Echoes back a fixed server-side value.",
				Parameters:  toolTextParameters(),
				Handler: func(_ any) (string, error) {
					called++
					return "STREAM_SERVER_RESULT_555", nil
				},
			},
		}

		messages := []core.MessageUnion{
			core.TextMessagePart{Role: core.RoleSystem, Content: "When told to use tools, you must use them."},
			core.TextMessagePart{Role: core.RoleUser, Content: "Use the server_echo tool once and then answer with its result."},
		}

		stream, err := adapter.ChatStream(context.Background(), baseChatParams(messages, tools, nil))
		if err != nil {
			t.Fatalf("claude live server-tool stream failed: %v", err)
		}

		snapshot := readClaudeStream(t, stream)
		if called == 0 {
			t.Fatal("expected server tool handler to be called in stream flow")
		}
		if !snapshot.SawToolCall {
			t.Fatal("expected stream to include tool_call chunk")
		}
		if !snapshot.SawToolResult {
			t.Fatal("expected stream to include tool_result chunk")
		}
		if !strings.Contains(strings.ToUpper(snapshot.Content), "STREAM_SERVER_RESULT_555") {
			t.Fatalf("expected stream assistant content to include tool result, got %q", snapshot.Content)
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
			core.TextMessagePart{Role: core.RoleSystem, Content: "When a tool is requested, call the tool and stop."},
			core.TextMessagePart{Role: core.RoleUser, Content: "Call the client_ping tool with text=ping and stop."},
		}

		stream, err := adapter.ChatStream(context.Background(), baseChatParams(messages, tools, nil))
		if err != nil {
			t.Fatalf("claude live client-tool stream failed: %v", err)
		}

		snapshot := readClaudeStream(t, stream)
		if !snapshot.SawToolCall {
			t.Fatal("expected stream to include tool_call chunk for client tool")
		}
		if snapshot.SawToolResult {
			t.Fatal("did not expect tool_result chunk for client-only tool call")
		}
		if snapshot.Done.FinishReason != "tool_calls" {
			t.Fatalf("expected stream finish reason tool_calls, got %q", snapshot.Done.FinishReason)
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
					return "STREAM_MIX_SERVER_RESULT_22", nil
				},
			},
			core.ClientTool{
				Name:        "client_ping",
				Description: "Client-side tool.",
				Parameters:  toolTextParameters(),
			},
		}

		messages := []core.MessageUnion{
			core.TextMessagePart{Role: core.RoleSystem, Content: "You must call both tools, first server_echo then client_ping."},
			core.TextMessagePart{Role: core.RoleUser, Content: "Call both tools now and stop after issuing the client tool call."},
		}

		params := baseChatParams(messages, tools, nil)
		params.MaxAgenticLoops = 4

		stream, err := adapter.ChatStream(context.Background(), params)
		if err != nil {
			t.Fatalf("claude live mixed-tool stream failed: %v", err)
		}

		snapshot := readClaudeStream(t, stream)
		if called == 0 {
			t.Fatal("expected server tool handler to be called in mixed stream flow")
		}
		if !snapshot.SawToolCall {
			t.Fatal("expected stream to include tool_call chunk in mixed flow")
		}
		if !snapshot.SawToolResult {
			t.Fatal("expected stream to include tool_result chunk in mixed flow")
		}
		if snapshot.Done.FinishReason != "tool_calls" {
			t.Fatalf("expected stream finish reason tool_calls, got %q", snapshot.Done.FinishReason)
		}
	})

	t.Run("OutputFormat", func(t *testing.T) {
		type outputPayload struct {
			Status string `json:"status"`
			Count  int    `json:"count"`
		}

		schema, err := core.New("live_claude_stream_output", outputPayload{})
		if err != nil {
			t.Fatalf("failed to build schema: %v", err)
		}

		stream, err := adapter.ChatStream(context.Background(), baseChatParams([]core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "Return JSON with status set to \"ok\" and count set to 9."},
		}, nil, &schema))
		if err != nil {
			t.Fatalf("claude live output-format stream failed: %v", err)
		}

		snapshot := readClaudeStream(t, stream)
		payload := normalizeJSONPayload(snapshot.Content)

		var decoded outputPayload
		if err := json.Unmarshal([]byte(payload), &decoded); err != nil {
			t.Fatalf("failed to decode streamed structured output %q: %v", snapshot.Content, err)
		}
		if strings.ToLower(strings.TrimSpace(decoded.Status)) != "ok" {
			t.Fatalf("expected status=ok, got %#v", decoded.Status)
		}
		if decoded.Count != 9 {
			t.Fatalf("expected count=9, got %d", decoded.Count)
		}
	})
}

type claudeStreamSnapshot struct {
	Content       string
	Done          core.StreamChunk
	SawDone       bool
	SawToolCall   bool
	SawToolResult bool
}

func runClaudeChat(t *testing.T, adapter *Adapter, params *core.ChatParams) *core.ChatResult {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), liveTestTimeout)
	defer cancel()

	result, err := adapter.Chat(ctx, params)
	if err != nil {
		t.Fatalf("claude live chat failed: %v", err)
	}
	if result == nil {
		t.Fatal("claude live chat returned nil result")
	}

	return result
}

func readClaudeStream(t *testing.T, stream <-chan core.StreamChunk) claudeStreamSnapshot {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), liveTestTimeout)
	defer cancel()

	var snapshot claudeStreamSnapshot

	for {
		select {
		case <-ctx.Done():
			t.Fatalf("claude live chat stream timed out: %v", ctx.Err())
		case chunk, ok := <-stream:
			if !ok {
				if !snapshot.SawDone {
					t.Fatal("claude live chat stream closed without done chunk")
				}
				return snapshot
			}

			switch chunk.Type {
			case core.StreamChunkError:
				t.Fatalf("claude live chat stream returned error chunk: %s", chunk.Error)
			case core.StreamChunkContent:
				if chunk.Content != "" {
					snapshot.Content = chunk.Content
				} else {
					snapshot.Content += chunk.Delta
				}
			case core.StreamChunkToolCall:
				snapshot.SawToolCall = true
			case core.StreamChunkToolResult:
				snapshot.SawToolResult = true
			case core.StreamChunkDone:
				snapshot.SawDone = true
				snapshot.Done = chunk
			}
		}
	}
}

func baseChatParams(messages []core.MessageUnion, tools []core.ToolUnion, output *core.Schema) *core.ChatParams {
	maxOutputTokens := int64(128)
	temperature := 0.0

	return &core.ChatParams{
		Messages:        messages,
		Tools:           tools,
		Output:          output,
		MaxOutputTokens: &maxOutputTokens,
		Temperature:     &temperature,
	}
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

func mustAssistantText(t *testing.T, result *core.ChatResult) string {
	t.Helper()

	text, err := core.LastAssistantText(result)
	if err != nil {
		t.Fatalf("failed to extract assistant text: %v", err)
	}
	if strings.TrimSpace(text) == "" {
		t.Fatal("assistant text is empty")
	}

	return text
}

func hasToolCallNamed(calls []core.ToolCall, name string) bool {
	for _, call := range calls {
		if strings.TrimSpace(call.Name) == name {
			return true
		}
	}
	return false
}

func hasToolCallMessage(messages []core.MessageUnion) bool {
	for _, message := range messages {
		switch typed := message.(type) {
		case core.ToolCallMessagePart:
			if len(typed.ToolCalls) > 0 {
				return true
			}
		case *core.ToolCallMessagePart:
			if typed != nil && len(typed.ToolCalls) > 0 {
				return true
			}
		}
	}
	return false
}

func hasToolResultMessage(messages []core.MessageUnion) bool {
	for _, message := range messages {
		switch message.(type) {
		case core.ToolResultMessagePart, *core.ToolResultMessagePart:
			return true
		}
	}
	return false
}

func normalizeJSONPayload(text string) string {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return trimmed
	}

	trimmed = strings.TrimPrefix(trimmed, "```json")
	trimmed = strings.TrimPrefix(trimmed, "```")
	trimmed = strings.TrimSuffix(trimmed, "```")
	trimmed = strings.TrimSpace(trimmed)

	if json.Valid([]byte(trimmed)) {
		return trimmed
	}

	start := strings.Index(trimmed, "{")
	end := strings.LastIndex(trimmed, "}")
	if start >= 0 && end > start {
		candidate := strings.TrimSpace(trimmed[start : end+1])
		if json.Valid([]byte(candidate)) {
			return candidate
		}
	}

	return trimmed
}

func requireClaudeLiveConfig(t *testing.T) (string, string, string, string) {
	t.Helper()

	apiKey := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY"))
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv("CLAUDE_API_KEY"))
	}
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY/CLAUDE_API_KEY not set; skipping live Claude chat tests")
	}

	model := strings.TrimSpace(os.Getenv("GOAI_CLAUDE_CHAT_MODEL"))
	if model == "" {
		model = "claude-3-5-haiku-latest"
	}

	baseURL := strings.TrimSpace(os.Getenv("GOAI_CLAUDE_BASE_URL"))

	version := strings.TrimSpace(os.Getenv("GOAI_CLAUDE_ANTHROPIC_VERSION"))
	if version == "" {
		version = "2023-06-01"
	}

	return apiKey, model, baseURL, version
}

func newClaudeLiveTestAdapter(apiKey, model, baseURL, version string) *Adapter {
	opts := make([]Option, 0, 2)
	if strings.TrimSpace(baseURL) != "" {
		opts = append(opts, WithBaseURL(baseURL))
	}
	if strings.TrimSpace(version) != "" {
		opts = append(opts, WithAnthropicVersion(version))
	}

	return New(apiKey, model, opts...)
}
