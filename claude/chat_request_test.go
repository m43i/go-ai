package claude

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/m43i/go-ai/core"
)

func TestChatRequestUsesMessagesAPIFields(t *testing.T) {
	t.Parallel()

	var request map[string]any
	var anthropicVersion string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/messages" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		anthropicVersion = r.Header.Get("anthropic-version")
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","role":"assistant","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}}`))
	}))
	defer server.Close()

	maxTokens := int64(42)
	topP := 0.8
	adapter := New("claude-test", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := core.Chat(context.Background(), core.TextOptions{
		Adapter:       adapter,
		SystemPrompts: []string{"Be brief."},
		Messages: []core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "hi"},
		},
		MaxTokens: &maxTokens,
		TopP:      &topP,
		Metadata:  map[string]any{"user_id": "user-123"},
		ModelOptions: map[string]any{
			"topK":          20,
			"stopSequences": []string{"END"},
			"serviceTier":   "standard_only",
		},
	})
	if err != nil {
		t.Fatalf("chat returned error: %v", err)
	}
	if result.Text != "hello" {
		t.Fatalf("unexpected result text: %q", result.Text)
	}
	if anthropicVersion != defaultVersion {
		t.Fatalf("expected default anthropic-version %q, got %q", defaultVersion, anthropicVersion)
	}
	if request["max_tokens"].(float64) != 42 {
		t.Fatalf("max_tokens not set correctly: %#v", request)
	}
	if request["system"] != "Be brief." {
		t.Fatalf("system prompt not mapped to top-level system: %#v", request)
	}
	if request["top_p"].(float64) != 0.8 {
		t.Fatalf("top_p not set correctly: %#v", request)
	}
	if request["top_k"].(float64) != 20 {
		t.Fatalf("model option topK not converted to top_k: %#v", request)
	}
	if request["service_tier"] != "standard_only" {
		t.Fatalf("model option serviceTier not converted: %#v", request)
	}
	if request["metadata"].(map[string]any)["user_id"] != "user-123" {
		t.Fatalf("metadata not forwarded: %#v", request)
	}
}

func TestChatRequestUsesOutputConfigForStructuredOutput(t *testing.T) {
	t.Parallel()

	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","role":"assistant","content":[{"type":"text","text":"{}"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}}`))
	}))
	defer server.Close()

	schema := core.Schema{
		Name:   "answer",
		Strict: true,
		Schema: map[string]any{
			"type":                 "object",
			"properties":           map[string]any{"answer": map[string]any{"type": "string"}},
			"required":             []string{"answer"},
			"additionalProperties": false,
		},
	}
	adapter := New("claude-test", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := core.Chat(context.Background(), core.TextOptions{
		Adapter: adapter,
		Messages: []core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "hi"},
		},
		Output:          &schema,
		ReasoningEffort: "low",
	})
	if err != nil {
		t.Fatalf("chat returned error: %v", err)
	}

	if request["output_config"] == nil {
		t.Fatalf("expected output_config, got %#v", request)
	}
	config := request["output_config"].(map[string]any)
	if config["effort"] != "low" {
		t.Fatalf("expected effort to be set, got %#v", config)
	}
	format := config["format"].(map[string]any)
	if format["type"] != "json_schema" {
		t.Fatalf("unexpected output format: %#v", format)
	}
	if format["schema"] == nil {
		t.Fatalf("schema was not forwarded: %#v", format)
	}
}

func TestChatRequestDefaultsMaxTokensAndAccountsForThinkingBudget(t *testing.T) {
	t.Parallel()

	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}}`))
	}))
	defer server.Close()

	adapter := New("claude-test", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := core.Chat(context.Background(), core.TextOptions{
		Adapter: adapter,
		Messages: []core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "hi"},
		},
		ModelOptions: map[string]any{
			"thinking": map[string]any{
				"type":          "enabled",
				"budget_tokens": 2048,
			},
		},
	})
	if err != nil {
		t.Fatalf("chat returned error: %v", err)
	}
	if request["max_tokens"].(float64) != 2049 {
		t.Fatalf("expected max_tokens to exceed thinking budget, got %#v", request["max_tokens"])
	}
}
