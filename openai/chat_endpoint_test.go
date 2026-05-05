package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/m43i/go-ai/core"
)

func TestChatCompletionsReceivesCommonAndModelOptions(t *testing.T) {
	t.Parallel()

	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`))
	}))
	defer server.Close()

	maxTokens := int64(42)
	topP := 0.9
	adapter := New("gpt-test", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := core.Chat(context.Background(), core.TextOptions{
		Adapter:       adapter,
		SystemPrompts: []string{"Be brief."},
		Messages: []core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "hi"},
		},
		MaxTokens: &maxTokens,
		TopP:      &topP,
		Metadata:  map[string]any{"trace": "abc"},
		ModelOptions: map[string]any{
			"responseFormat": map[string]any{"type": "json_object"},
		},
	})
	if err != nil {
		t.Fatalf("chat returned error: %v", err)
	}
	if result.Text != "hello" {
		t.Fatalf("unexpected text: %q", result.Text)
	}
	if request["max_completion_tokens"].(float64) != 42 {
		t.Fatalf("max tokens not mapped to chat completions: %#v", request)
	}
	if request["top_p"].(float64) != 0.9 {
		t.Fatalf("top_p not forwarded: %#v", request)
	}
	if request["response_format"] == nil {
		t.Fatalf("modelOptions responseFormat was not converted: %#v", request)
	}
	messages := request["messages"].([]any)
	if messages[0].(map[string]any)["role"] != "system" {
		t.Fatalf("system prompt was not prepended: %#v", messages)
	}
}

func TestResponsesEndpointReceivesCommonAndModelOptions(t *testing.T) {
	t.Parallel()

	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"completed","output_text":"hello","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}`))
	}))
	defer server.Close()

	maxTokens := int64(42)
	temperature := 0.2
	adapter := New("gpt-test", WithAPIKey("test-key"), WithBaseURL(server.URL), WithResponsesAPI())
	result, err := core.Chat(context.Background(), core.TextOptions{
		Adapter:       adapter,
		SystemPrompts: []string{"Be brief."},
		Messages: []core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "hi"},
		},
		MaxTokens:   &maxTokens,
		Temperature: &temperature,
		ModelOptions: map[string]any{
			"reasoning": map[string]any{"effort": "low"},
		},
	})
	if err != nil {
		t.Fatalf("chat returned error: %v", err)
	}
	if result.Text != "hello" {
		t.Fatalf("unexpected text: %q", result.Text)
	}
	if request["max_output_tokens"].(float64) != 42 {
		t.Fatalf("max tokens not mapped to responses: %#v", request)
	}
	if request["instructions"] != "Be brief." {
		t.Fatalf("system prompts not mapped to instructions: %#v", request)
	}
	if request["reasoning"] == nil {
		t.Fatalf("modelOptions reasoning was not forwarded: %#v", request)
	}
}
