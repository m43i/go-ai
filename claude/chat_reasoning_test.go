package claude

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/m43i/go-ai/core"
)

func TestChatIncludesReasoningAndUsageDetails(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/messages" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "msg_123",
			"role": "assistant",
			"content": [
				{"type": "thinking", "thinking": "first principles"},
				{"type": "text", "text": "hello"}
			],
			"stop_reason": "end_turn",
			"usage": {
				"input_tokens": 10,
				"output_tokens": 4,
				"cache_creation_input_tokens": 2,
				"cache_read_input_tokens": 1
			}
		}`))
	}))
	defer server.Close()

	adapter := New("test-key", "claude-3-5-haiku-latest", WithBaseURL(server.URL), WithHTTPClient(server.Client()))

	result, err := adapter.Chat(context.Background(), &core.ChatParams{
		Messages: []core.MessageUnion{core.TextMessagePart{Role: core.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("chat returned error: %v", err)
	}

	if result == nil {
		t.Fatal("chat returned nil result")
	}
	if strings.TrimSpace(result.Text) != "hello" {
		t.Fatalf("unexpected assistant text: %q", result.Text)
	}
	if strings.TrimSpace(result.Reasoning) != "first principles" {
		t.Fatalf("unexpected reasoning: %q", result.Reasoning)
	}
	if result.Usage == nil {
		t.Fatal("expected usage to be set")
	}
	if result.Usage.Details == nil {
		t.Fatal("expected usage details to be set")
	}
	if result.Usage.Details["cache_creation_input_tokens"] != 2 {
		t.Fatalf("expected cache_creation_input_tokens=2, got %d", result.Usage.Details["cache_creation_input_tokens"])
	}
	if result.Usage.Details["cache_read_input_tokens"] != 1 {
		t.Fatalf("expected cache_read_input_tokens=1, got %d", result.Usage.Details["cache_read_input_tokens"])
	}
}

func TestChatStreamIncludesReasoningAndUsageDetails(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/messages" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http flusher")
		}

		_, _ = fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"reasoning one\"}}\n\n")
		flusher.Flush()
		_, _ = fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n")
		flusher.Flush()
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message_stop\",\"usage\":{\"input_tokens\":2,\"output_tokens\":3,\"cache_read_input_tokens\":1}}\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	adapter := New("test-key", "claude-3-5-haiku-latest", WithBaseURL(server.URL), WithHTTPClient(server.Client()))

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	stream, err := adapter.ChatStream(ctx, &core.ChatParams{
		Messages: []core.MessageUnion{core.TextMessagePart{Role: core.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("chat stream returned error: %v", err)
	}

	content := ""
	reasoning := ""
	reasoningChunks := 0
	var done *core.StreamChunk

	for chunk := range stream {
		switch chunk.Type {
		case core.StreamChunkError:
			t.Fatalf("unexpected stream error: %s", chunk.Error)
		case core.StreamChunkReasoning:
			reasoningChunks++
			reasoning = chunk.Reasoning
		case core.StreamChunkContent:
			content = chunk.Content
		case core.StreamChunkDone:
			copy := chunk
			done = &copy
		}
	}

	if content != "hello" {
		t.Fatalf("unexpected final stream content: %q", content)
	}
	if done == nil {
		t.Fatal("expected done chunk")
	}
	if reasoningChunks != 1 {
		t.Fatalf("expected 1 reasoning chunk, got %d", reasoningChunks)
	}
	if strings.TrimSpace(reasoning) != "reasoning one" {
		t.Fatalf("unexpected stream reasoning chunk: %q", reasoning)
	}
	if strings.TrimSpace(done.Reasoning) != "reasoning one" {
		t.Fatalf("unexpected stream reasoning: %q", done.Reasoning)
	}
	if done.Usage == nil {
		t.Fatal("expected stream usage on done chunk")
	}
	if done.Usage.Details == nil || done.Usage.Details["cache_read_input_tokens"] != 1 {
		t.Fatalf("expected stream usage cache_read_input_tokens=1, got %#v", done.Usage)
	}
}
