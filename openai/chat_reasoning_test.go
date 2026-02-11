package openai

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
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"choices": [{
				"message": {
					"content": "hello",
					"reasoning_content": "step by step reasoning"
				},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 12,
				"completion_tokens": 7,
				"total_tokens": 19,
				"prompt_tokens_details": {
					"cached_tokens": 3
				},
				"completion_tokens_details": {
					"reasoning_tokens": 2,
					"accepted_prediction_tokens": 1
				}
			}
		}`))
	}))
	defer server.Close()

	adapter := New("test-key", "gpt-4o-mini", WithBaseURL(server.URL), WithHTTPClient(server.Client()))

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
	if strings.TrimSpace(result.Reasoning) != "step by step reasoning" {
		t.Fatalf("unexpected reasoning: %q", result.Reasoning)
	}
	if result.Usage == nil {
		t.Fatal("expected usage to be set")
	}
	if result.Usage.ReasoningTokens != 2 {
		t.Fatalf("expected reasoning tokens 2, got %d", result.Usage.ReasoningTokens)
	}
	if result.Usage.Details == nil {
		t.Fatal("expected usage details to be set")
	}
	if result.Usage.Details["cached_prompt_tokens"] != 3 {
		t.Fatalf("expected cached_prompt_tokens=3, got %d", result.Usage.Details["cached_prompt_tokens"])
	}
	if result.Usage.Details["accepted_prediction_tokens"] != 1 {
		t.Fatalf("expected accepted_prediction_tokens=1, got %d", result.Usage.Details["accepted_prediction_tokens"])
	}
}

func TestChatStreamIncludesReasoningAndUsageDetails(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http flusher")
		}

		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"first thought\"},\"finish_reason\":\"\"}]}\n\n")
		flusher.Flush()
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"\"}]}\n\n")
		flusher.Flush()
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"second thought\"},\"finish_reason\":\"\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3,\"completion_tokens_details\":{\"reasoning_tokens\":2}}}\n\n")
		flusher.Flush()
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	adapter := New("test-key", "gpt-4o-mini", WithBaseURL(server.URL), WithHTTPClient(server.Client()))

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
	if reasoningChunks != 2 {
		t.Fatalf("expected 2 reasoning chunks, got %d", reasoningChunks)
	}
	if strings.TrimSpace(reasoning) != "first thought\nsecond thought" {
		t.Fatalf("unexpected reasoning chunk aggregation: %q", reasoning)
	}
	if strings.TrimSpace(done.Reasoning) != "first thought\nsecond thought" {
		t.Fatalf("unexpected stream reasoning: %q", done.Reasoning)
	}
	if done.Usage == nil {
		t.Fatal("expected stream usage on done chunk")
	}
	if done.Usage.ReasoningTokens != 2 {
		t.Fatalf("expected stream reasoning tokens=2, got %d", done.Usage.ReasoningTokens)
	}
}
