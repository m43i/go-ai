package claude

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/m43i/go-ai/core"
)

func TestChatStreamReasoningUsesIncrementalDelta(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/messages" {
			http.NotFound(w, r)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintln(w, "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"The\"}}")
		_, _ = fmt.Fprintln(w)
		_, _ = fmt.Fprintln(w, "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"The user\"}}")
		_, _ = fmt.Fprintln(w)
		_, _ = fmt.Fprintln(w, "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"The user asks\"}}")
		_, _ = fmt.Fprintln(w)
		_, _ = fmt.Fprintln(w, "data: {\"type\":\"message_stop\"}")
		_, _ = fmt.Fprintln(w)
	}))
	defer server.Close()

	adapter := New("claude-test", WithAPIKey("test-key"), WithBaseURL(server.URL))
	stream, err := adapter.ChatStream(context.Background(), &core.ChatParams{Messages: []core.MessageUnion{core.TextMessagePart{Role: core.RoleUser, Content: "Hi"}}})
	if err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}

	var deltas []string
	var snapshots []string
	doneReasoning := ""

	for chunk := range stream {
		switch chunk.Type {
		case core.StreamChunkReasoning:
			deltas = append(deltas, chunk.Delta)
			snapshots = append(snapshots, chunk.Reasoning)
		case core.StreamChunkError:
			t.Fatalf("unexpected chunk error: %s", chunk.Error)
		case core.StreamChunkDone:
			doneReasoning = chunk.Reasoning
		}
	}

	if !reflect.DeepEqual(deltas, []string{"The", " user", " asks"}) {
		t.Fatalf("unexpected reasoning deltas: %#v", deltas)
	}
	if !reflect.DeepEqual(snapshots, []string{"The", "The user", "The user asks"}) {
		t.Fatalf("unexpected reasoning snapshots: %#v", snapshots)
	}
	if doneReasoning != "The user asks" {
		t.Fatalf("unexpected final reasoning: %q", doneReasoning)
	}
}
