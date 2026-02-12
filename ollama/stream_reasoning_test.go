package ollama

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/m43i/go-ai/core"
)

func TestChatStreamReasoningPreservesSpacesInDelta(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/chat" {
			http.NotFound(w, r)
			return
		}

		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprintln(w, "{\"message\":{\"thinking\":\"The\"},\"done\":false}")
		_, _ = fmt.Fprintln(w, "{\"message\":{\"thinking\":\" user\"},\"done\":false}")
		_, _ = fmt.Fprintln(w, "{\"message\":{\"thinking\":\" asks\"},\"done\":false}")
		_, _ = fmt.Fprintln(w, "{\"message\":{\"content\":\"hello\"},\"done\":false}")
		_, _ = fmt.Fprintln(w, "{\"message\":{\"content\":\" world\"},\"done\":true,\"done_reason\":\"stop\"}")
	}))
	defer server.Close()

	adapter := New("ollama-test", WithBaseURL(server.URL))
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
