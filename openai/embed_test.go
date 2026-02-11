package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/m43i/go-ai/core"
)

func TestEmbed(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("unexpected method %q", r.Method)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("unexpected authorization header %q", got)
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read request body: %v", err)
		}

		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("decode request body: %v", err)
		}

		if payload["model"] != "text-embedding-3-small" {
			t.Fatalf("unexpected model payload: %#v", payload["model"])
		}
		if payload["input"] != "hello world" {
			t.Fatalf("unexpected input payload: %#v", payload["input"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1,0.2,0.3]}],"usage":{"prompt_tokens":3,"total_tokens":3}}`))
	}))
	defer server.Close()

	adapter := New("test-key", "text-embedding-3-small", WithBaseURL(server.URL), WithHTTPClient(server.Client()))

	result, err := adapter.Embed(context.Background(), &core.EmbedParams{Input: "hello world"})
	if err != nil {
		t.Fatalf("embed returned error: %v", err)
	}

	if len(result.Embedding) != 3 {
		t.Fatalf("expected embedding length 3, got %d", len(result.Embedding))
	}
	if result.Embedding[0] != 0.1 || result.Embedding[1] != 0.2 || result.Embedding[2] != 0.3 {
		t.Fatalf("unexpected embedding values: %#v", result.Embedding)
	}
	if result.Usage == nil || result.Usage.PromptTokens != 3 || result.Usage.TotalTokens != 3 || result.Usage.CompletionTokens != 0 {
		t.Fatalf("unexpected usage: %#v", result.Usage)
	}
}

func TestEmbedManyOrdersByIndex(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read request body: %v", err)
		}

		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("decode request body: %v", err)
		}

		inputs, ok := payload["input"].([]any)
		if !ok || len(inputs) != 2 {
			t.Fatalf("unexpected input payload: %#v", payload["input"])
		}
		if payload["dimensions"] != float64(2) {
			t.Fatalf("unexpected dimensions payload: %#v", payload["dimensions"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"index":1,"embedding":[2.1,2.2]},{"index":0,"embedding":[1.1,1.2]}],"usage":{"prompt_tokens":4,"total_tokens":4}}`))
	}))
	defer server.Close()

	dimensions := int64(2)
	adapter := New("test-key", "text-embedding-3-small", WithBaseURL(server.URL), WithHTTPClient(server.Client()))

	result, err := adapter.EmbedMany(context.Background(), &core.EmbedManyParams{
		Inputs:     []string{"first", "second"},
		Dimensions: &dimensions,
	})
	if err != nil {
		t.Fatalf("embed many returned error: %v", err)
	}

	if len(result.Embeddings) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(result.Embeddings))
	}
	if result.Embeddings[0][0] != 1.1 || result.Embeddings[0][1] != 1.2 {
		t.Fatalf("unexpected first embedding: %#v", result.Embeddings[0])
	}
	if result.Embeddings[1][0] != 2.1 || result.Embeddings[1][1] != 2.2 {
		t.Fatalf("unexpected second embedding: %#v", result.Embeddings[1])
	}
}

func TestEmbedValidatesInput(t *testing.T) {
	t.Parallel()

	adapter := New("test-key", "text-embedding-3-small")
	_, err := adapter.Embed(context.Background(), &core.EmbedParams{Input: "   "})
	if err == nil {
		t.Fatal("expected validation error for empty input")
	}
	if !strings.Contains(err.Error(), "input is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}
