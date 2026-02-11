package openai

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/m43i/go-ai/core"
)

const embedLiveTimeout = 120 * time.Second

func TestPublicEmbedLive(t *testing.T) {
	adapter := requireEmbeddingLiveAdapter(t)

	t.Run("Embed", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), embedLiveTimeout)
		defer cancel()

		result, err := core.Embed(ctx, adapter, &core.EmbedParams{
			Input: "Return a stable embedding for this sentence.",
		})
		if err != nil {
			t.Fatalf("live embed failed: %v", err)
		}
		if result == nil {
			t.Fatal("live embed returned nil result")
		}
		if len(result.Embedding) == 0 {
			t.Fatal("live embed returned empty embedding")
		}
	})

	t.Run("EmbedMany", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), embedLiveTimeout)
		defer cancel()

		result, err := core.EmbedMany(ctx, adapter, &core.EmbedManyParams{
			Inputs: []string{
				"first embedding input",
				"second embedding input",
			},
		})
		if err != nil {
			t.Fatalf("live embed many failed: %v", err)
		}
		if result == nil {
			t.Fatal("live embed many returned nil result")
		}
		if len(result.Embeddings) != 2 {
			t.Fatalf("expected 2 embeddings, got %d", len(result.Embeddings))
		}
		if len(result.Embeddings[0]) == 0 || len(result.Embeddings[1]) == 0 {
			t.Fatalf("expected non-empty embeddings, got %#v", result.Embeddings)
		}
	})
}

func requireEmbeddingLiveAdapter(t *testing.T) *Adapter {
	t.Helper()

	apiKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set; skipping live OpenAI embedding tests")
	}

	model := strings.TrimSpace(os.Getenv("GOAI_OPENAI_EMBED_MODEL"))
	if model == "" {
		model = "text-embedding-3-small"
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
