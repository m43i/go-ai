package core

import (
	"context"
	"testing"
)

type embeddingAdapterStub struct {
	embedFn     func(context.Context, *EmbedParams) (*EmbedResult, error)
	embedManyFn func(context.Context, *EmbedManyParams) (*EmbedManyResult, error)
}

func (s embeddingAdapterStub) Embed(ctx context.Context, params *EmbedParams) (*EmbedResult, error) {
	return s.embedFn(ctx, params)
}

func (s embeddingAdapterStub) EmbedMany(ctx context.Context, params *EmbedManyParams) (*EmbedManyResult, error) {
	return s.embedManyFn(ctx, params)
}

func TestEmbed(t *testing.T) {
	expected := &EmbedResult{Embedding: []float64{1, 2, 3}}
	adapter := embeddingAdapterStub{
		embedFn: func(_ context.Context, params *EmbedParams) (*EmbedResult, error) {
			if params == nil || params.Input != "hello" {
				t.Fatalf("unexpected params: %#v", params)
			}
			return expected, nil
		},
		embedManyFn: func(context.Context, *EmbedManyParams) (*EmbedManyResult, error) {
			t.Fatal("embed many should not be called")
			return nil, nil
		},
	}

	result, err := Embed(context.Background(), adapter, &EmbedParams{Input: "hello"})
	if err != nil {
		t.Fatalf("embed returned error: %v", err)
	}
	if result != expected {
		t.Fatalf("expected result pointer %#v, got %#v", expected, result)
	}
}

func TestEmbedMany(t *testing.T) {
	expected := &EmbedManyResult{Embeddings: [][]float64{{1, 2}, {3, 4}}}
	adapter := embeddingAdapterStub{
		embedFn: func(context.Context, *EmbedParams) (*EmbedResult, error) {
			t.Fatal("embed should not be called")
			return nil, nil
		},
		embedManyFn: func(_ context.Context, params *EmbedManyParams) (*EmbedManyResult, error) {
			if params == nil || len(params.Inputs) != 2 {
				t.Fatalf("unexpected params: %#v", params)
			}
			return expected, nil
		},
	}

	result, err := EmbedMany(context.Background(), adapter, &EmbedManyParams{Inputs: []string{"a", "b"}})
	if err != nil {
		t.Fatalf("embed many returned error: %v", err)
	}
	if result != expected {
		t.Fatalf("expected result pointer %#v, got %#v", expected, result)
	}
}
