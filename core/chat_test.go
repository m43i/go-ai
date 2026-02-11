package core

import (
	"context"
	"testing"
)

type textAdapterStub struct {
	chatFn       func(context.Context, *ChatParams) (*ChatResult, error)
	chatStreamFn func(context.Context, *ChatParams) (<-chan StreamChunk, error)
}

func (s textAdapterStub) Chat(ctx context.Context, params *ChatParams) (*ChatResult, error) {
	return s.chatFn(ctx, params)
}

func (s textAdapterStub) ChatStream(ctx context.Context, params *ChatParams) (<-chan StreamChunk, error) {
	return s.chatStreamFn(ctx, params)
}

func TestChatDelegatesToAdapter(t *testing.T) {
	expected := &ChatResult{Text: "ok"}

	adapter := textAdapterStub{
		chatFn: func(_ context.Context, params *ChatParams) (*ChatResult, error) {
			if params == nil || len(params.Messages) != 1 {
				t.Fatalf("unexpected params: %#v", params)
			}
			return expected, nil
		},
		chatStreamFn: func(context.Context, *ChatParams) (<-chan StreamChunk, error) {
			t.Fatal("chat stream should not be called")
			return nil, nil
		},
	}

	result, err := Chat(context.Background(), adapter, &ChatParams{
		Messages: []MessageUnion{TextMessagePart{Role: RoleUser, Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("chat returned error: %v", err)
	}
	if result != expected {
		t.Fatalf("expected result pointer %#v, got %#v", expected, result)
	}
}

func TestChatStreamDelegatesToAdapter(t *testing.T) {
	expected := make(chan StreamChunk, 1)
	expected <- StreamChunk{Type: StreamChunkDone, FinishReason: "stop"}
	close(expected)

	adapter := textAdapterStub{
		chatFn: func(context.Context, *ChatParams) (*ChatResult, error) {
			t.Fatal("chat should not be called")
			return nil, nil
		},
		chatStreamFn: func(_ context.Context, params *ChatParams) (<-chan StreamChunk, error) {
			if params == nil || len(params.Messages) != 1 {
				t.Fatalf("unexpected params: %#v", params)
			}
			return expected, nil
		},
	}

	stream, err := ChatStream(context.Background(), adapter, &ChatParams{
		Messages: []MessageUnion{TextMessagePart{Role: RoleUser, Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("chat stream returned error: %v", err)
	}
	if stream != expected {
		t.Fatalf("expected stream channel %#v, got %#v", expected, stream)
	}
}
