package core

import (
	"context"
	"errors"
	"fmt"
)

// Chat sends a non-streaming chat request through the provided adapter.
//
// Preferred usage is to use core and add a provider adapter there; this
// helper exists for direct adapter calls.
func Chat(ctx context.Context, request any, params ...*ChatParams) (*ChatResult, error) {
	adapter, chatParams, err := resolveTextRequest(request, params...)
	if err != nil {
		return nil, err
	}
	return adapter.Chat(ctx, chatParams)
}

// ChatStream sends a streaming chat request through the provided adapter.
//
// Preferred usage is to use core and add a provider adapter there; this
// helper exists for direct adapter calls.
func ChatStream(ctx context.Context, request any, params ...*ChatParams) (<-chan StreamChunk, error) {
	adapter, chatParams, err := resolveTextRequest(request, params...)
	if err != nil {
		return nil, err
	}
	return adapter.ChatStream(ctx, chatParams)
}

func resolveTextRequest(request any, params ...*ChatParams) (TextAdapter, *ChatParams, error) {
	switch typed := request.(type) {
	case TextAdapter:
		if typed == nil {
			return nil, nil, errors.New("core: text adapter is required")
		}
		if len(params) == 0 {
			return typed, nil, nil
		}
		if len(params) > 1 {
			return nil, nil, errors.New("core: only one ChatParams value is supported")
		}
		return typed, params[0], nil

	case TextOptions:
		if typed.Adapter == nil {
			return nil, nil, errors.New("core: text options adapter is required")
		}
		if len(params) > 0 {
			return nil, nil, errors.New("core: ChatParams cannot be combined with TextOptions")
		}
		return typed.Adapter, (&typed).chatParams(), nil

	case *TextOptions:
		if typed == nil {
			return nil, nil, errors.New("core: text options are required")
		}
		if typed.Adapter == nil {
			return nil, nil, errors.New("core: text options adapter is required")
		}
		if len(params) > 0 {
			return nil, nil, errors.New("core: ChatParams cannot be combined with TextOptions")
		}
		return typed.Adapter, typed.chatParams(), nil
	}

	return nil, nil, fmt.Errorf("core: unsupported text request type %T", request)
}

// Embed creates a single embedding vector through the provided adapter.
//
// Preferred usage is to use core and add a provider adapter there; this
// helper exists for direct adapter calls.
func Embed(ctx context.Context, adapter EmbeddingAdapter, params *EmbedParams) (*EmbedResult, error) {
	return adapter.Embed(ctx, params)
}

// EmbedMany creates embedding vectors for multiple inputs through the provided adapter.
//
// Preferred usage is to use core and add a provider adapter there; this
// helper exists for direct adapter calls.
func EmbedMany(ctx context.Context, adapter EmbeddingAdapter, params *EmbedManyParams) (*EmbedManyResult, error) {
	return adapter.EmbedMany(ctx, params)
}

// GenerateImage creates images through the provided adapter.
//
// Preferred usage is to use core and add a provider adapter there; this
// helper exists for direct adapter calls.
func GenerateImage(ctx context.Context, adapter ImageAdapter, params *ImageParams) (*ImageResult, error) {
	return adapter.GenerateImage(ctx, params)
}

// Transcribe converts audio to text through the provided adapter.
//
// Preferred usage is to use core and add a provider adapter there; this
// helper exists for direct adapter calls.
func Transcribe(ctx context.Context, adapter TranscriptionAdapter, params *TranscriptionParams) (*TranscriptionResult, error) {
	return adapter.Transcribe(ctx, params)
}
