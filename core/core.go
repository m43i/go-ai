package core

import "context"

// Chat sends a non-streaming chat request through the provided adapter.
//
// Preferred usage is to use core and add a provider adapter there; this
// helper exists for direct adapter calls.
func Chat(ctx context.Context, adapter TextAdapter, params *ChatParams) (*ChatResult, error) {
	return adapter.Chat(ctx, params)
}

// ChatStream sends a streaming chat request through the provided adapter.
//
// Preferred usage is to use core and add a provider adapter there; this
// helper exists for direct adapter calls.
func ChatStream(ctx context.Context, adapter TextAdapter, params *ChatParams) (<-chan StreamChunk, error) {
	return adapter.ChatStream(ctx, params)
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
