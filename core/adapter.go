package core

import "context"

// TextAdapter defines chat capabilities for a model provider adapter.
//
// Preferred usage is to use core and add a provider adapter there. These
// interfaces stay available for direct adapter calls when needed.
type TextAdapter interface {
	Chat(ctx context.Context, params *ChatParams) (*ChatResult, error)
	ChatStream(ctx context.Context, params *ChatParams) (<-chan StreamChunk, error)
}

// EmbeddingAdapter defines embedding capabilities for a model provider adapter.
//
// Preferred usage is to use core and add a provider adapter there. This
// interface stays available for direct adapter calls when needed.
type EmbeddingAdapter interface {
	Embed(ctx context.Context, params *EmbedParams) (*EmbedResult, error)
	EmbedMany(ctx context.Context, params *EmbedManyParams) (*EmbedManyResult, error)
}

// ImageAdapter defines image generation capabilities for a model provider adapter.
//
// Preferred usage is to use core and add a provider adapter there. This
// interface stays available for direct adapter calls when needed.
type ImageAdapter interface {
	GenerateImage(ctx context.Context, params *ImageParams) (*ImageResult, error)
}

// TranscriptionAdapter defines audio transcription capabilities for a model provider adapter.
//
// Preferred usage is to use core and add a provider adapter there. This
// interface stays available for direct adapter calls when needed.
type TranscriptionAdapter interface {
	Transcribe(ctx context.Context, params *TranscriptionParams) (*TranscriptionResult, error)
}
