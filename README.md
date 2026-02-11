# go-ai

A provider-agnostic AI SDK for Go, inspired by [TanStack AI](https://tanstack.com/ai) and [Vercel AI SDK](https://sdk.vercel.ai). Zero external dependencies -- only the Go standard library.

## Features

- **Provider-agnostic** -- swap between OpenAI, Claude, and Ollama with a single line change
- **Chat completions** -- streaming and non-streaming
- **Tool calling** -- server tools (auto-executed in an agentic loop) and client tools (returned to the caller)
- **Structured output** -- build strict JSON schemas from Go structs, decode responses with generics
- **Multimodal** -- text, images, audio, and documents as message content
- **Embeddings** -- single and batch, with cosine similarity utility
- **Image generation** -- via OpenAI image models
- **Audio transcription** -- via OpenAI Whisper
- **Reasoning / thinking** -- extract chain-of-thought from reasoning models
- **Zero dependencies** -- built entirely on the Go standard library

## Supported Providers

| Provider | Chat | Streaming | Tools | Structured Output | Embeddings | Images | Transcription |
|----------|------|-----------|-------|--------------------|------------|--------|---------------|
| OpenAI   | Yes  | Yes       | Yes   | Yes                | Yes        | Yes    | Yes           |
| Claude   | Yes  | Yes       | Yes   | Yes                | --         | --     | --            |
| Ollama   | Yes  | Yes       | Yes   | Yes                | Yes        | --     | --            |

## Installation

```sh
go get github.com/m43i/go-ai
```

Requires Go 1.25.6 or later.

## Quick Start

### Basic Chat

```go
package main

import (
	"context"
	"fmt"

	"github.com/m43i/go-ai/core"
	"github.com/m43i/go-ai/openai"
)

func main() {
	adapter := openai.New("gpt-4o") // reads OPENAI_API_KEY from env

	result, err := core.Chat(context.Background(), adapter, &core.ChatParams{
		Messages: []core.MessageUnion{
			core.TextMessagePart{Role: core.RoleUser, Content: "What is the capital of France?"},
		},
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(result.Text)
}
```

### Using Claude

```go
import "github.com/m43i/go-ai/claude"

adapter := claude.New("claude-sonnet-4-20250514") // reads ANTHROPIC_API_KEY from env

result, err := core.Chat(context.Background(), adapter, &core.ChatParams{
	Messages: []core.MessageUnion{
		core.TextMessagePart{Role: core.RoleUser, Content: "Explain quantum computing in one paragraph."},
	},
})
```

### Using Ollama

```go
import "github.com/m43i/go-ai/ollama"

adapter := ollama.New("llama3.2") // reads OLLAMA_HOST, defaults to http://localhost:11434

result, err := core.Chat(context.Background(), adapter, &core.ChatParams{
	Messages: []core.MessageUnion{
		core.TextMessagePart{Role: core.RoleUser, Content: "Explain quantum computing in one paragraph."},
	},
})
```

### Streaming

```go
chunks, err := core.ChatStream(context.Background(), adapter, &core.ChatParams{
	Messages: []core.MessageUnion{
		core.TextMessagePart{Role: core.RoleUser, Content: "Write a haiku about Go."},
	},
})
if err != nil {
	panic(err)
}

for chunk := range chunks {
	switch chunk.Type {
	case core.StreamChunkContent:
		fmt.Print(chunk.Delta)
	case core.StreamChunkDone:
		fmt.Println("\n-- done --")
	case core.StreamChunkError:
		fmt.Println("error:", chunk.Error)
	}
}
```

### Server Tools (Agentic Loop)

Server tools are automatically executed by the adapter. The model calls the tool, the adapter runs your handler, and feeds the result back -- up to `MaxAgenticLoops` iterations (default 8).

```go
result, err := core.Chat(context.Background(), adapter, &core.ChatParams{
	Messages: []core.MessageUnion{
		core.TextMessagePart{Role: core.RoleUser, Content: "What is the weather in Berlin?"},
	},
	Tools: []core.ToolUnion{
		core.ServerTool{
			Name:        "get_weather",
			Description: "Get the current weather for a city",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
				"required": []string{"city"},
			},
			Handler: func(args any) (string, error) {
				// args is the parsed JSON arguments from the model
				return `{"temperature": "18°C", "condition": "partly cloudy"}`, nil
			},
		},
	},
})
```

### Client Tools

Client tools are not auto-executed. Instead, the adapter returns pending tool calls so your application can run them, append `ToolResultMessagePart` messages, and continue the loop.

```go
tools := []core.ToolUnion{
	core.ClientTool{
		Name:        "web_search",
		Description: "Search the web",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string"},
			},
			"required": []string{"query"},
		},
	},
	core.ClientTool{
		Name:        "ask_client",
		Description: "Hand off a question to the client application/user",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"question": map[string]any{"type": "string"},
			},
			"required": []string{"question"},
		},
	},
}

conversation := []core.MessageUnion{
	core.TextMessagePart{Role: core.RoleUser, Content: "Find the latest Go release and ask me if I want details."},
}

result, err := core.Chat(context.Background(), adapter, &core.ChatParams{
	Messages: conversation,
	Tools:    tools,
})
if err != nil {
	panic(err)
}

for len(result.ToolCalls) > 0 {
	conversation = append([]core.MessageUnion(nil), result.Messages...)

	for _, call := range result.ToolCalls {
		toolOutput := `{"error":"unknown tool"}`

		switch call.Name {
		case "web_search":
			toolOutput = `{"results":["Go 1.25.6 released"]}`
		case "ask_client":
			toolOutput = `{"answer":"Yes, show me the highlights."}`
		}

		conversation = append(conversation, core.ToolResultMessagePart{
			Role:       core.RoleToolResult,
			ToolCallID: call.ID,
			Name:       call.Name,
			Content:    toolOutput,
		})
	}

	result, err = core.Chat(context.Background(), adapter, &core.ChatParams{
		Messages: conversation,
		Tools:    tools,
	})
	if err != nil {
		panic(err)
	}
}

fmt.Println(result.Text)
```

### Structured Output

Build a strict JSON schema from a Go struct and decode the response with generics.

```go
type Sentiment struct {
	Sentiment  string  `json:"sentiment"`
	Confidence float64 `json:"confidence"`
	Reasoning  string  `json:"reasoning"`
}

schema, err := core.NewSchema("sentiment_analysis", Sentiment{})
if err != nil {
	panic(err)
}

result, err := core.Chat(context.Background(), adapter, &core.ChatParams{
	Messages: []core.MessageUnion{
		core.TextMessagePart{Role: core.RoleUser, Content: "Analyze the sentiment: 'Go is a great language!'"},
	},
	Output: &schema,
})
if err != nil {
	panic(err)
}

sentiment, err := core.DecodeLast[Sentiment](result)
if err != nil {
	panic(err)
}

fmt.Printf("Sentiment: %s (%.0f%% confidence)\n", sentiment.Sentiment, sentiment.Confidence*100)
```

### Multimodal Content

Send images, audio, or documents alongside text.

```go
result, err := core.Chat(context.Background(), adapter, &core.ChatParams{
	Messages: []core.MessageUnion{
		core.ContentMessagePart{
			Role: core.RoleUser,
			Parts: []core.ContentPart{
				core.TextPart{Text: "What's in this image?"},
				core.ImagePart{
					Source: core.URLSource{
						URL:      "https://example.com/photo.jpg",
						MimeType: "image/jpeg",
					},
				},
			},
		},
	},
})
```

Or with base64 data:

```go
core.ImagePart{
	Source: core.DataSource{
		Data:     base64EncodedString,
		MimeType: "image/png",
	},
}
```

### Embeddings

```go
adapter := openai.New("text-embedding-3-small")

// Single embedding
result, err := core.Embed(context.Background(), adapter, &core.EmbedParams{
	Input: "The quick brown fox",
})

// Batch embeddings
manyResult, err := core.EmbedMany(context.Background(), adapter, &core.EmbedManyParams{
	Inputs: []string{"Hello world", "Goodbye world"},
})
```

### Image Generation

```go
adapter := openai.New("gpt-image-1")

result, err := core.GenerateImage(context.Background(), adapter, &core.ImageParams{
	Prompt: "A serene mountain landscape at sunset",
	Size:   "1024x1024",
})

for _, img := range result.Images {
	fmt.Println("Image URL:", img.URL)
	fmt.Println("Revised prompt:", img.RevisedPrompt)
}
```

### Audio Transcription

```go
adapter := openai.New("whisper-1")

audioData, _ := os.ReadFile("recording.mp3")

result, err := core.Transcribe(context.Background(), adapter, &core.TranscriptionParams{
	Audio:    audioData,
	Filename: "recording.mp3",
	Language: "en",
})

fmt.Println(result.Text)
```

### Reasoning / Thinking

Extract chain-of-thought reasoning from models that support it.

```go
result, err := core.Chat(context.Background(), adapter, &core.ChatParams{
	Messages: []core.MessageUnion{
		core.TextMessagePart{Role: core.RoleUser, Content: "Solve: what is 127 * 843?"},
	},
	ReasoningEffort: "medium", // OpenAI reasoning effort
})

fmt.Println("Reasoning:", result.Reasoning)
fmt.Println("Answer:", result.Text)
```

## Adapter Configuration

All adapters support functional options:

```go
// OpenAI
adapter := openai.New("gpt-4o",
	openai.WithAPIKey("sk-..."),
	openai.WithBaseURL("https://custom-endpoint.example.com/v1"),
	openai.WithTimeout(2 * time.Minute),
	openai.WithHTTPClient(customClient),
)

// Claude
adapter := claude.New("claude-sonnet-4-20250514",
	claude.WithAPIKey("sk-ant-..."),
	claude.WithBaseURL("https://custom-endpoint.example.com/v1"),
	claude.WithTimeout(2 * time.Minute),
	claude.WithHTTPClient(customClient),
	claude.WithAnthropicVersion("2023-06-01"),
)

// Ollama
adapter := ollama.New("llama3.2",
	ollama.WithBaseURL("http://localhost:11434"),
	ollama.WithTimeout(2 * time.Minute),
	ollama.WithHTTPClient(customClient),
	ollama.WithAPIKey("optional-remote-token"),
)
```

API keys are resolved automatically from environment variables when not provided:

- **OpenAI**: `OPENAI_API_KEY`
- **Claude**: `ANTHROPIC_API_KEY`, then `CLAUDE_API_KEY`
- **Ollama**: `OLLAMA_HOST` (base URL), optional `OLLAMA_API_KEY`

## Core Interfaces

The `core` package defines four capability interfaces. Provider adapters implement whichever capabilities they support:

```go
type TextAdapter interface {
	Chat(ctx context.Context, params *ChatParams) (*ChatResult, error)
	ChatStream(ctx context.Context, params *ChatParams) (<-chan StreamChunk, error)
}

type EmbeddingAdapter interface {
	Embed(ctx context.Context, params *EmbedParams) (*EmbedResult, error)
	EmbedMany(ctx context.Context, params *EmbedManyParams) (*EmbedManyResult, error)
}

type ImageAdapter interface {
	GenerateImage(ctx context.Context, params *ImageParams) (*ImageResult, error)
}

type TranscriptionAdapter interface {
	Transcribe(ctx context.Context, params *TranscriptionParams) (*TranscriptionResult, error)
}
```

## License

MIT
