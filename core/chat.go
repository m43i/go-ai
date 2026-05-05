package core

type MessageUnion interface {
	isMessageUnion()
}

const (
	RoleSystem     = "system"
	RoleUser       = "user"
	RoleAssistant  = "assistant"
	RoleToolCall   = "tool_call"
	RoleToolResult = "tool_result"

	StreamChunkContent    = "content"
	StreamChunkReasoning  = "reasoning"
	StreamChunkToolCall   = "tool_call"
	StreamChunkToolResult = "tool_result"
	StreamChunkDone       = "done"
	StreamChunkError      = "error"
)

type TextMessagePart struct {
	Role    string
	Content string
}

func (TextMessagePart) isMessageUnion() {}

type ContentPart interface {
	isContentPart()
}

type TextPart struct {
	Text string
}

func (TextPart) isContentPart() {}

type ImagePart struct {
	Source   Source
	Metadata map[string]any
}

func (ImagePart) isContentPart() {}

type AudioPart struct {
	Source   Source
	Metadata map[string]any
}

func (AudioPart) isContentPart() {}

type DocumentPart struct {
	Source   Source
	Metadata map[string]any
}

func (DocumentPart) isContentPart() {}

type Source interface {
	isSource()
}

type DataSource struct {
	Data     string
	MimeType string
}

func (DataSource) isSource() {}

type URLSource struct {
	URL      string
	MimeType string
}

func (URLSource) isSource() {}

type ContentMessagePart struct {
	Role  string
	Parts []ContentPart
}

func (ContentMessagePart) isMessageUnion() {}

type ToolCallMessagePart struct {
	Role      string
	ToolCalls []ToolCall
}

func (ToolCallMessagePart) isMessageUnion() {}

type AssistantToolCallMessagePart = ToolCallMessagePart

type ToolResultMessagePart struct {
	Role       string
	ToolCallID string
	Name       string
	Content    string
}

func (ToolResultMessagePart) isMessageUnion() {}

type Usage struct {
	PromptTokens     int64
	CompletionTokens int64
	TotalTokens      int64
	ReasoningTokens  int64
	Details          map[string]int64
}

type StreamChunk struct {
	Type         string
	Role         string
	Delta        string
	Content      string
	Reasoning    string
	ToolCall     *ToolCall
	ToolCallID   string
	FinishReason string
	Usage        *Usage
	Error        string
}

type ChatResult struct {
	Text      string
	Reasoning string
	Messages  []MessageUnion
	ToolCalls []ToolCall

	FinishReason string
	Usage        *Usage
}

type ChatParams struct {
	Tools  []ToolUnion
	Output *Schema

	SystemPrompts []string
	Messages      []MessageUnion

	// ModelOptions holds provider-specific options that are passed through to the
	// selected adapter. Prefer common fields such as Temperature and MaxTokens
	// when they exist; use ModelOptions for provider-specific escape hatches.
	ModelOptions map[string]any
	Metadata     map[string]any

	MaxTokens       *int64
	MaxOutputTokens *int64
	Temperature     *float64
	TopP            *float64
	Thinking        string
	ReasoningEffort string

	MaxAgenticLoops int32
	MaxLength       int64
}

// TextOptions is the minimal text interface: common options live
// here, while provider-specific options are supplied through ModelOptions.
type TextOptions struct {
	Adapter TextAdapter

	Tools  []ToolUnion
	Output *Schema

	SystemPrompts []string
	Messages      []MessageUnion

	ModelOptions map[string]any
	Metadata     map[string]any

	MaxTokens       *int64
	MaxOutputTokens *int64
	Temperature     *float64
	TopP            *float64
	Thinking        string
	ReasoningEffort string

	MaxAgenticLoops int32
	MaxLength       int64
}

func (o *TextOptions) chatParams() *ChatParams {
	if o == nil {
		return nil
	}

	return &ChatParams{
		Tools:           o.Tools,
		Output:          o.Output,
		SystemPrompts:   o.SystemPrompts,
		Messages:        o.Messages,
		ModelOptions:    o.ModelOptions,
		Metadata:        o.Metadata,
		MaxTokens:       o.MaxTokens,
		MaxOutputTokens: o.MaxOutputTokens,
		Temperature:     o.Temperature,
		TopP:            o.TopP,
		Thinking:        o.Thinking,
		ReasoningEffort: o.ReasoningEffort,
		MaxAgenticLoops: o.MaxAgenticLoops,
		MaxLength:       o.MaxLength,
	}
}
