package claude

type messageRequest struct {
	Model       string      `json:"model"`
	System      string      `json:"system,omitempty"`
	Messages    []message   `json:"messages"`
	MaxTokens   *int64      `json:"max_tokens,omitempty"`
	Temperature *float64    `json:"temperature,omitempty"`
	Tools       []tool      `json:"tools,omitempty"`
	ToolChoice  *toolChoice `json:"tool_choice,omitempty"`
	Stream      bool        `json:"stream,omitempty"`
}

type message struct {
	Role    string         `json:"role"`
	Content []contentBlock `json:"content"`
}

type contentBlock struct {
	Type      string       `json:"type"`
	Text      string       `json:"text,omitempty"`
	Thinking  string       `json:"thinking,omitempty"`
	Signature string       `json:"signature,omitempty"`
	Source    *mediaSource `json:"source,omitempty"`
	ID        string       `json:"id,omitempty"`
	Name      string       `json:"name,omitempty"`
	Input     any          `json:"input,omitempty"`
	ToolUseID string       `json:"tool_use_id,omitempty"`
	Content   any          `json:"content,omitempty"`
}

type mediaSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema,omitempty"`
}

type toolChoice struct {
	Type string `json:"type"`
}

type messageResponse struct {
	ID         string         `json:"id"`
	Role       string         `json:"role"`
	Content    []contentBlock `json:"content"`
	StopReason string         `json:"stop_reason"`
	Usage      *usage         `json:"usage,omitempty"`
}

type streamEvent struct {
	Type  string       `json:"type"`
	Delta *streamDelta `json:"delta,omitempty"`
	Error *streamError `json:"error,omitempty"`
	Usage *usage       `json:"usage,omitempty"`
}

type streamDelta struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Thinking string `json:"thinking,omitempty"`
}

type streamError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type usage struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens,omitempty"`
}
