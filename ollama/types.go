package ollama

import "encoding/json"

type chatRequest struct {
	Model    string          `json:"model"`
	Messages []message       `json:"messages"`
	Tools    []tool          `json:"tools,omitempty"`
	Format   json.RawMessage `json:"format,omitempty"`
	Stream   *bool           `json:"stream,omitempty"`
	Think    any             `json:"think,omitempty"`
	Options  map[string]any  `json:"options,omitempty"`
}

type message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	Thinking   string     `json:"thinking,omitempty"`
	Images     []string   `json:"images,omitempty"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolName   string     `json:"tool_name,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type tool struct {
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
}

type toolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type toolCall struct {
	ID       string           `json:"id,omitempty"`
	Function toolCallFunction `json:"function"`
}

type toolCallFunction struct {
	Index     int    `json:"index,omitempty"`
	Name      string `json:"name"`
	Arguments any    `json:"arguments"`
}

type chatResponse struct {
	Model              string  `json:"model"`
	CreatedAt          string  `json:"created_at,omitempty"`
	Message            message `json:"message"`
	Done               bool    `json:"done"`
	DoneReason         string  `json:"done_reason,omitempty"`
	TotalDuration      int64   `json:"total_duration,omitempty"`
	LoadDuration       int64   `json:"load_duration,omitempty"`
	PromptEvalCount    int64   `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64   `json:"prompt_eval_duration,omitempty"`
	EvalCount          int64   `json:"eval_count,omitempty"`
	EvalDuration       int64   `json:"eval_duration,omitempty"`
}

type embedRequest struct {
	Model      string `json:"model"`
	Input      any    `json:"input"`
	Dimensions *int64 `json:"dimensions,omitempty"`
}

type embedResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float64 `json:"embeddings"`
	TotalDuration   int64       `json:"total_duration,omitempty"`
	LoadDuration    int64       `json:"load_duration,omitempty"`
	PromptEvalCount int64       `json:"prompt_eval_count,omitempty"`
}
