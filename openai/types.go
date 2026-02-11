package openai

import "encoding/json"

type chatCompletionRequest struct {
	Model               string        `json:"model"`
	Messages            []chatMessage `json:"messages"`
	Tools               []chatTool    `json:"tools,omitempty"`
	ToolChoice          string        `json:"tool_choice,omitempty"`
	ResponseFormat      any           `json:"response_format,omitempty"`
	MaxCompletionTokens *int64        `json:"max_completion_tokens,omitempty"`
	Temperature         *float64      `json:"temperature,omitempty"`
	ReasoningEffort     string        `json:"reasoning_effort,omitempty"`
	Stream              bool          `json:"stream,omitempty"`
}

type chatMessage struct {
	Role       string         `json:"role"`
	Content    any            `json:"content,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
}

type chatContentPart struct {
	Type       string          `json:"type"`
	Text       string          `json:"text,omitempty"`
	ImageURL   *chatImageURL   `json:"image_url,omitempty"`
	InputAudio *chatInputAudio `json:"input_audio,omitempty"`
}

type chatImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type chatInputAudio struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

type chatTool struct {
	Type     string           `json:"type"`
	Function chatToolFunction `json:"function"`
}

type chatToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type chatToolCall struct {
	ID       string               `json:"id"`
	Type     string               `json:"type"`
	Function chatToolCallFunction `json:"function"`
}

type chatToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatCompletionResponse struct {
	Choices    []chatChoice      `json:"choices"`
	Usage      *usage            `json:"usage,omitempty"`
	RawChoices []json.RawMessage `json:"-"`
}

type chatChoice struct {
	Message      chatResponseMessage `json:"message"`
	Text         string              `json:"text,omitempty"`
	DeltaText    string              `json:"delta_text,omitempty"`
	Reasoning    string              `json:"reasoning_content,omitempty"`
	FinishReason string              `json:"finish_reason"`
}

type chatResponseMessage struct {
	Content          json.RawMessage `json:"content"`
	ToolCalls        []chatToolCall  `json:"tool_calls"`
	ReasoningContent string          `json:"reasoning_content,omitempty"`
	Refusal          string          `json:"refusal,omitempty"`
}

type streamEvent struct {
	Choices []streamChoice `json:"choices"`
	Usage   *usage         `json:"usage,omitempty"`
}

type streamChoice struct {
	Delta        streamDelta `json:"delta"`
	Text         string      `json:"text,omitempty"`
	Reasoning    string      `json:"reasoning_content,omitempty"`
	FinishReason string      `json:"finish_reason"`
}

type streamDelta struct {
	Content          json.RawMessage `json:"content"`
	Text             string          `json:"text,omitempty"`
	ReasoningContent string          `json:"reasoning_content,omitempty"`
	Refusal          string          `json:"refusal,omitempty"`
}

type usage struct {
	PromptTokens            int64                   `json:"prompt_tokens"`
	CompletionTokens        int64                   `json:"completion_tokens"`
	TotalTokens             int64                   `json:"total_tokens"`
	PromptTokensDetails     *promptTokensDetails    `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *completionTokensDetail `json:"completion_tokens_details,omitempty"`
}

type promptTokensDetails struct {
	CachedTokens int64 `json:"cached_tokens,omitempty"`
	AudioTokens  int64 `json:"audio_tokens,omitempty"`
}

type completionTokensDetail struct {
	ReasoningTokens          int64 `json:"reasoning_tokens,omitempty"`
	AudioTokens              int64 `json:"audio_tokens,omitempty"`
	AcceptedPredictionTokens int64 `json:"accepted_prediction_tokens,omitempty"`
	RejectedPredictionTokens int64 `json:"rejected_prediction_tokens,omitempty"`
}
