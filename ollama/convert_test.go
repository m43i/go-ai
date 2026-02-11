package ollama

import (
	"strings"
	"testing"

	"github.com/m43i/go-ai/core"
)

func TestToMessageContentMessagePartWithImageData(t *testing.T) {
	t.Parallel()

	msg := core.ContentMessagePart{
		Role: "user",
		Parts: []core.ContentPart{
			core.TextPart{Text: "Describe this image"},
			core.ImagePart{Source: core.DataSource{Data: "aGVsbG8=", MimeType: "image/png"}},
		},
	}

	result, err := toMessage(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Role != "user" {
		t.Fatalf("unexpected role: %q", result.Role)
	}
	if result.Content != "Describe this image" {
		t.Fatalf("unexpected content: %q", result.Content)
	}
	if len(result.Images) != 1 || result.Images[0] != "aGVsbG8=" {
		t.Fatalf("unexpected images: %#v", result.Images)
	}
}

func TestToMessageContentMessagePartWithImageURLFails(t *testing.T) {
	t.Parallel()

	msg := core.ContentMessagePart{
		Role: "user",
		Parts: []core.ContentPart{
			core.ImagePart{Source: core.URLSource{URL: "https://example.com/image.png"}},
		},
	}

	_, err := toMessage(msg)
	if err == nil {
		t.Fatal("expected error for image URL source")
	}
	if !strings.Contains(err.Error(), "not supported") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestToMessageToolResultMessage(t *testing.T) {
	t.Parallel()

	result, err := toMessage(core.ToolResultMessagePart{
		Role:       core.RoleToolResult,
		ToolCallID: "call_1",
		Name:       "get_weather",
		Content:    "{\"temp\": 18}",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Role != "tool" {
		t.Fatalf("expected role=tool, got %q", result.Role)
	}
	if result.ToolCallID != "call_1" {
		t.Fatalf("unexpected tool call id: %q", result.ToolCallID)
	}
	if result.ToolName != "get_weather" {
		t.Fatalf("unexpected tool name: %q", result.ToolName)
	}
}

func TestToCoreToolCallsParsesJSONStringArguments(t *testing.T) {
	t.Parallel()

	calls, err := toCoreToolCalls([]toolCall{{
		ID: "call_1",
		Function: toolCallFunction{
			Name:      "lookup",
			Arguments: `{"query":"go"}`,
		},
	}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}

	args, ok := calls[0].Arguments.(map[string]any)
	if !ok {
		t.Fatalf("expected map arguments, got %T", calls[0].Arguments)
	}
	if args["query"] != "go" {
		t.Fatalf("unexpected args: %#v", args)
	}
}

func TestToToolsRejectsDuplicateNames(t *testing.T) {
	t.Parallel()

	_, _, _, err := toTools(&core.ChatParams{
		Tools: []core.ToolUnion{
			core.ClientTool{Name: "dup"},
			core.ClientTool{Name: "dup"},
		},
	})
	if err == nil {
		t.Fatal("expected duplicate tool name error")
	}
	if !strings.Contains(err.Error(), "duplicate tool name") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestThinkValueFromThinking(t *testing.T) {
	t.Parallel()

	value := thinkValue(&core.ChatParams{Thinking: "HIGH"})
	if value != "high" {
		t.Fatalf("expected high, got %#v", value)
	}
}

func TestThinkValueFromReasoningEffort(t *testing.T) {
	t.Parallel()

	value := thinkValue(&core.ChatParams{ReasoningEffort: "medium"})
	if value != "medium" {
		t.Fatalf("expected medium, got %#v", value)
	}
}

func TestFormatFromOutputUsesSchemaObject(t *testing.T) {
	t.Parallel()

	schema := &core.Schema{
		Name:   "summary",
		Strict: true,
		Schema: map[string]any{"type": "object", "properties": map[string]any{"text": map[string]any{"type": "string"}}},
	}

	format, err := formatFromOutput(schema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(string(format), "json_schema") {
		t.Fatalf("expected raw schema object, got wrapped payload: %s", string(format))
	}
	if !strings.Contains(string(format), "\"properties\"") {
		t.Fatalf("expected properties in schema payload: %s", string(format))
	}
}

func TestEmbeddingRequestFromManyRejectsEmptyInput(t *testing.T) {
	t.Parallel()

	_, _, err := embeddingRequestFromMany("embedding-model", &core.EmbedManyParams{Inputs: []string{"ok", "  "}})
	if err == nil {
		t.Fatal("expected error for empty input")
	}
	if !strings.Contains(err.Error(), "index 1") {
		t.Fatalf("unexpected error: %v", err)
	}
}
