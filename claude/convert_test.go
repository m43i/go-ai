package claude

import (
	"strings"
	"testing"

	"github.com/m43i/go-ai/core"
)

// ---------------------------------------------------------------------------
// toMessage — ContentMessagePart
// ---------------------------------------------------------------------------

func TestToMessageContentMessagePart(t *testing.T) {
	t.Parallel()

	msg := core.ContentMessagePart{
		Role: "user",
		Parts: []core.ContentPart{
			core.TextPart{Text: "Describe this image"},
			core.ImagePart{
				Source: core.URLSource{URL: "https://example.com/photo.jpg"},
			},
		},
	}

	result, systemText, err := toMessage(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if systemText != "" {
		t.Fatalf("unexpected system text: %q", systemText)
	}
	if result == nil {
		t.Fatal("expected non-nil message")
	}
	if result.Role != "user" {
		t.Fatalf("unexpected role: %q", result.Role)
	}
	if len(result.Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(result.Content))
	}
	if result.Content[0].Type != "text" || result.Content[0].Text != "Describe this image" {
		t.Fatalf("unexpected text block: %#v", result.Content[0])
	}
	if result.Content[1].Type != "image" {
		t.Fatalf("unexpected image block type: %q", result.Content[1].Type)
	}
	if result.Content[1].Source == nil || result.Content[1].Source.Type != "url" {
		t.Fatalf("unexpected image source: %#v", result.Content[1].Source)
	}
	if result.Content[1].Source.URL != "https://example.com/photo.jpg" {
		t.Fatalf("unexpected image URL: %q", result.Content[1].Source.URL)
	}
}

func TestToMessageContentMessagePartPointer(t *testing.T) {
	t.Parallel()

	msg := &core.ContentMessagePart{
		Role: "user",
		Parts: []core.ContentPart{
			core.TextPart{Text: "Hello"},
		},
	}

	result, _, err := toMessage(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Role != "user" {
		t.Fatalf("unexpected role: %q", result.Role)
	}
}

func TestToMessageContentMessagePartNilPointer(t *testing.T) {
	t.Parallel()

	var msg *core.ContentMessagePart
	_, _, err := toMessage(msg)
	if err == nil {
		t.Fatal("expected error for nil content message pointer")
	}
	if !strings.Contains(err.Error(), "nil") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestToMessageContentMessagePartSystemRole(t *testing.T) {
	t.Parallel()

	msg := core.ContentMessagePart{
		Role:  "system",
		Parts: []core.ContentPart{core.TextPart{Text: "hi"}},
	}

	_, _, err := toMessage(msg)
	if err == nil {
		t.Fatal("expected error for system role in content message")
	}
	if !strings.Contains(err.Error(), "content messages cannot use system role") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestToMessageContentMessagePartEmptyRole(t *testing.T) {
	t.Parallel()

	msg := core.ContentMessagePart{
		Role:  "",
		Parts: []core.ContentPart{core.TextPart{Text: "hi"}},
	}

	_, _, err := toMessage(msg)
	if err == nil {
		t.Fatal("expected error for empty role")
	}
	if !strings.Contains(err.Error(), "role is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestToMessageContentMessagePartNoParts(t *testing.T) {
	t.Parallel()

	msg := core.ContentMessagePart{
		Role:  "user",
		Parts: nil,
	}

	_, _, err := toMessage(msg)
	if err == nil {
		t.Fatal("expected error for empty parts")
	}
	if !strings.Contains(err.Error(), "at least one content part") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// ---------------------------------------------------------------------------
// toContentBlock — TextPart
// ---------------------------------------------------------------------------

func TestToContentBlockText(t *testing.T) {
	t.Parallel()

	result, err := toContentBlock(core.TextPart{Text: "hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Type != "text" || result.Text != "hello" {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestToContentBlockTextPointer(t *testing.T) {
	t.Parallel()

	result, err := toContentBlock(&core.TextPart{Text: "world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Type != "text" || result.Text != "world" {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestToContentBlockTextNilPointer(t *testing.T) {
	t.Parallel()

	var tp *core.TextPart
	_, err := toContentBlock(tp)
	if err == nil {
		t.Fatal("expected error for nil text part pointer")
	}
}

// ---------------------------------------------------------------------------
// toContentBlock — ImagePart
// ---------------------------------------------------------------------------

func TestImageBlockURL(t *testing.T) {
	t.Parallel()

	part := core.ImagePart{Source: core.URLSource{URL: "https://example.com/img.png"}}
	result, err := toContentBlock(part)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Type != "image" {
		t.Fatalf("unexpected type: %q", result.Type)
	}
	if result.Source == nil {
		t.Fatal("expected source to be set")
	}
	if result.Source.Type != "url" {
		t.Fatalf("unexpected source type: %q", result.Source.Type)
	}
	if result.Source.URL != "https://example.com/img.png" {
		t.Fatalf("unexpected URL: %q", result.Source.URL)
	}
}

func TestImageBlockBase64(t *testing.T) {
	t.Parallel()

	part := core.ImagePart{
		Source: core.DataSource{Data: "aGVsbG8=", MimeType: "image/png"},
	}
	result, err := toContentBlock(part)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Type != "image" {
		t.Fatalf("unexpected type: %q", result.Type)
	}
	if result.Source == nil {
		t.Fatal("expected source to be set")
	}
	if result.Source.Type != "base64" {
		t.Fatalf("unexpected source type: %q", result.Source.Type)
	}
	if result.Source.Data != "aGVsbG8=" {
		t.Fatalf("unexpected data: %q", result.Source.Data)
	}
	if result.Source.MediaType != "image/png" {
		t.Fatalf("unexpected media type: %q", result.Source.MediaType)
	}
}

func TestImageBlockNilSource(t *testing.T) {
	t.Parallel()

	part := core.ImagePart{Source: nil}
	_, err := toContentBlock(part)
	if err == nil {
		t.Fatal("expected error for nil image source")
	}
	if !strings.Contains(err.Error(), "image source is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImageBlockPointerNil(t *testing.T) {
	t.Parallel()

	var part *core.ImagePart
	_, err := toContentBlock(part)
	if err == nil {
		t.Fatal("expected error for nil image part pointer")
	}
}

func TestImageBlockEmptyURL(t *testing.T) {
	t.Parallel()

	part := core.ImagePart{Source: core.URLSource{URL: "  "}}
	_, err := toContentBlock(part)
	if err == nil {
		t.Fatal("expected error for empty URL")
	}
	if !strings.Contains(err.Error(), "source URL is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImageBlockEmptyData(t *testing.T) {
	t.Parallel()

	part := core.ImagePart{Source: core.DataSource{Data: "", MimeType: "image/png"}}
	_, err := toContentBlock(part)
	if err == nil {
		t.Fatal("expected error for empty data")
	}
	if !strings.Contains(err.Error(), "source data is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImageBlockEmptyMimeType(t *testing.T) {
	t.Parallel()

	part := core.ImagePart{Source: core.DataSource{Data: "aGVsbG8=", MimeType: ""}}
	_, err := toContentBlock(part)
	if err == nil {
		t.Fatal("expected error for empty mime type")
	}
	if !strings.Contains(err.Error(), "source mime type is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImageBlockNilURLSourcePointer(t *testing.T) {
	t.Parallel()

	var src *core.URLSource
	part := core.ImagePart{Source: src}
	_, err := toContentBlock(part)
	if err == nil {
		t.Fatal("expected error for nil URL source pointer")
	}
	if !strings.Contains(err.Error(), "nil") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImageBlockNilDataSourcePointer(t *testing.T) {
	t.Parallel()

	var src *core.DataSource
	part := core.ImagePart{Source: src}
	_, err := toContentBlock(part)
	if err == nil {
		t.Fatal("expected error for nil data source pointer")
	}
	if !strings.Contains(err.Error(), "nil") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImageBlockURLSourcePointer(t *testing.T) {
	t.Parallel()

	src := &core.URLSource{URL: "https://example.com/img.jpg"}
	part := core.ImagePart{Source: src}
	result, err := toContentBlock(part)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Source.URL != "https://example.com/img.jpg" {
		t.Fatalf("unexpected URL: %q", result.Source.URL)
	}
}

func TestImageBlockDataSourcePointer(t *testing.T) {
	t.Parallel()

	src := &core.DataSource{Data: "aGVsbG8=", MimeType: "image/jpeg"}
	part := core.ImagePart{Source: src}
	result, err := toContentBlock(part)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Source.Data != "aGVsbG8=" || result.Source.MediaType != "image/jpeg" {
		t.Fatalf("unexpected source: %#v", result.Source)
	}
}

// ---------------------------------------------------------------------------
// toContentBlock — AudioPart
// ---------------------------------------------------------------------------

func TestAudioBlockURL(t *testing.T) {
	t.Parallel()

	part := core.AudioPart{Source: core.URLSource{URL: "https://example.com/audio.mp3"}}
	result, err := toContentBlock(part)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Type != "audio" {
		t.Fatalf("unexpected type: %q", result.Type)
	}
	if result.Source == nil || result.Source.Type != "url" {
		t.Fatalf("unexpected source: %#v", result.Source)
	}
	if result.Source.URL != "https://example.com/audio.mp3" {
		t.Fatalf("unexpected URL: %q", result.Source.URL)
	}
}

func TestAudioBlockBase64(t *testing.T) {
	t.Parallel()

	part := core.AudioPart{
		Source: core.DataSource{Data: "YXVkaW8=", MimeType: "audio/wav"},
	}
	result, err := toContentBlock(part)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Type != "audio" {
		t.Fatalf("unexpected type: %q", result.Type)
	}
	if result.Source.Type != "base64" {
		t.Fatalf("unexpected source type: %q", result.Source.Type)
	}
	if result.Source.Data != "YXVkaW8=" {
		t.Fatalf("unexpected data: %q", result.Source.Data)
	}
	if result.Source.MediaType != "audio/wav" {
		t.Fatalf("unexpected media type: %q", result.Source.MediaType)
	}
}

func TestAudioBlockNilSource(t *testing.T) {
	t.Parallel()

	part := core.AudioPart{Source: nil}
	_, err := toContentBlock(part)
	if err == nil {
		t.Fatal("expected error for nil audio source")
	}
	if !strings.Contains(err.Error(), "audio source is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestAudioBlockPointerNil(t *testing.T) {
	t.Parallel()

	var part *core.AudioPart
	_, err := toContentBlock(part)
	if err == nil {
		t.Fatal("expected error for nil audio part pointer")
	}
}

// ---------------------------------------------------------------------------
// toContentBlock — DocumentPart
// ---------------------------------------------------------------------------

func TestDocumentBlockURL(t *testing.T) {
	t.Parallel()

	part := core.DocumentPart{Source: core.URLSource{URL: "https://example.com/doc.pdf"}}
	result, err := toContentBlock(part)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Type != "document" {
		t.Fatalf("unexpected type: %q", result.Type)
	}
	if result.Source == nil || result.Source.Type != "url" {
		t.Fatalf("unexpected source: %#v", result.Source)
	}
}

func TestDocumentBlockBase64(t *testing.T) {
	t.Parallel()

	part := core.DocumentPart{
		Source: core.DataSource{Data: "cGRm", MimeType: "application/pdf"},
	}
	result, err := toContentBlock(part)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Type != "document" {
		t.Fatalf("unexpected type: %q", result.Type)
	}
	if result.Source.Type != "base64" {
		t.Fatalf("unexpected source type: %q", result.Source.Type)
	}
	if result.Source.MediaType != "application/pdf" {
		t.Fatalf("unexpected media type: %q", result.Source.MediaType)
	}
}

func TestDocumentBlockNilSource(t *testing.T) {
	t.Parallel()

	part := core.DocumentPart{Source: nil}
	_, err := toContentBlock(part)
	if err == nil {
		t.Fatal("expected error for nil document source")
	}
	if !strings.Contains(err.Error(), "document source is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDocumentBlockPointerNil(t *testing.T) {
	t.Parallel()

	var part *core.DocumentPart
	_, err := toContentBlock(part)
	if err == nil {
		t.Fatal("expected error for nil document part pointer")
	}
}

// ---------------------------------------------------------------------------
// mediaSourceFromSource
// ---------------------------------------------------------------------------

func TestMediaSourceFromURLSource(t *testing.T) {
	t.Parallel()

	src := core.URLSource{URL: "https://example.com/file.png"}
	ms, err := mediaSourceFromSource(src)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ms.Type != "url" {
		t.Fatalf("unexpected type: %q", ms.Type)
	}
	if ms.URL != "https://example.com/file.png" {
		t.Fatalf("unexpected URL: %q", ms.URL)
	}
}

func TestMediaSourceFromDataSource(t *testing.T) {
	t.Parallel()

	src := core.DataSource{Data: "aGVsbG8=", MimeType: "image/png"}
	ms, err := mediaSourceFromSource(src)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ms.Type != "base64" {
		t.Fatalf("unexpected type: %q", ms.Type)
	}
	if ms.Data != "aGVsbG8=" {
		t.Fatalf("unexpected data: %q", ms.Data)
	}
	if ms.MediaType != "image/png" {
		t.Fatalf("unexpected media type: %q", ms.MediaType)
	}
}

func TestMediaSourceFromURLSourcePointer(t *testing.T) {
	t.Parallel()

	src := &core.URLSource{URL: "https://example.com/test.jpg"}
	ms, err := mediaSourceFromSource(src)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ms.URL != "https://example.com/test.jpg" {
		t.Fatalf("unexpected URL: %q", ms.URL)
	}
}

func TestMediaSourceFromDataSourcePointer(t *testing.T) {
	t.Parallel()

	src := &core.DataSource{Data: "aGVsbG8=", MimeType: "image/jpeg"}
	ms, err := mediaSourceFromSource(src)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ms.Data != "aGVsbG8=" || ms.MediaType != "image/jpeg" {
		t.Fatalf("unexpected source: %#v", ms)
	}
}

func TestMediaSourceFromNilURLSourcePointer(t *testing.T) {
	t.Parallel()

	var src *core.URLSource
	_, err := mediaSourceFromSource(src)
	if err == nil {
		t.Fatal("expected error for nil URL source pointer")
	}
}

func TestMediaSourceFromNilDataSourcePointer(t *testing.T) {
	t.Parallel()

	var src *core.DataSource
	_, err := mediaSourceFromSource(src)
	if err == nil {
		t.Fatal("expected error for nil data source pointer")
	}
}

func TestMediaSourceEmptyURL(t *testing.T) {
	t.Parallel()

	src := core.URLSource{URL: "  "}
	_, err := mediaSourceFromSource(src)
	if err == nil {
		t.Fatal("expected error for empty URL")
	}
	if !strings.Contains(err.Error(), "source URL is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestMediaSourceEmptyData(t *testing.T) {
	t.Parallel()

	src := core.DataSource{Data: "", MimeType: "image/png"}
	_, err := mediaSourceFromSource(src)
	if err == nil {
		t.Fatal("expected error for empty data")
	}
	if !strings.Contains(err.Error(), "source data is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestMediaSourceEmptyMimeType(t *testing.T) {
	t.Parallel()

	src := core.DataSource{Data: "aGVsbG8=", MimeType: ""}
	_, err := mediaSourceFromSource(src)
	if err == nil {
		t.Fatal("expected error for empty mime type")
	}
	if !strings.Contains(err.Error(), "source mime type is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// ---------------------------------------------------------------------------
// toContentBlocks — mixed multimodal
// ---------------------------------------------------------------------------

func TestToContentBlocksMixed(t *testing.T) {
	t.Parallel()

	parts := []core.ContentPart{
		core.TextPart{Text: "Look at these:"},
		core.ImagePart{Source: core.URLSource{URL: "https://example.com/photo.jpg"}},
		core.AudioPart{Source: core.DataSource{Data: "YXVkaW8=", MimeType: "audio/wav"}},
		core.DocumentPart{Source: core.URLSource{URL: "https://example.com/doc.pdf"}},
	}

	blocks, err := toContentBlocks(parts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(blocks) != 4 {
		t.Fatalf("expected 4 blocks, got %d", len(blocks))
	}
	if blocks[0].Type != "text" {
		t.Fatalf("unexpected first block type: %q", blocks[0].Type)
	}
	if blocks[1].Type != "image" {
		t.Fatalf("unexpected second block type: %q", blocks[1].Type)
	}
	if blocks[2].Type != "audio" {
		t.Fatalf("unexpected third block type: %q", blocks[2].Type)
	}
	if blocks[3].Type != "document" {
		t.Fatalf("unexpected fourth block type: %q", blocks[3].Type)
	}
}

// ---------------------------------------------------------------------------
// toMessagesAndSystem — full message conversion
// ---------------------------------------------------------------------------

func TestToMessagesAndSystemMultimodal(t *testing.T) {
	t.Parallel()

	params := &core.ChatParams{
		Messages: []core.MessageUnion{
			core.TextMessagePart{Role: "system", Content: "You are helpful."},
			core.ContentMessagePart{
				Role: "user",
				Parts: []core.ContentPart{
					core.TextPart{Text: "What is this?"},
					core.ImagePart{Source: core.URLSource{URL: "https://example.com/img.png"}},
				},
			},
		},
	}

	messages, system, err := toMessagesAndSystem(params)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if system != "You are helpful." {
		t.Fatalf("unexpected system: %q", system)
	}
	if len(messages) != 1 {
		t.Fatalf("expected 1 message (system extracted), got %d", len(messages))
	}
	if messages[0].Role != "user" {
		t.Fatalf("unexpected role: %q", messages[0].Role)
	}
	if len(messages[0].Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(messages[0].Content))
	}
}

func TestToMessagesAndSystemNilParams(t *testing.T) {
	t.Parallel()

	_, _, err := toMessagesAndSystem(nil)
	if err == nil {
		t.Fatal("expected error for nil params")
	}
}
