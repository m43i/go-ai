package openai

import (
	"strings"
	"testing"

	"github.com/m43i/go-ai/core"
)

// ---------------------------------------------------------------------------
// toChatMessage — ContentMessagePart
// ---------------------------------------------------------------------------

func TestToChatMessageContentMessagePart(t *testing.T) {
	t.Parallel()

	msg := core.ContentMessagePart{
		Role: "user",
		Parts: []core.ContentPart{
			core.TextPart{Text: "Describe this image"},
			core.ImagePart{
				Source:   core.URLSource{URL: "https://example.com/photo.jpg"},
				Metadata: map[string]any{"detail": "high"},
			},
		},
	}

	result, err := toChatMessage(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Role != "user" {
		t.Fatalf("unexpected role: %q", result.Role)
	}

	parts, ok := result.Content.([]chatContentPart)
	if !ok {
		t.Fatalf("expected []chatContentPart, got %T", result.Content)
	}
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(parts))
	}
	if parts[0].Type != "text" || parts[0].Text != "Describe this image" {
		t.Fatalf("unexpected text part: %#v", parts[0])
	}
	if parts[1].Type != "image_url" {
		t.Fatalf("unexpected image part type: %q", parts[1].Type)
	}
	if parts[1].ImageURL == nil || parts[1].ImageURL.URL != "https://example.com/photo.jpg" {
		t.Fatalf("unexpected image URL: %#v", parts[1].ImageURL)
	}
	if parts[1].ImageURL.Detail != "high" {
		t.Fatalf("expected detail=high, got %q", parts[1].ImageURL.Detail)
	}
}

func TestToChatMessageContentMessagePartPointer(t *testing.T) {
	t.Parallel()

	msg := &core.ContentMessagePart{
		Role: "user",
		Parts: []core.ContentPart{
			core.TextPart{Text: "Hello"},
		},
	}

	result, err := toChatMessage(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Role != "user" {
		t.Fatalf("unexpected role: %q", result.Role)
	}
}

func TestToChatMessageContentMessagePartNilPointer(t *testing.T) {
	t.Parallel()

	var msg *core.ContentMessagePart
	_, err := toChatMessage(msg)
	if err == nil {
		t.Fatal("expected error for nil content message pointer")
	}
	if !strings.Contains(err.Error(), "nil") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestToChatMessageContentMessagePartEmptyRole(t *testing.T) {
	t.Parallel()

	msg := core.ContentMessagePart{
		Role:  "",
		Parts: []core.ContentPart{core.TextPart{Text: "hi"}},
	}

	_, err := toChatMessage(msg)
	if err == nil {
		t.Fatal("expected error for empty role")
	}
	if !strings.Contains(err.Error(), "role is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestToChatMessageContentMessagePartNoParts(t *testing.T) {
	t.Parallel()

	msg := core.ContentMessagePart{
		Role:  "user",
		Parts: nil,
	}

	_, err := toChatMessage(msg)
	if err == nil {
		t.Fatal("expected error for empty parts")
	}
	if !strings.Contains(err.Error(), "at least one content part") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// ---------------------------------------------------------------------------
// toChatContentPart — TextPart
// ---------------------------------------------------------------------------

func TestToChatContentPartText(t *testing.T) {
	t.Parallel()

	result, err := toChatContentPart(core.TextPart{Text: "hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Type != "text" || result.Text != "hello" {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestToChatContentPartTextPointer(t *testing.T) {
	t.Parallel()

	result, err := toChatContentPart(&core.TextPart{Text: "world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Type != "text" || result.Text != "world" {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestToChatContentPartTextNilPointer(t *testing.T) {
	t.Parallel()

	var tp *core.TextPart
	_, err := toChatContentPart(tp)
	if err == nil {
		t.Fatal("expected error for nil text part pointer")
	}
}

// ---------------------------------------------------------------------------
// toChatContentPart — ImagePart with URL source
// ---------------------------------------------------------------------------

func TestImageContentPartURL(t *testing.T) {
	t.Parallel()

	part := core.ImagePart{Source: core.URLSource{URL: "https://example.com/img.png"}}
	result, err := toChatContentPart(part)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Type != "image_url" {
		t.Fatalf("unexpected type: %q", result.Type)
	}
	if result.ImageURL == nil || result.ImageURL.URL != "https://example.com/img.png" {
		t.Fatalf("unexpected image url: %#v", result.ImageURL)
	}
	if result.ImageURL.Detail != "" {
		t.Fatalf("expected no detail, got %q", result.ImageURL.Detail)
	}
}

func TestImageContentPartURLWithDetail(t *testing.T) {
	t.Parallel()

	part := core.ImagePart{
		Source:   core.URLSource{URL: "https://example.com/img.png"},
		Metadata: map[string]any{"detail": "low"},
	}
	result, err := toChatContentPart(part)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ImageURL.Detail != "low" {
		t.Fatalf("expected detail=low, got %q", result.ImageURL.Detail)
	}
}

func TestImageContentPartBase64(t *testing.T) {
	t.Parallel()

	part := core.ImagePart{
		Source: core.DataSource{Data: "aGVsbG8=", MimeType: "image/png"},
	}
	result, err := toChatContentPart(part)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Type != "image_url" {
		t.Fatalf("unexpected type: %q", result.Type)
	}
	if result.ImageURL == nil || result.ImageURL.URL != "data:image/png;base64,aGVsbG8=" {
		t.Fatalf("unexpected data URL: %q", result.ImageURL.URL)
	}
}

func TestImageContentPartBase64RejectsDataPrefix(t *testing.T) {
	t.Parallel()

	part := core.ImagePart{
		Source: core.DataSource{Data: "data:image/png;base64,aGVsbG8=", MimeType: "image/png"},
	}
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for data: prefixed data")
	}
	if !strings.Contains(err.Error(), "raw base64") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImageContentPartNilSource(t *testing.T) {
	t.Parallel()

	part := core.ImagePart{Source: nil}
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for nil image source")
	}
	if !strings.Contains(err.Error(), "image source is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImageContentPartPointerNil(t *testing.T) {
	t.Parallel()

	var part *core.ImagePart
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for nil image part pointer")
	}
}

func TestImageContentPartEmptyURL(t *testing.T) {
	t.Parallel()

	part := core.ImagePart{Source: core.URLSource{URL: "  "}}
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for empty URL")
	}
	if !strings.Contains(err.Error(), "image URL is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImageContentPartEmptyData(t *testing.T) {
	t.Parallel()

	part := core.ImagePart{Source: core.DataSource{Data: "", MimeType: "image/png"}}
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for empty data")
	}
	if !strings.Contains(err.Error(), "image data is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImageContentPartEmptyMimeType(t *testing.T) {
	t.Parallel()

	part := core.ImagePart{Source: core.DataSource{Data: "aGVsbG8=", MimeType: ""}}
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for empty mime type")
	}
	if !strings.Contains(err.Error(), "image mime type is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImageContentPartNilURLSourcePointer(t *testing.T) {
	t.Parallel()

	var src *core.URLSource
	part := core.ImagePart{Source: src}
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for nil URL source pointer")
	}
	if !strings.Contains(err.Error(), "nil") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImageContentPartNilDataSourcePointer(t *testing.T) {
	t.Parallel()

	var src *core.DataSource
	part := core.ImagePart{Source: src}
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for nil data source pointer")
	}
	if !strings.Contains(err.Error(), "nil") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImageContentPartURLSourcePointer(t *testing.T) {
	t.Parallel()

	src := &core.URLSource{URL: "https://example.com/img.jpg"}
	part := core.ImagePart{Source: src}
	result, err := toChatContentPart(part)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ImageURL.URL != "https://example.com/img.jpg" {
		t.Fatalf("unexpected URL: %q", result.ImageURL.URL)
	}
}

func TestImageContentPartDataSourcePointer(t *testing.T) {
	t.Parallel()

	src := &core.DataSource{Data: "aGVsbG8=", MimeType: "image/jpeg"}
	part := core.ImagePart{Source: src}
	result, err := toChatContentPart(part)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ImageURL.URL != "data:image/jpeg;base64,aGVsbG8=" {
		t.Fatalf("unexpected URL: %q", result.ImageURL.URL)
	}
}

// ---------------------------------------------------------------------------
// toChatContentPart — AudioPart
// ---------------------------------------------------------------------------

func TestAudioContentPartDataSource(t *testing.T) {
	t.Parallel()

	part := core.AudioPart{
		Source: core.DataSource{Data: "YXVkaW8=", MimeType: "audio/mp3"},
	}
	result, err := toChatContentPart(part)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Type != "input_audio" {
		t.Fatalf("unexpected type: %q", result.Type)
	}
	if result.InputAudio == nil {
		t.Fatal("expected InputAudio to be set")
	}
	if result.InputAudio.Data != "YXVkaW8=" {
		t.Fatalf("unexpected audio data: %q", result.InputAudio.Data)
	}
	if result.InputAudio.Format != "mp3" {
		t.Fatalf("unexpected audio format: %q", result.InputAudio.Format)
	}
}

func TestAudioContentPartVariousMimeTypes(t *testing.T) {
	t.Parallel()

	tests := []struct {
		mimeType string
		format   string
	}{
		{"audio/mp3", "mp3"},
		{"audio/mpeg", "mp3"},
		{"audio/wav", "wav"},
		{"audio/wave", "wav"},
		{"audio/x-wav", "wav"},
		{"audio/flac", "flac"},
		{"audio/ogg", "ogg"},
		{"audio/webm", "webm"},
	}

	for _, tt := range tests {
		t.Run(tt.mimeType, func(t *testing.T) {
			part := core.AudioPart{
				Source: core.DataSource{Data: "YXVkaW8=", MimeType: tt.mimeType},
			}
			result, err := toChatContentPart(part)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result.InputAudio.Format != tt.format {
				t.Fatalf("expected format %q, got %q", tt.format, result.InputAudio.Format)
			}
		})
	}
}

func TestAudioContentPartUnsupportedMimeType(t *testing.T) {
	t.Parallel()

	part := core.AudioPart{
		Source: core.DataSource{Data: "YXVkaW8=", MimeType: "audio/aac"},
	}
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for unsupported mime type")
	}
	if !strings.Contains(err.Error(), "unsupported audio mime type") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestAudioContentPartURLSource(t *testing.T) {
	t.Parallel()

	part := core.AudioPart{
		Source: core.URLSource{URL: "https://example.com/audio.mp3"},
	}
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for URL source audio (only DataSource supported)")
	}
	if !strings.Contains(err.Error(), "only DataSource is supported") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestAudioContentPartNilSource(t *testing.T) {
	t.Parallel()

	part := core.AudioPart{Source: nil}
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for nil audio source")
	}
}

func TestAudioContentPartPointerNil(t *testing.T) {
	t.Parallel()

	var part *core.AudioPart
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for nil audio part pointer")
	}
}

func TestAudioContentPartEmptyData(t *testing.T) {
	t.Parallel()

	part := core.AudioPart{
		Source: core.DataSource{Data: "", MimeType: "audio/mp3"},
	}
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for empty audio data")
	}
	if !strings.Contains(err.Error(), "audio data is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestAudioContentPartEmptyMimeType(t *testing.T) {
	t.Parallel()

	part := core.AudioPart{
		Source: core.DataSource{Data: "YXVkaW8=", MimeType: ""},
	}
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for empty audio mime type")
	}
	if !strings.Contains(err.Error(), "audio mime type is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// ---------------------------------------------------------------------------
// toChatContentPart — DocumentPart
// ---------------------------------------------------------------------------

func TestDocumentContentPartNotSupported(t *testing.T) {
	t.Parallel()

	part := core.DocumentPart{
		Source: core.URLSource{URL: "https://example.com/doc.pdf"},
	}
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for document content")
	}
	if !strings.Contains(err.Error(), "not supported") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDocumentContentPartNilSource(t *testing.T) {
	t.Parallel()

	part := core.DocumentPart{Source: nil}
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for nil document source")
	}
}

func TestDocumentContentPartPointerNil(t *testing.T) {
	t.Parallel()

	var part *core.DocumentPart
	_, err := toChatContentPart(part)
	if err == nil {
		t.Fatal("expected error for nil document part pointer")
	}
}

// ---------------------------------------------------------------------------
// imageDetail
// ---------------------------------------------------------------------------

func TestImageDetailExtraction(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		metadata map[string]any
		expected string
	}{
		{"nil metadata", nil, ""},
		{"empty metadata", map[string]any{}, ""},
		{"no detail key", map[string]any{"other": "value"}, ""},
		{"detail high", map[string]any{"detail": "high"}, "high"},
		{"detail low", map[string]any{"detail": "low"}, "low"},
		{"detail auto", map[string]any{"detail": "auto"}, "auto"},
		{"detail whitespace", map[string]any{"detail": "  high  "}, "high"},
		{"detail non-string", map[string]any{"detail": 123}, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := imageDetail(tt.metadata)
			if result != tt.expected {
				t.Fatalf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// audioFormatFromMime
// ---------------------------------------------------------------------------

func TestAudioFormatFromMime(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input    string
		expected string
	}{
		{"audio/mp3", "mp3"},
		{"audio/mpeg", "mp3"},
		{"Audio/MP3", "mp3"},
		{"audio/wav", "wav"},
		{"audio/wave", "wav"},
		{"audio/x-wav", "wav"},
		{"audio/flac", "flac"},
		{"audio/ogg", "ogg"},
		{"audio/webm", "webm"},
		{"  audio/mp3  ", "mp3"},
		{"audio/aac", ""},
		{"video/mp4", ""},
		{"", ""},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := audioFormatFromMime(tt.input)
			if result != tt.expected {
				t.Fatalf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// toChatContentParts — mixed multimodal message
// ---------------------------------------------------------------------------

func TestToChatContentPartsMixed(t *testing.T) {
	t.Parallel()

	parts := []core.ContentPart{
		core.TextPart{Text: "Look at this:"},
		core.ImagePart{Source: core.URLSource{URL: "https://example.com/photo.jpg"}},
		core.AudioPart{Source: core.DataSource{Data: "YXVkaW8=", MimeType: "audio/wav"}},
	}

	result, err := toChatContentParts(parts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 3 {
		t.Fatalf("expected 3 parts, got %d", len(result))
	}
	if result[0].Type != "text" {
		t.Fatalf("unexpected first part type: %q", result[0].Type)
	}
	if result[1].Type != "image_url" {
		t.Fatalf("unexpected second part type: %q", result[1].Type)
	}
	if result[2].Type != "input_audio" {
		t.Fatalf("unexpected third part type: %q", result[2].Type)
	}
}

// ---------------------------------------------------------------------------
// toChatMessages — full message conversion
// ---------------------------------------------------------------------------

func TestToChatMessagesMultimodal(t *testing.T) {
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

	messages, err := toChatMessages(params)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}

	// First message: plain text
	if messages[0].Role != "system" {
		t.Fatalf("unexpected first message role: %q", messages[0].Role)
	}
	if messages[0].Content != "You are helpful." {
		t.Fatalf("unexpected first message content: %#v", messages[0].Content)
	}

	// Second message: multimodal
	if messages[1].Role != "user" {
		t.Fatalf("unexpected second message role: %q", messages[1].Role)
	}
	contentParts, ok := messages[1].Content.([]chatContentPart)
	if !ok {
		t.Fatalf("expected []chatContentPart for second message, got %T", messages[1].Content)
	}
	if len(contentParts) != 2 {
		t.Fatalf("expected 2 content parts, got %d", len(contentParts))
	}
}

func TestToChatMessagesNilParams(t *testing.T) {
	t.Parallel()

	_, err := toChatMessages(nil)
	if err == nil {
		t.Fatal("expected error for nil params")
	}
}
