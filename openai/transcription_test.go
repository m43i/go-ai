package openai

import (
	"context"
	"encoding/json"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/m43i/go-ai/core"
)

func TestTranscribe(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/audio/transcriptions" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("unexpected method %q", r.Method)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("unexpected authorization header %q", got)
		}

		contentType := r.Header.Get("Content-Type")
		mediaType, params, err := mime.ParseMediaType(contentType)
		if err != nil {
			t.Fatalf("parse content type: %v", err)
		}
		if mediaType != "multipart/form-data" {
			t.Fatalf("unexpected media type %q", mediaType)
		}

		reader := multipart.NewReader(r.Body, params["boundary"])
		fields := make(map[string]string)
		var fileData []byte
		var fileName string

		for {
			part, err := reader.NextPart()
			if err != nil {
				break
			}
			data, _ := io.ReadAll(part)
			if part.FormName() == "file" {
				fileData = data
				fileName = part.FileName()
			} else {
				fields[part.FormName()] = string(data)
			}
		}

		if fields["model"] != "whisper-1" {
			t.Fatalf("unexpected model: %q", fields["model"])
		}
		if fields["language"] != "en" {
			t.Fatalf("unexpected language: %q", fields["language"])
		}
		if fileName != "test.mp3" {
			t.Fatalf("unexpected filename: %q", fileName)
		}
		if string(fileData) != "fake-audio-bytes" {
			t.Fatalf("unexpected file data: %q", string(fileData))
		}
		if fields["temperature"] != "0" {
			t.Fatalf("unexpected temperature: %q", fields["temperature"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"text": "Hello world, this is a test.",
			"language": "en",
			"duration": 3.5,
			"segments": [
				{
					"start": 0.0,
					"end": 1.5,
					"text": "Hello world,",
					"words": [
						{"word": "Hello", "start": 0.0, "end": 0.5},
						{"word": "world,", "start": 0.6, "end": 1.0}
					]
				},
				{
					"start": 1.5,
					"end": 3.5,
					"text": " this is a test."
				}
			]
		}`))
	}))
	defer server.Close()

	adapter := New("test-key", "whisper-1", WithBaseURL(server.URL), WithHTTPClient(server.Client()))

	result, err := adapter.Transcribe(context.Background(), &core.TranscriptionParams{
		Audio:    []byte("fake-audio-bytes"),
		Filename: "test.mp3",
		Language: "en",
		ModelOptions: map[string]any{
			"temperature": 0,
		},
	})
	if err != nil {
		t.Fatalf("transcribe returned error: %v", err)
	}

	if result.Text != "Hello world, this is a test." {
		t.Fatalf("unexpected text: %q", result.Text)
	}
	if result.Language != "en" {
		t.Fatalf("unexpected language: %q", result.Language)
	}
	if result.Duration != 3.5 {
		t.Fatalf("unexpected duration: %v", result.Duration)
	}
	if len(result.Segments) != 2 {
		t.Fatalf("expected 2 segments, got %d", len(result.Segments))
	}

	seg0 := result.Segments[0]
	if seg0.Start != 0.0 || seg0.End != 1.5 {
		t.Fatalf("unexpected segment 0 timing: start=%v end=%v", seg0.Start, seg0.End)
	}
	if seg0.Text != "Hello world," {
		t.Fatalf("unexpected segment 0 text: %q", seg0.Text)
	}
	if len(seg0.Words) != 2 {
		t.Fatalf("expected 2 words in segment 0, got %d", len(seg0.Words))
	}
	if seg0.Words[0].Word != "Hello" || seg0.Words[0].Start != 0.0 || seg0.Words[0].End != 0.5 {
		t.Fatalf("unexpected word 0: %#v", seg0.Words[0])
	}

	seg1 := result.Segments[1]
	if len(seg1.Words) != 0 {
		t.Fatalf("expected 0 words in segment 1, got %d", len(seg1.Words))
	}
}

func TestTranscribeTopLevelWords(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"text": "Hello world.",
			"language": "en",
			"duration": 2.0,
			"words": [
				{"word": "Hello", "start": 0.0, "end": 0.8},
				{"word": "world.", "start": 0.9, "end": 1.8}
			]
		}`))
	}))
	defer server.Close()

	adapter := New("test-key", "whisper-1", WithBaseURL(server.URL), WithHTTPClient(server.Client()))

	result, err := adapter.Transcribe(context.Background(), &core.TranscriptionParams{
		Audio:    []byte("fake"),
		Filename: "test.wav",
	})
	if err != nil {
		t.Fatalf("transcribe returned error: %v", err)
	}

	// Top-level words should be wrapped into a single synthetic segment.
	if len(result.Segments) != 1 {
		t.Fatalf("expected 1 synthetic segment, got %d", len(result.Segments))
	}
	if result.Segments[0].Start != 0 || result.Segments[0].End != 2.0 {
		t.Fatalf("unexpected synthetic segment timing: start=%v end=%v", result.Segments[0].Start, result.Segments[0].End)
	}
	if len(result.Segments[0].Words) != 2 {
		t.Fatalf("expected 2 words, got %d", len(result.Segments[0].Words))
	}
}

func TestTranscribeAPIError(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"error":{"message":"Invalid file format","type":"invalid_request_error"}}`))
	}))
	defer server.Close()

	adapter := New("test-key", "whisper-1", WithBaseURL(server.URL), WithHTTPClient(server.Client()))

	_, err := adapter.Transcribe(context.Background(), &core.TranscriptionParams{
		Audio:    []byte("bad-data"),
		Filename: "test.mp3",
	})
	if err == nil {
		t.Fatal("expected error for bad request")
	}
}

func TestTranscribeValidation(t *testing.T) {
	t.Parallel()

	adapter := New("test-key", "whisper-1")

	tests := []struct {
		name   string
		params *core.TranscriptionParams
		errMsg string
	}{
		{
			name:   "nil params",
			params: nil,
			errMsg: "params are required",
		},
		{
			name:   "empty audio",
			params: &core.TranscriptionParams{Filename: "test.mp3"},
			errMsg: "audio data is required",
		},
		{
			name:   "empty filename",
			params: &core.TranscriptionParams{Audio: []byte("data"), Filename: "   "},
			errMsg: "filename is required",
		},
		{
			name: "reserved model option",
			params: &core.TranscriptionParams{
				Audio:        []byte("data"),
				Filename:     "test.mp3",
				ModelOptions: map[string]any{"model": "override"},
			},
			errMsg: "conflicts with top-level transcription parameters",
		},
		{
			name: "reserved file option",
			params: &core.TranscriptionParams{
				Audio:        []byte("data"),
				Filename:     "test.mp3",
				ModelOptions: map[string]any{"file": "override"},
			},
			errMsg: "conflicts with top-level transcription parameters",
		},
		{
			name: "reserved language option",
			params: &core.TranscriptionParams{
				Audio:        []byte("data"),
				Filename:     "test.mp3",
				ModelOptions: map[string]any{"language": "override"},
			},
			errMsg: "conflicts with top-level transcription parameters",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := adapter.Transcribe(context.Background(), tt.params)
			if err == nil {
				t.Fatalf("expected validation error containing %q", tt.errMsg)
			}
			if !strings.Contains(err.Error(), tt.errMsg) {
				t.Fatalf("expected error containing %q, got: %v", tt.errMsg, err)
			}
		})
	}
}

func TestBuildTranscriptionFormModelOptions(t *testing.T) {
	t.Parallel()

	params := &core.TranscriptionParams{
		Audio:    []byte("data"),
		Filename: "test.mp3",
		Language: "fr",
		ModelOptions: map[string]any{
			"responseFormat":         "verbose_json",
			"timestampGranularities": []string{"word", "segment"},
			"temperature":            0.5,
			"prompt":                 "technical terms",
		},
	}

	buf, contentType, err := buildTranscriptionForm("whisper-1", params)
	if err != nil {
		t.Fatalf("buildTranscriptionForm returned error: %v", err)
	}

	_, mimeParams, err := mime.ParseMediaType(contentType)
	if err != nil {
		t.Fatalf("parse content type: %v", err)
	}

	reader := multipart.NewReader(buf, mimeParams["boundary"])
	fields := make(map[string]string)
	var fileData []byte

	for {
		part, err := reader.NextPart()
		if err != nil {
			break
		}
		data, _ := io.ReadAll(part)
		if part.FormName() == "file" {
			fileData = data
		} else {
			fields[part.FormName()] = string(data)
		}
	}

	if fields["model"] != "whisper-1" {
		t.Fatalf("unexpected model: %q", fields["model"])
	}
	if fields["language"] != "fr" {
		t.Fatalf("unexpected language: %q", fields["language"])
	}
	if string(fileData) != "data" {
		t.Fatalf("unexpected file data: %q", string(fileData))
	}
	// responseFormat should be normalized to response_format
	if fields["response_format"] != "verbose_json" {
		t.Fatalf("expected response_format=verbose_json, got %q", fields["response_format"])
	}
	// timestampGranularities should be normalized to timestamp_granularities[]
	if fields["timestamp_granularities[]"] != "word,segment" {
		t.Fatalf("expected timestamp_granularities[]=word,segment, got %q", fields["timestamp_granularities[]"])
	}
}

func TestModelOptionToString(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		value    any
		expected string
	}{
		{"string", "hello", "hello"},
		{"int", 42, "42"},
		{"int64", int64(99), "99"},
		{"float64", 0.5, "0.5"},
		{"float32", float32(1.5), "1.5"},
		{"bool true", true, "true"},
		{"bool false", false, "false"},
		{"string slice", []string{"a", "b"}, "a,b"},
		{"json fallback", map[string]any{"key": "value"}, `{"key":"value"}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := modelOptionToString(tt.value)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result != tt.expected {
				t.Fatalf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestNormalizeTranscriptionModelOptionKey(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input    string
		expected string
	}{
		{"responseFormat", "response_format"},
		{"timestampGranularities", "timestamp_granularities[]"},
		{"temperature", "temperature"},
		{"prompt", "prompt"},
		{"customKey", "customKey"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := normalizeTranscriptionModelOptionKey(tt.input)
			if result != tt.expected {
				t.Fatalf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestToCoreTranscriptionResult(t *testing.T) {
	t.Parallel()

	t.Run("nil response", func(t *testing.T) {
		result := toCoreTranscriptionResult(nil)
		if result == nil {
			t.Fatal("expected non-nil result")
		}
		if result.Text != "" {
			t.Fatalf("expected empty text, got %q", result.Text)
		}
	})

	t.Run("full response with segments and words", func(t *testing.T) {
		resp := &transcriptionResponse{
			Text:     "Hello world.",
			Language: "en",
			Duration: 2.5,
			Segments: []transcriptionSegment{
				{
					Start: 0,
					End:   2.5,
					Text:  "Hello world.",
					Words: []transcriptionWord{
						{Word: "Hello", Start: 0, End: 1.0},
						{Word: "world.", Start: 1.1, End: 2.5},
					},
				},
			},
		}

		result := toCoreTranscriptionResult(resp)
		if result.Text != "Hello world." {
			t.Fatalf("unexpected text: %q", result.Text)
		}
		if len(result.Segments) != 1 {
			t.Fatalf("expected 1 segment, got %d", len(result.Segments))
		}
		if len(result.Segments[0].Words) != 2 {
			t.Fatalf("expected 2 words, got %d", len(result.Segments[0].Words))
		}
	})

	t.Run("top-level words without segments", func(t *testing.T) {
		resp := &transcriptionResponse{
			Text:     "Hi there.",
			Duration: 1.5,
			Words: []transcriptionWord{
				{Word: "Hi", Start: 0, End: 0.5},
				{Word: "there.", Start: 0.6, End: 1.5},
			},
		}

		result := toCoreTranscriptionResult(resp)
		if len(result.Segments) != 1 {
			t.Fatalf("expected 1 synthetic segment, got %d", len(result.Segments))
		}
		seg := result.Segments[0]
		if seg.Start != 0 || seg.End != 1.5 {
			t.Fatalf("unexpected segment timing: start=%v end=%v", seg.Start, seg.End)
		}
		if seg.Text != "Hi there." {
			t.Fatalf("unexpected segment text: %q", seg.Text)
		}
		if len(seg.Words) != 2 {
			t.Fatalf("expected 2 words, got %d", len(seg.Words))
		}
	})

	t.Run("segments take priority over top-level words", func(t *testing.T) {
		resp := &transcriptionResponse{
			Text:     "Hello.",
			Duration: 1.0,
			Segments: []transcriptionSegment{
				{Start: 0, End: 1.0, Text: "Hello."},
			},
			Words: []transcriptionWord{
				{Word: "Hello.", Start: 0, End: 1.0},
			},
		}

		result := toCoreTranscriptionResult(resp)
		// When both segments and words exist, segments take priority.
		// Top-level words are only used when no segments exist.
		if len(result.Segments) != 1 {
			t.Fatalf("expected 1 segment, got %d", len(result.Segments))
		}
		if len(result.Segments[0].Words) != 0 {
			t.Fatalf("expected 0 words in segment (segments have their own words), got %d", len(result.Segments[0].Words))
		}
	})
}

func TestTranscribeEmptyModel(t *testing.T) {
	t.Parallel()

	adapter := New("test-key", " ")
	_, err := adapter.Transcribe(context.Background(), &core.TranscriptionParams{
		Audio:    []byte("data"),
		Filename: "test.mp3",
	})
	if err == nil {
		t.Fatal("expected error for empty model")
	}
	if !strings.Contains(err.Error(), "model is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestTranscribeDuplicateModelOption(t *testing.T) {
	t.Parallel()

	// Use two keys that normalize to the same value
	_, _, err := buildTranscriptionForm("whisper-1", &core.TranscriptionParams{
		Audio:    []byte("data"),
		Filename: "test.mp3",
		ModelOptions: map[string]any{
			"response_format": "json",
			"responseFormat":  "verbose_json",
		},
	})
	if err == nil {
		t.Fatal("expected error for duplicate model option keys")
	}
	if !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestTranscribeNilResponse(t *testing.T) {
	t.Parallel()

	result := toCoreTranscriptionResult(nil)
	if result == nil {
		t.Fatal("expected non-nil result for nil response")
	}
	if result.Text != "" || result.Language != "" || result.Duration != 0 || len(result.Segments) != 0 {
		t.Fatalf("expected zero-value result, got %#v", result)
	}
}

// TestTranscribeResponseFormatPassthrough verifies the multipart form includes the correct
// response_format when the JSON verbose format is explicitly requested.
func TestTranscribeResponseFormatPassthrough(t *testing.T) {
	t.Parallel()

	var receivedFields map[string]string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		contentType := r.Header.Get("Content-Type")
		_, params, _ := mime.ParseMediaType(contentType)
		reader := multipart.NewReader(r.Body, params["boundary"])
		receivedFields = make(map[string]string)

		for {
			part, err := reader.NextPart()
			if err != nil {
				break
			}
			data, _ := io.ReadAll(part)
			if part.FormName() != "file" {
				receivedFields[part.FormName()] = string(data)
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"text":"ok"}`))
	}))
	defer server.Close()

	adapter := New("test-key", "whisper-1", WithBaseURL(server.URL), WithHTTPClient(server.Client()))

	_, err := adapter.Transcribe(context.Background(), &core.TranscriptionParams{
		Audio:    []byte("data"),
		Filename: "test.mp3",
		ModelOptions: map[string]any{
			"responseFormat": "verbose_json",
		},
	})
	if err != nil {
		t.Fatalf("transcribe returned error: %v", err)
	}

	if receivedFields["response_format"] != "verbose_json" {
		t.Fatalf("expected response_format=verbose_json, got %q", receivedFields["response_format"])
	}
}

// TestModelOptionToStringJSONFallback ensures the model option JSON fallback path works for
// non-primitive types by passing a struct-like value through the multipart form.
func TestModelOptionToStringJSONFallback(t *testing.T) {
	t.Parallel()

	type customType struct {
		Foo string `json:"foo"`
	}

	result, err := modelOptionToString(customType{Foo: "bar"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var decoded map[string]string
	if err := json.Unmarshal([]byte(result), &decoded); err != nil {
		t.Fatalf("result is not valid JSON: %v", err)
	}
	if decoded["foo"] != "bar" {
		t.Fatalf("unexpected JSON: %q", result)
	}
}
