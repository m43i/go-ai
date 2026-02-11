package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/m43i/go-ai/core"
)

func TestGenerateImage(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/images/generations" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("unexpected method %q", r.Method)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("unexpected authorization header %q", got)
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read request body: %v", err)
		}

		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("decode request body: %v", err)
		}

		if payload["model"] != "gpt-image-1" {
			t.Fatalf("unexpected model payload: %#v", payload["model"])
		}
		if payload["prompt"] != "A watercolor fox in a forest" {
			t.Fatalf("unexpected prompt payload: %#v", payload["prompt"])
		}
		if payload["n"] != float64(2) {
			t.Fatalf("unexpected n payload: %#v", payload["n"])
		}
		if payload["size"] != "1024x1024" {
			t.Fatalf("unexpected size payload: %#v", payload["size"])
		}
		if payload["quality"] != "high" {
			t.Fatalf("unexpected quality payload: %#v", payload["quality"])
		}
		if payload["output_format"] != "png" {
			t.Fatalf("unexpected output_format payload: %#v", payload["output_format"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"created":1700000001,"data":[{"url":"https://example.com/first.png","b64_json":"ZmFrZQ==","revised_prompt":"A watercolor fox in a pine forest"},{"url":"https://example.com/second.png"}],"usage":{"input_tokens":11,"output_tokens":22,"total_tokens":33}}`))
	}))
	defer server.Close()

	numberOfImages := int64(2)
	adapter := New("test-key", "gpt-image-1", WithBaseURL(server.URL), WithHTTPClient(server.Client()))

	result, err := adapter.GenerateImage(context.Background(), &core.ImageParams{
		Prompt:         "A watercolor fox in a forest",
		NumberOfImages: &numberOfImages,
		Size:           "1024x1024",
		ModelOptions: map[string]any{
			"quality":      "high",
			"outputFormat": "png",
		},
	})
	if err != nil {
		t.Fatalf("generate image returned error: %v", err)
	}

	if result.ID != "img_1700000001" {
		t.Fatalf("unexpected result id: %q", result.ID)
	}
	if result.Model != "gpt-image-1" {
		t.Fatalf("unexpected model: %q", result.Model)
	}
	if len(result.Images) != 2 {
		t.Fatalf("expected 2 generated images, got %d", len(result.Images))
	}
	if result.Images[0].URL != "https://example.com/first.png" {
		t.Fatalf("unexpected first image url: %#v", result.Images[0])
	}
	if result.Images[0].B64JSON != "ZmFrZQ==" {
		t.Fatalf("unexpected first image b64_json: %#v", result.Images[0])
	}
	if result.Images[0].RevisedPrompt != "A watercolor fox in a pine forest" {
		t.Fatalf("unexpected first image revised prompt: %#v", result.Images[0])
	}
	if result.Usage == nil {
		t.Fatal("expected usage to be present")
	}
	if result.Usage.InputTokens != 11 || result.Usage.OutputTokens != 22 || result.Usage.TotalTokens != 33 {
		t.Fatalf("unexpected usage payload: %#v", result.Usage)
	}
}

func TestGenerateImageValidatesPrompt(t *testing.T) {
	t.Parallel()

	adapter := New("test-key", "gpt-image-1")
	_, err := adapter.GenerateImage(context.Background(), &core.ImageParams{Prompt: "   "})
	if err == nil {
		t.Fatal("expected prompt validation error")
	}
	if !strings.Contains(err.Error(), "prompt is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestGenerateImageValidatesModelOptionConflicts(t *testing.T) {
	t.Parallel()

	adapter := New("test-key", "gpt-image-1")
	_, err := adapter.GenerateImage(context.Background(), &core.ImageParams{
		Prompt: "A red bicycle",
		ModelOptions: map[string]any{
			"prompt": "override",
		},
	})
	if err == nil {
		t.Fatal("expected conflict error for reserved model option")
	}
	if !strings.Contains(err.Error(), "conflicts with top-level image parameters") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestGenerateImageRejectsEmptyResponseImages(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"created":1700000001,"data":[]}`))
	}))
	defer server.Close()

	adapter := New("test-key", "gpt-image-1", WithBaseURL(server.URL), WithHTTPClient(server.Client()))
	_, err := adapter.GenerateImage(context.Background(), &core.ImageParams{Prompt: "A red bicycle"})
	if err == nil {
		t.Fatal("expected empty response image error")
	}
	if !strings.Contains(err.Error(), "did not include any images") {
		t.Fatalf("unexpected error: %v", err)
	}
}
