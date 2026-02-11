package core

import (
	"context"
	"testing"
)

type imageAdapterStub struct {
	generateImageFn func(context.Context, *ImageParams) (*ImageResult, error)
}

func (s imageAdapterStub) GenerateImage(ctx context.Context, params *ImageParams) (*ImageResult, error) {
	return s.generateImageFn(ctx, params)
}

func TestGenerateImage(t *testing.T) {
	expected := &ImageResult{Images: []GeneratedImage{{URL: "https://example.com/image.png"}}}
	adapter := imageAdapterStub{
		generateImageFn: func(_ context.Context, params *ImageParams) (*ImageResult, error) {
			if params == nil || params.Prompt != "a scenic valley" {
				t.Fatalf("unexpected params: %#v", params)
			}
			return expected, nil
		},
	}

	result, err := GenerateImage(context.Background(), adapter, &ImageParams{Prompt: "a scenic valley"})
	if err != nil {
		t.Fatalf("generate image returned error: %v", err)
	}
	if result != expected {
		t.Fatalf("expected result pointer %#v, got %#v", expected, result)
	}
}
