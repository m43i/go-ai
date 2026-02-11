package openai

import (
	"errors"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/m43i/go-ai/core"
)

const (
	defaultBaseURL         = "https://api.openai.com/v1"
	defaultMaxAgenticLoops = 8
	defaultHTTPTimeout     = 5 * time.Minute
)

type Adapter struct {
	APIKey     string
	Model      string
	BaseURL    string
	HTTPClient *http.Client
}

var _ core.TextAdapter = (*Adapter)(nil)
var _ core.EmbeddingAdapter = (*Adapter)(nil)
var _ core.ImageAdapter = (*Adapter)(nil)
var _ core.TranscriptionAdapter = (*Adapter)(nil)

type Option func(*Adapter)

// New creates an OpenAI adapter.
//
// Preferred usage is to use core and add this adapter there.
//
// If no API key is provided via options, New reads OPENAI_API_KEY from the environment.
func New(model string, opts ...Option) *Adapter {
	adapter := &Adapter{
		APIKey:     strings.TrimSpace(os.Getenv("OPENAI_API_KEY")),
		Model:      strings.TrimSpace(model),
		BaseURL:    defaultBaseURL,
		HTTPClient: &http.Client{Timeout: defaultHTTPTimeout},
	}

	for _, opt := range opts {
		if opt == nil {
			continue
		}
		opt(adapter)
	}

	return adapter
}

// WithAPIKey sets the API key used by the adapter.
func WithAPIKey(apiKey string) Option {
	return func(adapter *Adapter) {
		if strings.TrimSpace(apiKey) == "" {
			return
		}
		adapter.APIKey = strings.TrimSpace(apiKey)
	}
}

// WithBaseURL sets the API base URL used by the adapter.
func WithBaseURL(baseURL string) Option {
	return func(adapter *Adapter) {
		if strings.TrimSpace(baseURL) == "" {
			return
		}
		adapter.BaseURL = strings.TrimSpace(baseURL)
	}
}

// WithEndpointURL sets the API base URL used by the adapter.
//
// It is an alias for WithBaseURL.
func WithEndpointURL(endpointURL string) Option {
	return WithBaseURL(endpointURL)
}

// WithHTTPClient sets the HTTP client used by the adapter.
func WithHTTPClient(client *http.Client) Option {
	return func(adapter *Adapter) {
		if client == nil {
			return
		}
		adapter.HTTPClient = client
	}
}

// WithTimeout sets the timeout on the adapter HTTP client.
func WithTimeout(timeout time.Duration) Option {
	return func(adapter *Adapter) {
		if timeout <= 0 {
			return
		}
		if adapter.HTTPClient == nil {
			adapter.HTTPClient = &http.Client{}
		}
		adapter.HTTPClient.Timeout = timeout
	}
}

func (a *Adapter) validate() error {
	if a == nil {
		return errors.New("openai: adapter is nil")
	}

	if strings.TrimSpace(a.APIKey) == "" {
		a.APIKey = strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	}
	if strings.TrimSpace(a.APIKey) == "" {
		return errors.New("openai: API key is required (set OPENAI_API_KEY or use openai.WithAPIKey)")
	}

	if strings.TrimSpace(a.Model) == "" {
		return errors.New("openai: model is required")
	}

	return nil
}

func (a *Adapter) client() *http.Client {
	if a.HTTPClient != nil {
		return a.HTTPClient
	}
	return &http.Client{Timeout: defaultHTTPTimeout}
}

func (a *Adapter) baseURL() string {
	if strings.TrimSpace(a.BaseURL) == "" {
		return defaultBaseURL
	}
	return a.BaseURL
}
