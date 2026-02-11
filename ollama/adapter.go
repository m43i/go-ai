package ollama

import (
	"errors"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/m43i/go-ai/core"
)

const (
	defaultBaseURL         = "http://localhost:11434"
	defaultMaxAgenticLoops = 8
	defaultHTTPTimeout     = 5 * time.Minute
	envOllamaHost          = "OLLAMA_HOST"
	envOllamaAPIKey        = "OLLAMA_API_KEY"
)

type Adapter struct {
	APIKey     string
	Model      string
	BaseURL    string
	HTTPClient *http.Client
}

var _ core.TextAdapter = (*Adapter)(nil)
var _ core.EmbeddingAdapter = (*Adapter)(nil)

type Option func(*Adapter)

// New creates an Ollama adapter.
//
// Preferred usage is to use core and add this adapter there.
//
// If no base URL is provided via options, New reads OLLAMA_HOST and falls back
// to http://localhost:11434.
func New(model string, opts ...Option) *Adapter {
	baseURL := strings.TrimSpace(os.Getenv(envOllamaHost))
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	adapter := &Adapter{
		APIKey:     strings.TrimSpace(os.Getenv(envOllamaAPIKey)),
		Model:      strings.TrimSpace(model),
		BaseURL:    baseURL,
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

// WithAPIKey sets the optional API key used by the adapter.
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
		return errors.New("ollama: adapter is nil")
	}

	if strings.TrimSpace(a.Model) == "" {
		return errors.New("ollama: model is required")
	}

	if strings.TrimSpace(a.APIKey) == "" {
		a.APIKey = strings.TrimSpace(os.Getenv(envOllamaAPIKey))
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
		if host := strings.TrimSpace(os.Getenv(envOllamaHost)); host != "" {
			return host
		}
		return defaultBaseURL
	}
	return a.BaseURL
}
