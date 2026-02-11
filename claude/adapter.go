package claude

import (
	"errors"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/m43i/go-ai/core"
)

const (
	defaultBaseURL         = "https://api.anthropic.com/v1"
	defaultMaxAgenticLoops = 8
	defaultHTTPTimeout     = 5 * time.Minute
	envAnthropicAPIKey     = "ANTHROPIC_API_KEY"
	envClaudeAPIKey        = "CLAUDE_API_KEY"
)

type Adapter struct {
	APIKey           string
	Model            string
	BaseURL          string
	AnthropicVersion string
	HTTPClient       *http.Client
}

var _ core.TextAdapter = (*Adapter)(nil)

type Option func(*Adapter)

// New creates a Claude adapter.
//
// Preferred usage is to use core and add this adapter there.
//
// If apiKey is empty, New reads ANTHROPIC_API_KEY and then CLAUDE_API_KEY.
func New(apiKey, model string, opts ...Option) *Adapter {
	if strings.TrimSpace(apiKey) == "" {
		apiKey = resolveAPIKey()
	}

	adapter := &Adapter{
		APIKey:     strings.TrimSpace(apiKey),
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

// WithAnthropicVersion sets the anthropic-version request header value.
func WithAnthropicVersion(version string) Option {
	return func(adapter *Adapter) {
		if strings.TrimSpace(version) == "" {
			return
		}
		adapter.AnthropicVersion = strings.TrimSpace(version)
	}
}

func (a *Adapter) validate() error {
	if a == nil {
		return errors.New("claude: adapter is nil")
	}

	if strings.TrimSpace(a.APIKey) == "" {
		a.APIKey = resolveAPIKey()
	}
	if strings.TrimSpace(a.APIKey) == "" {
		return errors.New("claude: API key is required (set ANTHROPIC_API_KEY/CLAUDE_API_KEY or pass apiKey)")
	}

	if strings.TrimSpace(a.Model) == "" {
		return errors.New("claude: model is required")
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

func (a *Adapter) version() string {
	return strings.TrimSpace(a.AnthropicVersion)
}

func resolveAPIKey() string {
	key := strings.TrimSpace(os.Getenv(envAnthropicAPIKey))
	if key != "" {
		return key
	}
	return strings.TrimSpace(os.Getenv(envClaudeAPIKey))
}
