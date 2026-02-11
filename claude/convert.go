package claude

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/m43i/go-ai/core"
)

func toMessagesAndSystem(params *core.ChatParams) ([]message, string, error) {
	if params == nil {
		return nil, "", errors.New("claude: chat params are required")
	}

	messages := make([]message, 0, len(params.Messages))
	systemParts := make([]string, 0)

	for i, union := range params.Messages {
		msg, systemText, err := toMessage(union)
		if err != nil {
			return nil, "", fmt.Errorf("claude: invalid message at index %d: %w", i, err)
		}
		if systemText != "" {
			systemParts = append(systemParts, systemText)
		}
		if msg != nil {
			messages = append(messages, *msg)
		}
	}

	return messages, strings.Join(systemParts, "\n\n"), nil
}

func toMessage(union core.MessageUnion) (*message, string, error) {
	switch msg := union.(type) {
	case core.TextMessagePart:
		return textMessage(msg.Role, msg.Content)
	case *core.TextMessagePart:
		if msg == nil {
			return nil, "", errors.New("text message is nil")
		}
		return textMessage(msg.Role, msg.Content)

	case core.ContentMessagePart:
		return contentMessage(msg.Role, msg.Parts)
	case *core.ContentMessagePart:
		if msg == nil {
			return nil, "", errors.New("content message is nil")
		}
		return contentMessage(msg.Role, msg.Parts)

	case core.AssistantToolCallMessagePart:
		return assistantToolCallMessage(msg.Role, msg.ToolCalls)
	case *core.AssistantToolCallMessagePart:
		if msg == nil {
			return nil, "", errors.New("assistant tool call message is nil")
		}
		return assistantToolCallMessage(msg.Role, msg.ToolCalls)

	case core.ToolResultMessagePart:
		return toolResultMessage(msg.Role, msg.ToolCallID, msg.Content)
	case *core.ToolResultMessagePart:
		if msg == nil {
			return nil, "", errors.New("tool result message is nil")
		}
		return toolResultMessage(msg.Role, msg.ToolCallID, msg.Content)
	}

	return nil, "", fmt.Errorf("unsupported message type %T", union)
}

func textMessage(role, content string) (*message, string, error) {
	normalizedRole, err := normalizeRole(role)
	if err != nil {
		return nil, "", err
	}

	if normalizedRole == "system" {
		return nil, content, nil
	}

	return &message{
		Role: normalizedRole,
		Content: []contentBlock{
			{Type: "text", Text: content},
		},
	}, "", nil
}

func contentMessage(role string, parts []core.ContentPart) (*message, string, error) {
	normalizedRole, err := normalizeRole(role)
	if err != nil {
		return nil, "", err
	}
	if normalizedRole == "system" {
		return nil, "", errors.New("content messages cannot use system role")
	}

	blocks, err := toContentBlocks(parts)
	if err != nil {
		return nil, "", err
	}

	return &message{Role: normalizedRole, Content: blocks}, "", nil
}

func toContentBlocks(parts []core.ContentPart) ([]contentBlock, error) {
	if len(parts) == 0 {
		return nil, errors.New("content message must include at least one content part")
	}

	out := make([]contentBlock, 0, len(parts))
	for i, part := range parts {
		block, err := toContentBlock(part)
		if err != nil {
			return nil, fmt.Errorf("content part at index %d: %w", i, err)
		}
		out = append(out, block)
	}

	return out, nil
}

func toContentBlock(part core.ContentPart) (contentBlock, error) {
	switch typed := part.(type) {
	case core.TextPart:
		return contentBlock{Type: "text", Text: typed.Text}, nil
	case *core.TextPart:
		if typed == nil {
			return contentBlock{}, errors.New("text part is nil")
		}
		return contentBlock{Type: "text", Text: typed.Text}, nil

	case core.ImagePart:
		return imageBlock(typed.Source)
	case *core.ImagePart:
		if typed == nil {
			return contentBlock{}, errors.New("image part is nil")
		}
		return imageBlock(typed.Source)

	case core.AudioPart:
		return audioBlock(typed.Source)
	case *core.AudioPart:
		if typed == nil {
			return contentBlock{}, errors.New("audio part is nil")
		}
		return audioBlock(typed.Source)

	case core.DocumentPart:
		return documentBlock(typed.Source)
	case *core.DocumentPart:
		if typed == nil {
			return contentBlock{}, errors.New("document part is nil")
		}
		return documentBlock(typed.Source)
	}

	return contentBlock{}, fmt.Errorf("unsupported content part type %T", part)
}

func imageBlock(source core.Source) (contentBlock, error) {
	if source == nil {
		return contentBlock{}, errors.New("image source is required")
	}

	ms, err := mediaSourceFromSource(source)
	if err != nil {
		return contentBlock{}, err
	}

	return contentBlock{Type: "image", Source: ms}, nil
}

func audioBlock(source core.Source) (contentBlock, error) {
	if source == nil {
		return contentBlock{}, errors.New("audio source is required")
	}

	ms, err := mediaSourceFromSource(source)
	if err != nil {
		return contentBlock{}, err
	}

	return contentBlock{Type: "audio", Source: ms}, nil
}

func documentBlock(source core.Source) (contentBlock, error) {
	if source == nil {
		return contentBlock{}, errors.New("document source is required")
	}

	ms, err := mediaSourceFromSource(source)
	if err != nil {
		return contentBlock{}, err
	}

	return contentBlock{Type: "document", Source: ms}, nil
}

func mediaSourceFromSource(source core.Source) (*mediaSource, error) {
	switch typed := source.(type) {
	case core.URLSource:
		return urlMediaSource(typed)
	case *core.URLSource:
		if typed == nil {
			return nil, errors.New("URL source is nil")
		}
		return urlMediaSource(*typed)

	case core.DataSource:
		return dataMediaSource(typed)
	case *core.DataSource:
		if typed == nil {
			return nil, errors.New("data source is nil")
		}
		return dataMediaSource(*typed)
	}

	return nil, fmt.Errorf("unsupported source type %T", source)
}

func urlMediaSource(source core.URLSource) (*mediaSource, error) {
	url := strings.TrimSpace(source.URL)
	if url == "" {
		return nil, errors.New("source URL is required")
	}

	return &mediaSource{Type: "url", URL: url}, nil
}

func dataMediaSource(source core.DataSource) (*mediaSource, error) {
	data := strings.TrimSpace(source.Data)
	if data == "" {
		return nil, errors.New("source data is required")
	}

	mimeType := strings.TrimSpace(source.MimeType)
	if mimeType == "" {
		return nil, errors.New("source mime type is required")
	}

	return &mediaSource{Type: "base64", MediaType: mimeType, Data: data}, nil
}

func assistantToolCallMessage(role string, calls []core.ToolCall) (*message, string, error) {
	role = strings.TrimSpace(strings.ToLower(role))
	if role == "" {
		role = core.RoleToolCall
	}
	if role != core.RoleToolCall && role != core.RoleAssistant {
		return nil, "", fmt.Errorf("tool call message role must be %q or %q, got %q", core.RoleToolCall, core.RoleAssistant, role)
	}
	if len(calls) == 0 {
		return nil, "", errors.New("assistant tool call message must include at least one tool call")
	}

	blocks := make([]contentBlock, 0, len(calls))
	for i, call := range calls {
		name := strings.TrimSpace(call.Name)
		if name == "" {
			return nil, "", fmt.Errorf("tool call at index %d is missing a name", i)
		}

		id := strings.TrimSpace(call.ID)
		if id == "" {
			id = fmt.Sprintf("call_%d", i+1)
		}

		input := call.Arguments
		if input == nil {
			input = map[string]any{}
		}

		blocks = append(blocks, contentBlock{
			Type:  "tool_use",
			ID:    id,
			Name:  name,
			Input: input,
		})
	}

	return &message{Role: "assistant", Content: blocks}, "", nil
}

func toolResultMessage(role, toolCallID, content string) (*message, string, error) {
	role = strings.TrimSpace(strings.ToLower(role))
	if role == "" {
		role = core.RoleToolResult
	}
	if role != core.RoleToolResult && role != "tool" && role != core.RoleUser {
		return nil, "", fmt.Errorf("tool result message role must be %q, %q, or %q, got %q", core.RoleToolResult, "tool", core.RoleUser, role)
	}
	if strings.TrimSpace(toolCallID) == "" {
		return nil, "", errors.New("tool result message tool call ID is required")
	}

	return &message{
		Role: "user",
		Content: []contentBlock{
			{
				Type:      "tool_result",
				ToolUseID: strings.TrimSpace(toolCallID),
				Content:   content,
			},
		},
	}, "", nil
}

func toCoreToolCalls(blocks []contentBlock) []core.ToolCall {
	out := make([]core.ToolCall, 0, len(blocks))
	for _, block := range blocks {
		if block.Type != "tool_use" {
			continue
		}
		out = append(out, core.ToolCall{
			ID:        block.ID,
			Name:      block.Name,
			Arguments: block.Input,
		})
	}
	return out
}

func normalizeRole(role string) (string, error) {
	normalized := strings.ToLower(strings.TrimSpace(role))
	if normalized == "" {
		return "", errors.New("message role is required")
	}

	switch normalized {
	case "user", "assistant", "system":
		return normalized, nil
	default:
		return "", fmt.Errorf("unsupported role %q", role)
	}
}

func toTools(params *core.ChatParams) ([]tool, map[string]core.ServerTool, map[string]struct{}, error) {
	if params == nil || len(params.Tools) == 0 {
		return nil, nil, nil, nil
	}

	tools := make([]tool, 0, len(params.Tools))
	serverTools := make(map[string]core.ServerTool)
	clientTools := make(map[string]struct{})
	seenNames := make(map[string]struct{})

	for i, union := range params.Tools {
		switch toolValue := union.(type) {
		case core.ServerTool:
			definition, serverTool, err := newServerTool(toolValue)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("claude: invalid server tool at index %d: %w", i, err)
			}
			if err := assertNewToolName(seenNames, serverTool.Name); err != nil {
				return nil, nil, nil, err
			}
			tools = append(tools, definition)
			serverTools[serverTool.Name] = serverTool

		case *core.ServerTool:
			if toolValue == nil {
				return nil, nil, nil, fmt.Errorf("claude: server tool at index %d is nil", i)
			}
			definition, serverTool, err := newServerTool(*toolValue)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("claude: invalid server tool at index %d: %w", i, err)
			}
			if err := assertNewToolName(seenNames, serverTool.Name); err != nil {
				return nil, nil, nil, err
			}
			tools = append(tools, definition)
			serverTools[serverTool.Name] = serverTool

		case core.ClientTool:
			definition, err := newClientTool(toolValue)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("claude: invalid client tool at index %d: %w", i, err)
			}
			if err := assertNewToolName(seenNames, definition.Name); err != nil {
				return nil, nil, nil, err
			}
			tools = append(tools, definition)
			clientTools[definition.Name] = struct{}{}

		case *core.ClientTool:
			if toolValue == nil {
				return nil, nil, nil, fmt.Errorf("claude: client tool at index %d is nil", i)
			}
			definition, err := newClientTool(*toolValue)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("claude: invalid client tool at index %d: %w", i, err)
			}
			if err := assertNewToolName(seenNames, definition.Name); err != nil {
				return nil, nil, nil, err
			}
			tools = append(tools, definition)
			clientTools[definition.Name] = struct{}{}

		default:
			return nil, nil, nil, fmt.Errorf("claude: unsupported tool type %T", union)
		}
	}

	return tools, serverTools, clientTools, nil
}

func newServerTool(toolValue core.ServerTool) (tool, core.ServerTool, error) {
	name := strings.TrimSpace(toolValue.Name)
	if name == "" {
		return tool{}, core.ServerTool{}, errors.New("tool name is required")
	}
	if toolValue.Handler == nil {
		return tool{}, core.ServerTool{}, fmt.Errorf("tool %q handler is required", name)
	}

	toolValue.Name = name
	return newToolDefinition(name, toolValue.Description, toolValue.Parameters), toolValue, nil
}

func newClientTool(toolValue core.ClientTool) (tool, error) {
	name := strings.TrimSpace(toolValue.Name)
	if name == "" {
		return tool{}, errors.New("tool name is required")
	}

	return newToolDefinition(name, toolValue.Description, toolValue.Parameters), nil
}

func newToolDefinition(name, description string, inputSchema map[string]any) tool {
	if inputSchema == nil {
		inputSchema = map[string]any{
			"type":                 "object",
			"properties":           map[string]any{},
			"additionalProperties": false,
		}
	}

	return tool{Name: name, Description: description, InputSchema: inputSchema}
}

func assertNewToolName(seen map[string]struct{}, name string) error {
	if _, exists := seen[name]; exists {
		return fmt.Errorf("claude: duplicate tool name %q", name)
	}
	seen[name] = struct{}{}
	return nil
}

func maxTokens(params *core.ChatParams) *int64 {
	if params == nil {
		return nil
	}
	if params.MaxOutputTokens != nil && *params.MaxOutputTokens > 0 {
		return params.MaxOutputTokens
	}
	if params.MaxLength > 0 {
		value := params.MaxLength
		return &value
	}
	return nil
}

func temperature(params *core.ChatParams) *float64 {
	if params == nil {
		return nil
	}
	return params.Temperature
}

func maxLoops(params *core.ChatParams, hasServerTools bool) int {
	if !hasServerTools {
		return 1
	}
	if params != nil && params.MaxAgenticLoops > 0 {
		return int(params.MaxAgenticLoops)
	}
	return defaultMaxAgenticLoops
}

func outputInstruction(schema *core.Schema) string {
	if schema == nil || schema.Schema == nil {
		return ""
	}

	b, err := json.MarshalIndent(schema.Schema, "", "  ")
	if err != nil {
		return ""
	}

	return "Return only valid JSON that strictly matches this JSON Schema:\n" + string(b)
}
