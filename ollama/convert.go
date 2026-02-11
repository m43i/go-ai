package ollama

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/m43i/go-ai/core"
)

func toMessages(params *core.ChatParams) ([]message, error) {
	if params == nil {
		return nil, errors.New("ollama: chat params are required")
	}

	out := make([]message, 0, len(params.Messages))
	for i, union := range params.Messages {
		msg, err := toMessage(union)
		if err != nil {
			return nil, fmt.Errorf("ollama: invalid message at index %d: %w", i, err)
		}
		out = append(out, msg)
	}

	return out, nil
}

func toMessage(union core.MessageUnion) (message, error) {
	switch msg := union.(type) {
	case core.TextMessagePart:
		return textMessage(msg.Role, msg.Content)
	case *core.TextMessagePart:
		if msg == nil {
			return message{}, errors.New("text message is nil")
		}
		return textMessage(msg.Role, msg.Content)

	case core.ContentMessagePart:
		return contentMessage(msg.Role, msg.Parts)
	case *core.ContentMessagePart:
		if msg == nil {
			return message{}, errors.New("content message is nil")
		}
		return contentMessage(msg.Role, msg.Parts)

	case core.AssistantToolCallMessagePart:
		return assistantToolCallMessage(msg.Role, msg.ToolCalls)
	case *core.AssistantToolCallMessagePart:
		if msg == nil {
			return message{}, errors.New("assistant tool call message is nil")
		}
		return assistantToolCallMessage(msg.Role, msg.ToolCalls)

	case core.ToolResultMessagePart:
		return toolResultMessage(msg.Role, msg.ToolCallID, msg.Name, msg.Content)
	case *core.ToolResultMessagePart:
		if msg == nil {
			return message{}, errors.New("tool result message is nil")
		}
		return toolResultMessage(msg.Role, msg.ToolCallID, msg.Name, msg.Content)
	}

	return message{}, fmt.Errorf("unsupported message type %T", union)
}

func textMessage(role, content string) (message, error) {
	normalizedRole, err := normalizeRole(role)
	if err != nil {
		return message{}, err
	}

	return message{Role: normalizedRole, Content: content}, nil
}

func contentMessage(role string, parts []core.ContentPart) (message, error) {
	normalizedRole, err := normalizeRole(role)
	if err != nil {
		return message{}, err
	}
	if normalizedRole == "tool" {
		return message{}, errors.New("content messages cannot use tool role")
	}

	content, images, err := contentAndImages(parts)
	if err != nil {
		return message{}, err
	}

	return message{Role: normalizedRole, Content: content, Images: images}, nil
}

func contentAndImages(parts []core.ContentPart) (string, []string, error) {
	if len(parts) == 0 {
		return "", nil, errors.New("content message must include at least one content part")
	}

	var textBuilder strings.Builder
	images := make([]string, 0)

	for i, part := range parts {
		switch typed := part.(type) {
		case core.TextPart:
			textBuilder.WriteString(typed.Text)
		case *core.TextPart:
			if typed == nil {
				return "", nil, fmt.Errorf("content part at index %d: text part is nil", i)
			}
			textBuilder.WriteString(typed.Text)

		case core.ImagePart:
			imageData, err := imageDataFromSource(typed.Source)
			if err != nil {
				return "", nil, fmt.Errorf("content part at index %d: %w", i, err)
			}
			images = append(images, imageData)
		case *core.ImagePart:
			if typed == nil {
				return "", nil, fmt.Errorf("content part at index %d: image part is nil", i)
			}
			imageData, err := imageDataFromSource(typed.Source)
			if err != nil {
				return "", nil, fmt.Errorf("content part at index %d: %w", i, err)
			}
			images = append(images, imageData)

		case core.AudioPart, *core.AudioPart:
			return "", nil, fmt.Errorf("content part at index %d: ollama: audio content is not supported", i)
		case core.DocumentPart, *core.DocumentPart:
			return "", nil, fmt.Errorf("content part at index %d: ollama: document content is not supported", i)
		default:
			return "", nil, fmt.Errorf("content part at index %d: unsupported content part type %T", i, part)
		}
	}

	return textBuilder.String(), images, nil
}

func imageDataFromSource(source core.Source) (string, error) {
	if source == nil {
		return "", errors.New("image source is required")
	}

	switch typed := source.(type) {
	case core.DataSource:
		return dataImageSource(typed)
	case *core.DataSource:
		if typed == nil {
			return "", errors.New("image data source is nil")
		}
		return dataImageSource(*typed)

	case core.URLSource, *core.URLSource:
		return "", errors.New("image URL source is not supported (use DataSource with base64 image data)")
	}

	return "", fmt.Errorf("unsupported image source type %T", source)
}

func dataImageSource(source core.DataSource) (string, error) {
	data := strings.TrimSpace(source.Data)
	if data == "" {
		return "", errors.New("image data is required")
	}
	if strings.HasPrefix(strings.ToLower(data), "data:") {
		return "", errors.New("image data must be raw base64")
	}

	if strings.TrimSpace(source.MimeType) == "" {
		return "", errors.New("image mime type is required")
	}

	return data, nil
}

func assistantToolCallMessage(role string, calls []core.ToolCall) (message, error) {
	role = strings.ToLower(strings.TrimSpace(role))
	if role == "" {
		role = core.RoleToolCall
	}
	if role != core.RoleToolCall && role != core.RoleAssistant {
		return message{}, fmt.Errorf("tool call message role must be %q or %q, got %q", core.RoleToolCall, core.RoleAssistant, role)
	}

	toolCalls, err := toToolCalls(calls)
	if err != nil {
		return message{}, err
	}

	return message{Role: core.RoleAssistant, ToolCalls: toolCalls}, nil
}

func toolResultMessage(role, toolCallID, name, content string) (message, error) {
	role = strings.ToLower(strings.TrimSpace(role))
	if role == "" {
		role = core.RoleToolResult
	}
	if role != core.RoleToolResult && role != "tool" && role != core.RoleUser {
		return message{}, fmt.Errorf("tool result message role must be %q, %q, or %q, got %q", core.RoleToolResult, "tool", core.RoleUser, role)
	}
	if strings.TrimSpace(toolCallID) == "" {
		return message{}, errors.New("tool result message tool call ID is required")
	}

	out := message{
		Role:       "tool",
		ToolCallID: strings.TrimSpace(toolCallID),
		Content:    content,
	}
	if strings.TrimSpace(name) != "" {
		out.ToolName = strings.TrimSpace(name)
	}

	return out, nil
}

func toToolCalls(calls []core.ToolCall) ([]toolCall, error) {
	if len(calls) == 0 {
		return nil, errors.New("assistant tool call message must include at least one tool call")
	}

	out := make([]toolCall, 0, len(calls))
	for i, call := range calls {
		name := strings.TrimSpace(call.Name)
		if name == "" {
			return nil, fmt.Errorf("tool call at index %d is missing a name", i)
		}

		id := strings.TrimSpace(call.ID)
		if id == "" {
			id = fmt.Sprintf("call_%d", i+1)
		}

		arguments := call.Arguments
		if arguments == nil {
			arguments = map[string]any{}
		}

		out = append(out, toolCall{
			ID: id,
			Function: toolCallFunction{
				Index:     i,
				Name:      name,
				Arguments: arguments,
			},
		})
	}

	return out, nil
}

func toCoreToolCalls(calls []toolCall) ([]core.ToolCall, error) {
	out := make([]core.ToolCall, 0, len(calls))
	for i, call := range calls {
		name := strings.TrimSpace(call.Function.Name)
		if name == "" {
			return nil, fmt.Errorf("ollama: tool call at index %d is missing a function name", i)
		}

		id := strings.TrimSpace(call.ID)
		if id == "" {
			id = fmt.Sprintf("call_%d", i+1)
		}

		arguments, err := normalizeToolArguments(call.Function.Arguments)
		if err != nil {
			return nil, fmt.Errorf("ollama: invalid arguments for tool %q: %w", name, err)
		}

		out = append(out, core.ToolCall{
			ID:        id,
			Name:      name,
			Arguments: arguments,
		})
	}

	return out, nil
}

func normalizeToolArguments(arguments any) (any, error) {
	if arguments == nil {
		return map[string]any{}, nil
	}

	switch typed := arguments.(type) {
	case string:
		trimmed := strings.TrimSpace(typed)
		if trimmed == "" {
			return map[string]any{}, nil
		}
		var decoded any
		if err := json.Unmarshal([]byte(trimmed), &decoded); err == nil {
			return decoded, nil
		}
		return typed, nil

	case json.RawMessage:
		trimmed := strings.TrimSpace(string(typed))
		if trimmed == "" {
			return map[string]any{}, nil
		}
		var decoded any
		if err := json.Unmarshal([]byte(trimmed), &decoded); err != nil {
			return nil, err
		}
		return decoded, nil
	}

	return arguments, nil
}

func normalizeRole(role string) (string, error) {
	normalized := strings.ToLower(strings.TrimSpace(role))
	if normalized == "" {
		return "", errors.New("message role is required")
	}

	switch normalized {
	case core.RoleSystem, core.RoleUser, core.RoleAssistant:
		return normalized, nil
	case core.RoleToolResult, "tool":
		return "tool", nil
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
				return nil, nil, nil, fmt.Errorf("ollama: invalid server tool at index %d: %w", i, err)
			}
			if err := assertNewToolName(seenNames, serverTool.Name); err != nil {
				return nil, nil, nil, err
			}
			tools = append(tools, definition)
			serverTools[serverTool.Name] = serverTool

		case *core.ServerTool:
			if toolValue == nil {
				return nil, nil, nil, fmt.Errorf("ollama: server tool at index %d is nil", i)
			}
			definition, serverTool, err := newServerTool(*toolValue)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("ollama: invalid server tool at index %d: %w", i, err)
			}
			if err := assertNewToolName(seenNames, serverTool.Name); err != nil {
				return nil, nil, nil, err
			}
			tools = append(tools, definition)
			serverTools[serverTool.Name] = serverTool

		case core.ClientTool:
			definition, err := newClientTool(toolValue)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("ollama: invalid client tool at index %d: %w", i, err)
			}
			if err := assertNewToolName(seenNames, definition.Function.Name); err != nil {
				return nil, nil, nil, err
			}
			tools = append(tools, definition)
			clientTools[definition.Function.Name] = struct{}{}

		case *core.ClientTool:
			if toolValue == nil {
				return nil, nil, nil, fmt.Errorf("ollama: client tool at index %d is nil", i)
			}
			definition, err := newClientTool(*toolValue)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("ollama: invalid client tool at index %d: %w", i, err)
			}
			if err := assertNewToolName(seenNames, definition.Function.Name); err != nil {
				return nil, nil, nil, err
			}
			tools = append(tools, definition)
			clientTools[definition.Function.Name] = struct{}{}

		default:
			return nil, nil, nil, fmt.Errorf("ollama: unsupported tool type %T", union)
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

func newToolDefinition(name, description string, parameters map[string]any) tool {
	if parameters == nil {
		parameters = map[string]any{
			"type":                 "object",
			"properties":           map[string]any{},
			"additionalProperties": false,
		}
	}

	return tool{
		Type: "function",
		Function: toolFunction{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		},
	}
}

func assertNewToolName(seen map[string]struct{}, name string) error {
	if _, exists := seen[name]; exists {
		return fmt.Errorf("ollama: duplicate tool name %q", name)
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

func requestOptions(params *core.ChatParams) map[string]any {
	if params == nil {
		return nil
	}

	options := map[string]any{}
	if max := maxTokens(params); max != nil {
		options["num_predict"] = *max
	}
	if temp := temperature(params); temp != nil {
		options["temperature"] = *temp
	}

	if len(options) == 0 {
		return nil
	}

	return options
}

func thinkValue(params *core.ChatParams) any {
	if params == nil {
		return nil
	}

	raw := strings.TrimSpace(params.Thinking)
	if raw == "" {
		raw = strings.TrimSpace(params.ReasoningEffort)
	}
	if raw == "" {
		return nil
	}

	lower := strings.ToLower(raw)
	switch lower {
	case "true":
		return true
	case "false":
		return false
	default:
		return lower
	}
}

func formatFromOutput(output *core.Schema) (json.RawMessage, error) {
	if output == nil {
		return nil, nil
	}
	if output.Schema == nil {
		return nil, errors.New("ollama: output schema is required")
	}

	payload, err := json.Marshal(output.Schema)
	if err != nil {
		return nil, fmt.Errorf("ollama: marshal output schema: %w", err)
	}

	return payload, nil
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
