package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/m43i/go-ai/core"
)

func toChatMessages(params *core.ChatParams) ([]chatMessage, error) {
	if params == nil {
		return nil, errors.New("openai: chat params are required")
	}

	out := make([]chatMessage, 0, len(params.Messages))
	for i, union := range params.Messages {
		message, err := toChatMessage(union)
		if err != nil {
			return nil, fmt.Errorf("openai: invalid message at index %d: %w", i, err)
		}
		out = append(out, message)
	}

	return out, nil
}

func toChatMessage(union core.MessageUnion) (chatMessage, error) {
	switch msg := union.(type) {
	case core.TextMessagePart:
		return newTextChatMessage(msg.Role, msg.Content)
	case *core.TextMessagePart:
		if msg == nil {
			return chatMessage{}, errors.New("text message is nil")
		}
		return newTextChatMessage(msg.Role, msg.Content)

	case core.ContentMessagePart:
		return newContentChatMessage(msg.Role, msg.Parts)
	case *core.ContentMessagePart:
		if msg == nil {
			return chatMessage{}, errors.New("content message is nil")
		}
		return newContentChatMessage(msg.Role, msg.Parts)

	case core.AssistantToolCallMessagePart:
		return newAssistantToolCallChatMessage(msg.Role, msg.ToolCalls)
	case *core.AssistantToolCallMessagePart:
		if msg == nil {
			return chatMessage{}, errors.New("assistant tool call message is nil")
		}
		return newAssistantToolCallChatMessage(msg.Role, msg.ToolCalls)

	case core.ToolResultMessagePart:
		return newToolResultChatMessage(msg.Role, msg.ToolCallID, msg.Content)
	case *core.ToolResultMessagePart:
		if msg == nil {
			return chatMessage{}, errors.New("tool result message is nil")
		}
		return newToolResultChatMessage(msg.Role, msg.ToolCallID, msg.Content)
	}

	return chatMessage{}, fmt.Errorf("unsupported message type %T", union)
}

func newTextChatMessage(role, content string) (chatMessage, error) {
	role = strings.TrimSpace(role)
	if role == "" {
		return chatMessage{}, errors.New("text message role is required")
	}

	return chatMessage{Role: role, Content: content}, nil
}

func newContentChatMessage(role string, parts []core.ContentPart) (chatMessage, error) {
	role = strings.TrimSpace(role)
	if role == "" {
		return chatMessage{}, errors.New("content message role is required")
	}

	contentParts, err := toChatContentParts(parts)
	if err != nil {
		return chatMessage{}, err
	}

	return chatMessage{Role: role, Content: contentParts}, nil
}

func toChatContentParts(parts []core.ContentPart) ([]chatContentPart, error) {
	if len(parts) == 0 {
		return nil, errors.New("content message must include at least one content part")
	}

	out := make([]chatContentPart, 0, len(parts))
	for i, part := range parts {
		contentPart, err := toChatContentPart(part)
		if err != nil {
			return nil, fmt.Errorf("content part at index %d: %w", i, err)
		}
		out = append(out, contentPart)
	}

	return out, nil
}

func toChatContentPart(part core.ContentPart) (chatContentPart, error) {
	switch typed := part.(type) {
	case core.TextPart:
		return chatContentPart{Type: "text", Text: typed.Text}, nil
	case *core.TextPart:
		if typed == nil {
			return chatContentPart{}, errors.New("text part is nil")
		}
		return chatContentPart{Type: "text", Text: typed.Text}, nil

	case core.ImagePart:
		return imageContentPart(typed.Source, typed.Metadata)
	case *core.ImagePart:
		if typed == nil {
			return chatContentPart{}, errors.New("image part is nil")
		}
		return imageContentPart(typed.Source, typed.Metadata)

	case core.AudioPart:
		return audioContentPart(typed.Source)
	case *core.AudioPart:
		if typed == nil {
			return chatContentPart{}, errors.New("audio part is nil")
		}
		return audioContentPart(typed.Source)

	case core.DocumentPart:
		return documentContentPart(typed.Source)
	case *core.DocumentPart:
		if typed == nil {
			return chatContentPart{}, errors.New("document part is nil")
		}
		return documentContentPart(typed.Source)
	}

	return chatContentPart{}, fmt.Errorf("unsupported content part type %T", part)
}

func imageContentPart(source core.Source, metadata map[string]any) (chatContentPart, error) {
	if source == nil {
		return chatContentPart{}, errors.New("image source is required")
	}

	url, err := imageURLFromSource(source)
	if err != nil {
		return chatContentPart{}, err
	}

	image := &chatImageURL{URL: url}
	if detail := imageDetail(metadata); detail != "" {
		image.Detail = detail
	}

	return chatContentPart{Type: "image_url", ImageURL: image}, nil
}

func audioContentPart(source core.Source) (chatContentPart, error) {
	if source == nil {
		return chatContentPart{}, errors.New("audio source is required")
	}

	data, format, err := audioPayloadFromSource(source)
	if err != nil {
		return chatContentPart{}, err
	}

	return chatContentPart{
		Type:       "input_audio",
		InputAudio: &chatInputAudio{Data: data, Format: format},
	}, nil
}

func documentContentPart(source core.Source) (chatContentPart, error) {
	if source == nil {
		return chatContentPart{}, errors.New("document source is required")
	}

	return chatContentPart{}, errors.New("openai: document content is not supported")
}

func imageDetail(metadata map[string]any) string {
	if metadata == nil {
		return ""
	}

	value, ok := metadata["detail"]
	if !ok {
		return ""
	}

	if detail, ok := value.(string); ok {
		return strings.TrimSpace(detail)
	}

	return ""
}

func imageURLFromSource(source core.Source) (string, error) {
	switch typed := source.(type) {
	case core.URLSource:
		return urlFromURLSource(typed)
	case *core.URLSource:
		if typed == nil {
			return "", errors.New("image URL source is nil")
		}
		return urlFromURLSource(*typed)

	case core.DataSource:
		return dataURLFromDataSource(typed)
	case *core.DataSource:
		if typed == nil {
			return "", errors.New("image data source is nil")
		}
		return dataURLFromDataSource(*typed)
	}

	return "", fmt.Errorf("unsupported image source type %T", source)
}

func urlFromURLSource(source core.URLSource) (string, error) {
	url := strings.TrimSpace(source.URL)
	if url == "" {
		return "", errors.New("image URL is required")
	}
	return url, nil
}

func dataURLFromDataSource(source core.DataSource) (string, error) {
	data := strings.TrimSpace(source.Data)
	if data == "" {
		return "", errors.New("image data is required")
	}
	if strings.HasPrefix(data, "data:") {
		return "", errors.New("image data must be raw base64")
	}

	mimeType := strings.TrimSpace(source.MimeType)
	if mimeType == "" {
		return "", errors.New("image mime type is required")
	}

	return fmt.Sprintf("data:%s;base64,%s", mimeType, data), nil
}

func audioPayloadFromSource(source core.Source) (string, string, error) {
	switch typed := source.(type) {
	case core.DataSource:
		return audioPayloadFromDataSource(typed)
	case *core.DataSource:
		if typed == nil {
			return "", "", errors.New("audio data source is nil")
		}
		return audioPayloadFromDataSource(*typed)
	}

	return "", "", fmt.Errorf("unsupported audio source type %T (only DataSource is supported)", source)
}

func audioPayloadFromDataSource(source core.DataSource) (string, string, error) {
	data := strings.TrimSpace(source.Data)
	if data == "" {
		return "", "", errors.New("audio data is required")
	}

	mimeType := strings.TrimSpace(source.MimeType)
	if mimeType == "" {
		return "", "", errors.New("audio mime type is required")
	}

	format := audioFormatFromMime(mimeType)
	if format == "" {
		return "", "", fmt.Errorf("unsupported audio mime type %q", mimeType)
	}

	return data, format, nil
}

func audioFormatFromMime(mimeType string) string {
	switch strings.ToLower(strings.TrimSpace(mimeType)) {
	case "audio/mp3", "audio/mpeg":
		return "mp3"
	case "audio/wav", "audio/wave", "audio/x-wav":
		return "wav"
	case "audio/flac":
		return "flac"
	case "audio/ogg":
		return "ogg"
	case "audio/webm":
		return "webm"
	default:
		return ""
	}
}

func newAssistantToolCallChatMessage(role string, toolCalls []core.ToolCall) (chatMessage, error) {
	role = strings.ToLower(strings.TrimSpace(role))
	if role == "" {
		role = core.RoleToolCall
	}
	if role != core.RoleToolCall && role != core.RoleAssistant {
		return chatMessage{}, fmt.Errorf("tool call message role must be %q or %q, got %q", core.RoleToolCall, core.RoleAssistant, role)
	}

	calls, err := toChatToolCalls(toolCalls)
	if err != nil {
		return chatMessage{}, err
	}

	return chatMessage{Role: core.RoleAssistant, ToolCalls: calls}, nil
}

func newToolResultChatMessage(role, toolCallID, content string) (chatMessage, error) {
	role = strings.ToLower(strings.TrimSpace(role))
	if role == "" {
		role = core.RoleToolResult
	}
	if role != core.RoleToolResult && role != "tool" && role != core.RoleUser {
		return chatMessage{}, fmt.Errorf("tool result message role must be %q, %q, or %q, got %q", core.RoleToolResult, "tool", core.RoleUser, role)
	}
	if strings.TrimSpace(toolCallID) == "" {
		return chatMessage{}, errors.New("tool result message tool call ID is required")
	}

	return chatMessage{
		Role:       "tool",
		ToolCallID: strings.TrimSpace(toolCallID),
		Content:    content,
	}, nil
}

func toChatToolCalls(calls []core.ToolCall) ([]chatToolCall, error) {
	if len(calls) == 0 {
		return nil, errors.New("assistant tool call message must include at least one tool call")
	}

	out := make([]chatToolCall, 0, len(calls))
	for i, call := range calls {
		name := strings.TrimSpace(call.Name)
		if name == "" {
			return nil, fmt.Errorf("tool call at index %d is missing a name", i)
		}

		id := strings.TrimSpace(call.ID)
		if id == "" {
			id = fmt.Sprintf("call_%d", i+1)
		}

		arguments, err := stringifyToolArguments(call.Arguments)
		if err != nil {
			return nil, fmt.Errorf("tool call %q arguments: %w", name, err)
		}

		out = append(out, chatToolCall{
			ID:   id,
			Type: "function",
			Function: chatToolCallFunction{
				Name:      name,
				Arguments: arguments,
			},
		})
	}

	return out, nil
}

func toCoreToolCalls(calls []chatToolCall) ([]core.ToolCall, error) {
	out := make([]core.ToolCall, 0, len(calls))
	for _, call := range calls {
		arguments, err := parseToolArguments(call.Function.Arguments)
		if err != nil {
			return nil, fmt.Errorf("openai: invalid arguments for tool %q: %w", call.Function.Name, err)
		}

		out = append(out, core.ToolCall{
			ID:        call.ID,
			Name:      call.Function.Name,
			Arguments: arguments,
		})
	}

	return out, nil
}

func stringifyToolArguments(arguments any) (string, error) {
	if arguments == nil {
		return "{}", nil
	}

	switch v := arguments.(type) {
	case string:
		trimmed := strings.TrimSpace(v)
		if trimmed == "" {
			return "{}", nil
		}
		var decoded any
		if err := json.Unmarshal([]byte(trimmed), &decoded); err == nil {
			return trimmed, nil
		}
	case json.RawMessage:
		trimmed := strings.TrimSpace(string(v))
		if trimmed == "" {
			return "{}", nil
		}
		var decoded any
		if err := json.Unmarshal([]byte(trimmed), &decoded); err == nil {
			return trimmed, nil
		}
	}

	b, err := json.Marshal(arguments)
	if err != nil {
		return "", err
	}

	return string(b), nil
}

func toChatTools(params *core.ChatParams) ([]chatTool, map[string]core.ServerTool, map[string]struct{}, error) {
	if params == nil || len(params.Tools) == 0 {
		return nil, nil, nil, nil
	}

	tools := make([]chatTool, 0, len(params.Tools))
	serverTools := make(map[string]core.ServerTool)
	clientTools := make(map[string]struct{})
	seenNames := make(map[string]struct{})

	for i, union := range params.Tools {
		switch tool := union.(type) {
		case core.ServerTool:
			def, serverTool, err := newServerChatTool(tool)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("openai: invalid server tool at index %d: %w", i, err)
			}
			if _, exists := seenNames[serverTool.Name]; exists {
				return nil, nil, nil, fmt.Errorf("openai: duplicate tool name %q", serverTool.Name)
			}
			seenNames[serverTool.Name] = struct{}{}
			tools = append(tools, def)
			serverTools[serverTool.Name] = serverTool

		case *core.ServerTool:
			if tool == nil {
				return nil, nil, nil, fmt.Errorf("openai: server tool at index %d is nil", i)
			}
			def, serverTool, err := newServerChatTool(*tool)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("openai: invalid server tool at index %d: %w", i, err)
			}
			if _, exists := seenNames[serverTool.Name]; exists {
				return nil, nil, nil, fmt.Errorf("openai: duplicate tool name %q", serverTool.Name)
			}
			seenNames[serverTool.Name] = struct{}{}
			tools = append(tools, def)
			serverTools[serverTool.Name] = serverTool

		case core.ClientTool:
			def, err := newClientChatTool(tool)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("openai: invalid client tool at index %d: %w", i, err)
			}
			if _, exists := seenNames[def.Function.Name]; exists {
				return nil, nil, nil, fmt.Errorf("openai: duplicate tool name %q", def.Function.Name)
			}
			seenNames[def.Function.Name] = struct{}{}
			tools = append(tools, def)
			clientTools[def.Function.Name] = struct{}{}

		case *core.ClientTool:
			if tool == nil {
				return nil, nil, nil, fmt.Errorf("openai: client tool at index %d is nil", i)
			}
			def, err := newClientChatTool(*tool)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("openai: invalid client tool at index %d: %w", i, err)
			}
			if _, exists := seenNames[def.Function.Name]; exists {
				return nil, nil, nil, fmt.Errorf("openai: duplicate tool name %q", def.Function.Name)
			}
			seenNames[def.Function.Name] = struct{}{}
			tools = append(tools, def)
			clientTools[def.Function.Name] = struct{}{}

		default:
			return nil, nil, nil, fmt.Errorf("openai: unsupported tool type %T", union)
		}
	}

	return tools, serverTools, clientTools, nil
}

func newServerChatTool(tool core.ServerTool) (chatTool, core.ServerTool, error) {
	name := strings.TrimSpace(tool.Name)
	if name == "" {
		return chatTool{}, core.ServerTool{}, errors.New("tool name is required")
	}
	if tool.Handler == nil {
		return chatTool{}, core.ServerTool{}, fmt.Errorf("tool %q handler is required", name)
	}

	tool.Name = name
	return chatToolFromDefinition(name, tool.Description, tool.Parameters), tool, nil
}

func newClientChatTool(tool core.ClientTool) (chatTool, error) {
	name := strings.TrimSpace(tool.Name)
	if name == "" {
		return chatTool{}, errors.New("tool name is required")
	}

	return chatToolFromDefinition(name, tool.Description, tool.Parameters), nil
}

func chatToolFromDefinition(name, description string, parameters map[string]any) chatTool {
	if parameters == nil {
		parameters = map[string]any{
			"type":                 "object",
			"properties":           map[string]any{},
			"additionalProperties": false,
		}
	}

	return chatTool{
		Type: "function",
		Function: chatToolFunction{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		},
	}
}

func maxTokens(params *core.ChatParams) *int64 {
	if params == nil {
		return nil
	}
	if params.MaxOutputTokens != nil {
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

func reasoningEffort(params *core.ChatParams) string {
	if params == nil {
		return ""
	}
	return strings.TrimSpace(params.ReasoningEffort)
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
