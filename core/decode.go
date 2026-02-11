package core

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// LastAssistantText returns the final non-empty assistant text from a chat result.
//
// It prefers result.Text and falls back to scanning result.Messages in reverse order.
func LastAssistantText(result *ChatResult) (string, error) {
	if result == nil {
		return "", errors.New("chat result is nil")
	}

	if strings.TrimSpace(result.Text) != "" {
		return result.Text, nil
	}

	for i := len(result.Messages) - 1; i >= 0; i-- {
		switch message := result.Messages[i].(type) {
		case TextMessagePart:
			if strings.TrimSpace(message.Role) == RoleAssistant && strings.TrimSpace(message.Content) != "" {
				return message.Content, nil
			}
		case *TextMessagePart:
			if message != nil && strings.TrimSpace(message.Role) == RoleAssistant && strings.TrimSpace(message.Content) != "" {
				return message.Content, nil
			}
		}
	}

	return "", errors.New("no assistant text message found")
}

// DecodeLast decodes the final assistant text in result into T.
//
// The assistant text must be valid JSON for the target type.
func DecodeLast[T any](result *ChatResult) (T, error) {
	var out T

	text, err := LastAssistantText(result)
	if err != nil {
		return out, err
	}

	if err := json.Unmarshal([]byte(text), &out); err != nil {
		return out, fmt.Errorf("decode last assistant message: %w", err)
	}

	return out, nil
}

// DecodeLastInto decodes the final assistant text in result into out.
//
// The assistant text must be valid JSON for the target value.
func DecodeLastInto(result *ChatResult, out any) error {
	if out == nil {
		return errors.New("decode target is nil")
	}

	text, err := LastAssistantText(result)
	if err != nil {
		return err
	}

	if err := json.Unmarshal([]byte(text), out); err != nil {
		return fmt.Errorf("decode last assistant message: %w", err)
	}

	return nil
}
