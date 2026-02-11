package core

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"
)

type Schema struct {
	Name   string
	Strict bool
	Schema map[string]any
}

type responseFormatJSONSchema struct {
	Type       string           `json:"type"`
	JSONSchema jsonSchemaObject `json:"json_schema"`
}

type jsonSchemaObject struct {
	Name   string         `json:"name"`
	Strict bool           `json:"strict"`
	Schema map[string]any `json:"schema"`
}

// NewSchema builds a strict JSON schema from a struct type.
//
// name identifies the schema in provider requests, and v must be a struct value
// or pointer to a struct value.
func NewSchema(name string, v any) (Schema, error) {
	if name == "" {
		return Schema{}, errors.New("schema name must not be empty")
	}

	t := reflect.TypeOf(v)
	if t == nil {
		return Schema{}, errors.New("schema value is nil (pass a struct value)")
	}

	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}

	if t.Kind() != reflect.Struct {
		return Schema{}, fmt.Errorf("schema must be built from a struct, got %s", t.Kind())
	}

	visited := map[reflect.Type]bool{}
	root, err := schemaForType(t, visited)
	if err != nil {
		return Schema{}, err
	}

	return Schema{
		Name:   name,
		Strict: true,
		Schema: root,
	}, nil
}

// MarshalJSON encodes Schema into the response_format payload expected by chat APIs.
func (s Schema) MarshalJSON() ([]byte, error) {
	if s.Name == "" || s.Schema == nil {
		return nil, errors.New("invalid schema: missing Name or Schema")
	}

	payload := responseFormatJSONSchema{
		Type: "json_schema",
		JSONSchema: jsonSchemaObject{
			Name:   s.Name,
			Strict: s.Strict,
			Schema: s.Schema,
		},
	}

	return json.Marshal(payload)
}

// String returns the schema JSON representation with indentation.
func (s Schema) String() string {
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return ""
	}
	return string(b)
}

var timeType = reflect.TypeFor[time.Time]()

func schemaForType(t reflect.Type, visited map[reflect.Type]bool) (map[string]any, error) {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}

	if t == timeType {
		return map[string]any{
			"type":   "string",
			"format": "date-time",
		}, nil
	}

	switch t.Kind() {
	case reflect.Struct:
		if visited[t] {
			return nil, fmt.Errorf("recursive type not supported: %s", t.String())
		}
		visited[t] = true
		defer delete(visited, t)

		props := map[string]any{}
		required := make([]string, 0)

		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if !f.IsExported() {
				continue
			}

			name, omitempty, skip := parseJSONTag(f)
			if skip {
				continue
			}

			fieldType := f.Type
			isPtr := fieldType.Kind() == reflect.Pointer

			fieldSchema, err := schemaForType(fieldType, visited)
			if err != nil {
				return nil, fmt.Errorf("field %s: %w", f.Name, err)
			}

			if desc := f.Tag.Get("description"); desc != "" {
				fieldSchema["description"] = desc
			}

			props[name] = fieldSchema

			if !omitempty && !isPtr {
				required = append(required, name)
			}
		}

		out := map[string]any{
			"type":                 "object",
			"properties":           props,
			"additionalProperties": false,
		}
		if len(required) > 0 {
			out["required"] = required
		}
		return out, nil

	case reflect.String:
		return map[string]any{"type": "string"}, nil
	case reflect.Bool:
		return map[string]any{"type": "boolean"}, nil

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}, nil

	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}, nil

	case reflect.Slice, reflect.Array:
		items, err := schemaForType(t.Elem(), visited)
		if err != nil {
			return nil, err
		}
		return map[string]any{
			"type":  "array",
			"items": items,
		}, nil

	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			return nil, fmt.Errorf("only map[string]T supported, got map[%s]%s",
				t.Key().Kind(), t.Elem().Kind())
		}
		ap, err := schemaForType(t.Elem(), visited)
		if err != nil {
			return nil, err
		}
		return map[string]any{
			"type":                 "object",
			"additionalProperties": ap,
		}, nil

	default:
		return nil, fmt.Errorf("unsupported kind: %s", t.Kind())
	}
}

func parseJSONTag(f reflect.StructField) (name string, omitempty bool, skip bool) {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return "", false, true
	}
	if tag == "" {
		return f.Name, false, false
	}

	parts := strings.Split(tag, ",")
	if parts[0] == "" {
		name = f.Name
	} else {
		name = parts[0]
	}

	for _, p := range parts[1:] {
		if p == "omitempty" {
			omitempty = true
		}
	}

	return name, omitempty, false
}
