package core

type ToolUnion interface {
	isToolUnion()
}

type ToolCall struct {
	ID        string
	Name      string
	Arguments any
}

type ServerTool struct {
	Name        string
	Description string
	Parameters  map[string]any
	Handler     func(fn any) (string, error)
}

func (ServerTool) isToolUnion() {}

type ClientTool struct {
	Name        string
	Description string
	Parameters  map[string]any
}

func (ClientTool) isToolUnion() {}
