package openai

type transcriptionResponse struct {
	Text     string                 `json:"text"`
	Language string                 `json:"language,omitempty"`
	Duration float64                `json:"duration,omitempty"`
	Segments []transcriptionSegment `json:"segments,omitempty"`
	Words    []transcriptionWord    `json:"words,omitempty"`
}

type transcriptionSegment struct {
	Start float64             `json:"start"`
	End   float64             `json:"end"`
	Text  string              `json:"text"`
	Words []transcriptionWord `json:"words,omitempty"`
}

type transcriptionWord struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}
