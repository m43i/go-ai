package core

// TranscriptionParams configures an audio transcription request.
type TranscriptionParams struct {
	// Audio is the audio data to transcribe. Required.
	// Provide raw bytes of the audio file (e.g., MP3, WAV, FLAC).
	Audio []byte

	// Filename is the original filename including extension (e.g., "recording.mp3").
	// Used to communicate the audio format to the provider.
	// Required.
	Filename string

	// Language is a BCP-47 / ISO-639-1 language code hint (e.g., "en", "es", "fr").
	// Optional; improves accuracy and latency when specified.
	Language string

	// ModelOptions holds provider-specific options that are passed through
	// directly to the API (e.g., response_format, temperature, prompt).
	// Keys that conflict with top-level fields are rejected.
	ModelOptions map[string]any
}

// TranscriptionResult holds the output of an audio transcription.
type TranscriptionResult struct {
	// Text is the full transcribed text.
	Text string

	// Language is the detected or specified language code.
	Language string

	// Duration is the audio duration in seconds.
	Duration float64

	// Segments contains timestamped segments when verbose output is requested.
	Segments []TranscriptionSegment
}

// TranscriptionSegment is a timestamped portion of a transcription.
type TranscriptionSegment struct {
	// Start is the segment start time in seconds.
	Start float64

	// End is the segment end time in seconds.
	End float64

	// Text is the transcribed text for this segment.
	Text string

	// Words contains word-level timestamps when requested.
	Words []TranscriptionWord
}

// TranscriptionWord is a single word with timing and confidence data.
type TranscriptionWord struct {
	Word  string
	Start float64
	End   float64
}
