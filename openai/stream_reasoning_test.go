package openai

import "testing"

func TestAppendStreamSegmentHandlesCumulativeSnapshots(t *testing.T) {
	t.Parallel()

	current := ""
	next, delta := appendStreamSegment(current, "The")
	if next != "The" || delta != "The" {
		t.Fatalf("unexpected first segment: next=%q delta=%q", next, delta)
	}

	next, delta = appendStreamSegment(next, "The user")
	if next != "The user" || delta != " user" {
		t.Fatalf("unexpected second segment: next=%q delta=%q", next, delta)
	}

	next, delta = appendStreamSegment(next, "The user asks")
	if next != "The user asks" || delta != " asks" {
		t.Fatalf("unexpected third segment: next=%q delta=%q", next, delta)
	}
}

func TestAppendStreamSegmentHandlesDeltaUpdates(t *testing.T) {
	t.Parallel()

	current := "The user"
	next, delta := appendStreamSegment(current, " asks")
	if next != "The user asks" || delta != " asks" {
		t.Fatalf("unexpected segment append: next=%q delta=%q", next, delta)
	}
}

func TestParseStreamDeltaReasoningPreservesWhitespace(t *testing.T) {
	t.Parallel()

	got := parseStreamDeltaReasoning(streamDelta{ReasoningContent: " asks"})
	if got != " asks" {
		t.Fatalf("expected leading space to be preserved, got %q", got)
	}
}
