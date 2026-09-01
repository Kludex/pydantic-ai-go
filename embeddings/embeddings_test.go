package embeddings

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
)

type testModel struct {
	result   *Result
	err      error
	inputs   []string
	kind     InputType
	settings Settings
}

func (model *testModel) Embed(
	_ context.Context, inputs []string, inputType InputType, settings Settings,
) (*Result, error) {
	model.inputs = inputs
	model.kind = inputType
	model.settings = settings
	return model.result, model.err
}

func (*testModel) Name() string         { return "test" }
func (*testModel) ProviderName() string { return "test" }
func (*testModel) ProviderURL() string  { return "https://example.com" }

type capableModel struct{ *testModel }

func (*capableModel) MaxInputTokens(context.Context) (int, bool, error) { return 42, true, nil }
func (*capableModel) CountTokens(context.Context, string) (int, error)  { return 7, nil }

type cyclicModel struct{ Model }

func (model *cyclicModel) UnwrapModel() Model { return model }

type valueWrapper struct{ Model }

func (wrapper valueWrapper) UnwrapModel() Model { return wrapper.Model }

type nilWrapper struct{ Model }

func (nilWrapper) UnwrapModel() Model { return (*testModel)(nil) }

type staticEmbeddingModel float64

func (model staticEmbeddingModel) Embed(
	_ context.Context, inputs []string, inputType InputType, _ Settings,
) (*Result, error) {
	return &Result{Embeddings: [][]float64{{float64(model)}}, Inputs: inputs, InputType: inputType}, nil
}
func (staticEmbeddingModel) Name() string         { return "static" }
func (staticEmbeddingModel) ProviderName() string { return "test" }
func (staticEmbeddingModel) ProviderURL() string  { return "" }

func TestEmbedderOperationsAndDetachment(t *testing.T) {
	dimensions := 2
	baseHeader := map[string]string{"base": "one"}
	baseBody := map[string]any{"nested": map[string]any{"value": "base"}}
	model := &testModel{result: &Result{
		Embeddings: [][]float64{{1, 2}, {3, 4}}, Inputs: []string{"wrong", "wrong"},
		InputType: InputTypeDocument, ProviderDetails: map[string]any{"items": []any{"detail"}},
		Usage: ai.Usage{Details: map[string]int{"tokens": 2}},
	}}
	embedder := New(model, WithSettings(Settings{
		Dimensions: &dimensions, ExtraHeaders: baseHeader, ExtraBody: baseBody,
	}))
	baseHeader["base"] = "mutated"
	baseBody["nested"].(map[string]any)["value"] = "mutated"
	override := 3
	result, err := embedder.EmbedQueries(context.Background(), []string{"one", "two"}, Settings{Dimensions: &override})
	if err != nil {
		t.Fatal(err)
	}
	if embedder.Model() != model || model.kind != InputTypeQuery || !reflect.DeepEqual(model.inputs, []string{"one", "two"}) {
		t.Fatalf("unexpected request: %#v %#v", model.inputs, model.kind)
	}
	if *model.settings.Dimensions != 3 || model.settings.ExtraHeaders["base"] != "one" ||
		model.settings.ExtraBody["nested"].(map[string]any)["value"] != "base" {
		t.Fatalf("unexpected merged settings: %#v", model.settings)
	}
	model.inputs[0] = "mutated"
	model.settings.ExtraBody["nested"].(map[string]any)["value"] = "request-mutated"
	model.result.Embeddings[0][0] = 99
	model.result.ProviderDetails["items"].([]any)[0] = "mutated"
	model.result.Usage.Details["tokens"] = 99
	if result.Embeddings[0][0] != 1 || result.ProviderDetails["items"].([]any)[0] != "detail" ||
		result.Usage.Details["tokens"] != 2 {
		t.Fatalf("result was not detached: %#v", result)
	}
	vector, ok := result.At(0)
	if !ok || !reflect.DeepEqual(vector, []float64{1, 2}) {
		t.Fatalf("unexpected vector: %#v %v", vector, ok)
	}
	vector[0] = 88
	if result.Embeddings[0][0] != 1 {
		t.Fatal("At exposed result storage")
	}
	if _, ok := result.At(-1); ok {
		t.Fatal("negative index unexpectedly succeeded")
	}
	if _, ok := result.At(2); ok {
		t.Fatal("out-of-range index unexpectedly succeeded")
	}
	result.Inputs = []string{"one", "two"}
	vector, ok = result.ForInput("two")
	if !ok || !reflect.DeepEqual(vector, []float64{3, 4}) {
		t.Fatalf("unexpected input vector: %#v %v", vector, ok)
	}
	vector[0] = 77
	if result.Embeddings[1][0] != 3 {
		t.Fatal("ForInput exposed result storage")
	}
	if _, ok := result.ForInput("missing"); ok {
		t.Fatal("missing input unexpectedly succeeded")
	}
	result.Inputs = append(result.Inputs, "orphan")
	if _, ok := result.ForInput("orphan"); ok {
		t.Fatal("orphan input unexpectedly succeeded")
	}
}

func TestEmbedderConvenienceMethods(t *testing.T) {
	model := &testModel{result: &Result{Embeddings: [][]float64{{1}}}}
	embedder := New(model)
	cases := []struct {
		call func() (*Result, error)
		kind InputType
		text string
	}{
		{call: func() (*Result, error) {
			return embedder.EmbedQuery(context.Background(), "query")
		}, kind: InputTypeQuery, text: "query"},
		{call: func() (*Result, error) {
			return embedder.EmbedDocument(context.Background(), "document", Settings{})
		}, kind: InputTypeDocument, text: "document"},
		{call: func() (*Result, error) {
			return embedder.EmbedDocuments(context.Background(), []string{"documents"}, Settings{})
		}, kind: InputTypeDocument, text: "documents"},
	}
	for _, test := range cases {
		if _, err := test.call(); err != nil {
			t.Fatal(err)
		}
		if model.kind != test.kind || !reflect.DeepEqual(model.inputs, []string{test.text}) {
			t.Fatalf("unexpected convenience request: %#v %#v", model.kind, model.inputs)
		}
	}
}

func TestEmbedderErrorsAndCapabilities(t *testing.T) {
	if panicValue := capturePanic(func() { New(nil) }); panicValue != "embeddings: model must not be nil" {
		t.Fatalf("unexpected panic: %v", panicValue)
	}
	if panicValue := capturePanic(func() { WrapModel(nil) }); panicValue != "embeddings: cannot wrap a nil model" {
		t.Fatalf("unexpected panic: %v", panicValue)
	}
	modelErr := errors.New("model failed")
	model := &testModel{err: modelErr}
	embedder := New(model)
	if _, err := embedder.Embed(context.Background(), nil, InputTypeQuery, Settings{}); err == nil ||
		!strings.Contains(err.Error(), "must not be empty") {
		t.Fatalf("unexpected empty error: %v", err)
	}
	if _, err := embedder.Embed(context.Background(), []string{"x"}, InputType("bad"), Settings{}); err == nil ||
		!strings.Contains(err.Error(), `invalid input type "bad"`) {
		t.Fatalf("unexpected type error: %v", err)
	}
	if _, err := embedder.Embed(
		context.Background(), []string{"x"}, InputTypeQuery, Settings{}, Settings{},
	); err == nil || !strings.Contains(err.Error(), "at most one") {
		t.Fatalf("unexpected settings count error: %v", err)
	}
	zero := 0
	if _, err := embedder.Embed(
		context.Background(), []string{"x"}, InputTypeQuery, Settings{Dimensions: &zero},
	); err == nil || !strings.Contains(err.Error(), "dimensions must be greater than zero") {
		t.Fatalf("unexpected dimensions error: %v", err)
	}
	if _, err := embedder.EmbedQuery(context.Background(), "x", Settings{}); !errors.Is(err, modelErr) {
		t.Fatalf("unexpected model error: %v", err)
	}
	model.err = nil
	if _, err := embedder.EmbedQuery(context.Background(), "x", Settings{}); err == nil ||
		!strings.Contains(err.Error(), "nil result") {
		t.Fatalf("unexpected nil result error: %v", err)
	}
	model.result = &Result{Embeddings: [][]float64{{1}, {2}}}
	if _, err := embedder.EmbedQuery(context.Background(), "x", Settings{}); err == nil ||
		!strings.Contains(err.Error(), "2 vectors for 1 inputs") {
		t.Fatalf("unexpected vector error: %v", err)
	}
	if maximum, known, err := embedder.MaxInputTokens(context.Background()); err != nil || known || maximum != 0 {
		t.Fatalf("unexpected absent maximum: %d %v %v", maximum, known, err)
	}
	if _, err := embedder.CountTokens(context.Background(), "x"); !errors.Is(err, ErrTokenCountingUnsupported) {
		t.Fatalf("unexpected unsupported error: %v", err)
	}
	capable := New(&capableModel{testModel: &testModel{}})
	if maximum, known, err := capable.MaxInputTokens(context.Background()); err != nil || !known || maximum != 42 {
		t.Fatalf("unexpected maximum: %d %v %v", maximum, known, err)
	}
	if count, err := capable.CountTokens(context.Background(), "x"); err != nil || count != 7 {
		t.Fatalf("unexpected count: %d %v", count, err)
	}
	wrapped := WrapModel(model)
	if wrapped.Name() != "test" || wrapped.ProviderName() != "test" || wrapped.ProviderURL() != "https://example.com" {
		t.Fatalf("wrapper did not delegate identity: %#v", wrapped)
	}
}

func TestSettingsCloneAndMerge(t *testing.T) {
	dimensions := 4
	truncate := true
	cycle := map[string]any{}
	cycle["self"] = cycle
	cyclicSlice := []any{nil}
	cyclicSlice[0] = cyclicSlice
	array := [1][]string{{"array"}}
	settings := Settings{
		Dimensions: &dimensions, Truncate: &truncate, ExtraHeaders: map[string]string{"x": "one"},
		ExtraBody: map[string]any{
			"map": map[string][]string{"values": {"one"}}, "slice": []any{[]string{"two"}},
			"array": array, "nil_map": map[string]any(nil), "nil_slice": []string(nil), "nil_interface": nil,
			"nested_nil": map[string]any{"nil": nil}, "cycle": cycle, "slice_cycle": cyclicSlice,
			"scalar": "value",
		},
	}
	cloned := settings.Clone()
	*settings.Dimensions = 9
	*settings.Truncate = false
	settings.ExtraHeaders["x"] = "changed"
	settings.ExtraBody["map"].(map[string][]string)["values"][0] = "changed"
	settings.ExtraBody["slice"].([]any)[0].([]string)[0] = "changed"
	settings.ExtraBody["array"].([1][]string)[0][0] = "changed"
	if *cloned.Dimensions != 4 || !*cloned.Truncate || cloned.ExtraHeaders["x"] != "one" ||
		cloned.ExtraBody["map"].(map[string][]string)["values"][0] != "one" ||
		cloned.ExtraBody["slice"].([]any)[0].([]string)[0] != "two" ||
		cloned.ExtraBody["array"].([1][]string)[0][0] != "array" {
		t.Fatalf("settings were not cloned: %#v", cloned)
	}
	clonedCycle := cloned.ExtraBody["cycle"].(map[string]any)
	if reflect.ValueOf(clonedCycle).Pointer() != reflect.ValueOf(clonedCycle["self"]).Pointer() {
		t.Fatal("map cycle was not retained")
	}
	newDimensions := 6
	newTruncate := false
	merged := MergeSettings(cloned, Settings{
		Dimensions: &newDimensions, Truncate: &newTruncate, ExtraHeaders: map[string]string{}, ExtraBody: map[string]any{},
	})
	if *merged.Dimensions != 6 || *merged.Truncate || len(merged.ExtraHeaders) != 0 || len(merged.ExtraBody) != 0 {
		t.Fatalf("unexpected merge: %#v", merged)
	}
	clonedSlice := cloned.ExtraBody["slice_cycle"].([]any)
	if reflect.ValueOf(clonedSlice).Pointer() != reflect.ValueOf(clonedSlice[0]).Pointer() {
		t.Fatal("slice cycle was not retained")
	}
	emptySettings := (Settings{}).Clone()
	emptyResult := (Result{}).Clone()
	if emptySettings.ExtraBody != nil || emptyResult.Embeddings != nil {
		t.Fatal("nil clones should remain nil")
	}
}

func TestContextModelOverride(t *testing.T) {
	base := &capableModel{testModel: &testModel{result: &Result{Embeddings: [][]float64{{1}}}}}
	override := &capableModel{testModel: &testModel{result: &Result{Embeddings: [][]float64{{2}}}}}
	embedder := New(base)
	ctx := WithModel(context.Background(), override)
	result, err := embedder.EmbedQuery(ctx, "text")
	if err != nil || result.Embeddings[0][0] != 2 || embedder.Model() != base {
		t.Fatalf("unexpected override result: %#v %v", result, err)
	}
	if count, err := embedder.CountTokens(ctx, "text"); err != nil || count != 7 {
		t.Fatalf("unexpected override token count: %d %v", count, err)
	}
	if maximum, known, err := embedder.MaxInputTokens(ctx); err != nil || !known || maximum != 42 {
		t.Fatalf("unexpected override limit: %d %v %v", maximum, known, err)
	}
	if result, err := embedder.EmbedQuery(context.Background(), "text"); err != nil || result.Embeddings[0][0] != 1 {
		t.Fatalf("base model was not restored: %#v %v", result, err)
	}

	for _, function := range []func(){
		func() { WithModel(context.Background(), nil) },
	} {
		if panicValue := capturePanic(function); panicValue == nil {
			t.Fatal("invalid override did not panic")
		}
	}
	concurrent := New(staticEmbeddingModel(1))
	var group sync.WaitGroup
	failures := make(chan float64, 40)
	for index := range 40 {
		index := index
		group.Go(func() {
			selected := staticEmbeddingModel(float64(index + 2))
			result, err := concurrent.EmbedQuery(WithModel(context.Background(), selected), "text")
			if err != nil {
				failures <- 0
				return
			}
			failures <- result.Embeddings[0][0]
		})
	}
	group.Wait()
	close(failures)
	seen := map[float64]bool{}
	for value := range failures {
		seen[value] = true
	}
	if len(seen) != 40 {
		t.Fatalf("concurrent overrides crossed requests: %#v", seen)
	}
}

func TestWrapperOptionalCapabilities(t *testing.T) {
	base := &capableModel{testModel: &testModel{}}
	embedder := New(WrapModel(WrapModel(base)))
	if maximum, known, err := embedder.MaxInputTokens(context.Background()); err != nil || !known || maximum != 42 {
		t.Fatalf("unexpected wrapped limit: %d %v %v", maximum, known, err)
	}
	if count, err := embedder.CountTokens(context.Background(), "text"); err != nil || count != 7 {
		t.Fatalf("unexpected wrapped token count: %d %v", count, err)
	}

	cycle := &cyclicModel{Model: &testModel{}}
	if _, err := New(cycle).CountTokens(context.Background(), "text"); !errors.Is(err, ErrTokenCountingUnsupported) {
		t.Fatalf("unexpected cycle error: %v", err)
	}
	if maximum, known, err := New(valueWrapper{Model: &testModel{}}).MaxInputTokens(context.Background()); err != nil || known || maximum != 0 {
		t.Fatalf("unexpected value-wrapper limit: %d %v %v", maximum, known, err)
	}
	if _, err := New(nilWrapper{Model: &testModel{}}).CountTokens(context.Background(), "text"); !errors.Is(err, ErrTokenCountingUnsupported) {
		t.Fatalf("unexpected nil-wrapper error: %v", err)
	}
}

func TestResultPriceAndClone(t *testing.T) {
	result := Result{
		Embeddings: [][]float64{{1}}, Inputs: []string{"one"}, InputType: InputTypeQuery,
		ModelName: "text-embedding-3-small", ProviderName: "openai",
		Timestamp: time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC), Usage: ai.Usage{InputTokens: 1},
		Warnings: []string{"warning"},
	}
	calculation, err := result.Price()
	if err != nil || calculation.TotalPrice <= 0 {
		t.Fatalf("unexpected price: %#v %v", calculation, err)
	}
	cloned := result.Clone()
	cloned.Embeddings[0][0] = 2
	cloned.Inputs[0] = "two"
	cloned.Warnings[0] = "changed"
	if result.Embeddings[0][0] != 1 || result.Inputs[0] != "one" || result.Warnings[0] != "warning" {
		t.Fatal("clone changed source result")
	}
}

func capturePanic(fn func()) (value any) {
	defer func() { value = recover() }()
	fn()
	return nil
}
