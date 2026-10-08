package ai_test

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type requestedCacheModel struct{ *ai.ModelWrapper }

func (*requestedCacheModel) CacheRetention(settings ai.ModelSettings) (time.Duration, bool) {
	retention, known := settings.ExtraBody["retention"].(time.Duration)
	return retention, known
}

func TestCacheHealthRetentionWindows(t *testing.T) {
	for _, test := range []struct {
		name      string
		profile   time.Duration
		requested time.Duration
		point     bool
		reason    string
	}{
		{name: "default", profile: 5 * time.Minute, reason: "ttl_expired"},
		{name: "extended setting", profile: 5 * time.Minute, requested: time.Hour, reason: "unexpected"},
		{name: "shorter setting", profile: time.Hour, requested: 5 * time.Minute, reason: "ttl_expired"},
		{name: "setting without profile", requested: time.Hour, reason: "unexpected"},
		{name: "cache point", profile: 5 * time.Minute, point: true, reason: "unexpected"},
		{name: "unknown with point", point: true, reason: "unknown"},
	} {
		t.Run(test.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				exporter, provider := cacheExporter(t)
				base := ai.NewProfiledModel(cacheModel(cacheResponse(0, 14000), cacheResponse(1000, 0)), ai.ModelProfile{
					DefaultOutputMode: ai.OutputModeTool, DefaultCacheRetention: test.profile,
				})
				model := &requestedCacheModel{ai.WrapModel(base)}
				agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider))))
				contents := []ai.UserContent{ai.TextContent{Text: "turn"}}
				if test.point {
					contents = append(contents, ai.CachePoint{TTL: ai.CachePointTTL1Hour})
				}
				first, err := agent.RunParts(t.Context(), contents, struct{}{})
				if err != nil {
					t.Fatal(err)
				}
				time.Sleep(30 * time.Minute)
				settings := ai.ModelSettings{}
				if test.requested != 0 {
					settings.ExtraBody = map[string]any{"retention": test.requested}
				}
				if _, err := agent.Run(t.Context(), "second", struct{}{}, ai.WithConversation(first.Conversation()), ai.WithRunModelSettings(settings)); err != nil {
					t.Fatal(err)
				}
				if got := cacheAttributes(cacheChatSpans(exporter)[1])["pydantic_ai.cache.collapse_reason"]; got != test.reason {
					t.Fatalf("retention reason = %v, want %s", got, test.reason)
				}
			})
		})
	}
}

func TestCacheHealthUnreportedDoesNotRefreshRetention(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		exporter, provider := cacheExporter(t)
		model := ai.NewProfiledModel(cacheModel(cacheResponse(0, 14000), cacheResponse(0, 0), cacheResponse(1000, 0), cacheResponse(0, 1000)), ai.ModelProfile{
			DefaultOutputMode: ai.OutputModeTool, DefaultCacheRetention: 5 * time.Minute,
		})
		agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider))))
		var history []ai.ModelMessage
		for index := range 4 {
			switch index {
			case 1:
				time.Sleep(4 * time.Minute)
			case 2:
				time.Sleep(2 * time.Minute)
			}
			result, err := agent.Run(t.Context(), "turn", struct{}{}, ai.WithMessageHistory(history))
			if err != nil {
				t.Fatal(err)
			}
			history = result.Messages()
		}
		spans := cacheChatSpans(exporter)
		if cacheAttributes(spans[1])["pydantic_ai.cache.collapse_reason"] != "unreported" ||
			cacheAttributes(spans[2])["pydantic_ai.cache.collapse_reason"] != "ttl_expired" {
			t.Fatalf("unreported request refreshed the idle clock: %+v", spans)
		}
	})
}

func fillCacheConversations(t *testing.T, model ai.Model, count int) {
	t.Helper()
	for index := range count {
		_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{
			ConversationID: fmt.Sprintf("%s/%d", t.Name(), index),
		}}, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestCacheHealthBoundedRetention(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		exporter, provider := cacheExporter(t)
		writer := ai.NewInstrumentedModel(fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
			return cacheResponse(0, 14000), nil
		}), ai.WithInstrumentationTracerProvider(provider))
		reader := ai.NewInstrumentedModel(fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
			return cacheResponse(1000, 0), nil
		}), ai.WithInstrumentationTracerProvider(provider))
		old := []ai.ModelMessage{ai.ModelRequest{ConversationID: t.Name() + "/old"}}
		if _, err := writer.Request(t.Context(), old, ai.ModelRequestParams{}); err != nil {
			t.Fatal(err)
		}
		fillCacheConversations(t, writer, 4095)
		if _, err := writer.Request(t.Context(), old, ai.ModelRequestParams{}); err != nil {
			t.Fatal(err)
		}
		if _, err := writer.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{ConversationID: t.Name() + "/new"}}, ai.ModelRequestParams{}); err != nil {
			t.Fatal(err)
		}
		if _, err := reader.Request(t.Context(), old, ai.ModelRequestParams{}); err != nil {
			t.Fatal(err)
		}
		spans := cacheChatSpans(exporter)
		if cacheAttributes(spans[len(spans)-1])["pydantic_ai.cache.collapsed"] != true {
			t.Fatal("recently refreshed conversation was evicted")
		}
		forgotten := []ai.ModelMessage{ai.ModelRequest{ConversationID: t.Name() + "/0"}}
		if _, err := reader.Request(t.Context(), forgotten, ai.ModelRequestParams{}); err != nil {
			t.Fatal(err)
		}
		spans = cacheChatSpans(exporter)
		if cacheAttributes(spans[len(spans)-1])["pydantic_ai.cache.collapsed"] != nil {
			t.Fatal("least recently updated conversation was retained")
		}
		time.Sleep(25 * time.Hour)
		if _, err := reader.Request(t.Context(), old, ai.ModelRequestParams{}); err != nil {
			t.Fatal(err)
		}
		spans = cacheChatSpans(exporter)
		if cacheAttributes(spans[len(spans)-1])["pydantic_ai.cache.collapsed"] != nil {
			t.Fatal("conversation survived beyond the 24-hour horizon")
		}
	})
}

func TestCacheHealthConcurrentConversationMarks(t *testing.T) {
	exporter, provider := cacheExporter(t)
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	var group sync.WaitGroup
	for _, name := range []string{"first", "second"} {
		model := ai.NewInstrumentedModel(fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
			started <- struct{}{}
			<-release
			response := cacheResponse(0, 14000)
			response.ModelName = name
			return response, nil
		}), ai.WithInstrumentationTracerProvider(provider))
		group.Go(func() {
			if _, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{ConversationID: t.Name()}}, ai.ModelRequestParams{}); err != nil {
				t.Error(err)
			}
		})
	}
	<-started
	<-started
	close(release)
	group.Wait()
	for _, name := range []string{"first", "second"} {
		response := cacheResponse(1000, 0)
		response.ModelName = name
		model := ai.NewInstrumentedModel(cacheModel(response), ai.WithInstrumentationTracerProvider(provider))
		if _, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{ConversationID: t.Name()}}, ai.ModelRequestParams{}); err != nil {
			t.Fatal(err)
		}
	}
	for _, span := range cacheChatSpans(exporter)[2:] {
		if cacheAttributes(span)["pydantic_ai.cache.collapsed"] != true {
			t.Fatalf("concurrent run lost another model's mark: %+v", cacheAttributes(span))
		}
	}
}
