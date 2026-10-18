package openaidecisions_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"strings"
	"testing"
	"testing/iotest"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openaidecisions"
)

const (
	predicateAnswer = `{"type":"predicate","name":"p","probability":0.9}`
	choiceAnswer    = `{"type":"choice","name":"c","choice":"a","confidence":0.9,` +
		`"probabilities":[{"value":"a","probability":0.9},{"value":"b","probability":0.1}]}`
	scoreAnswer = `{"type":"score","name":"s","score":1.5,"confidence":0.5,"probabilities":[` +
		`{"value":0,"label":"0","probability":0},{"value":1,"label":"1","probability":0.5},` +
		`{"value":2,"label":"2","probability":0.5}]}`
)

func validationParams() ai.ModelRequestParams {
	return output(`{"type":"object","properties":{
		"p":{"type":"boolean","description":"Is it urgent?"},
		"c":{"type":"string","enum":["a","b"],"description":"Which one?"},
		"s":{"type":"integer","description":"How bad?","anyOf":[
			{"const":0,"description":"Fine"},{"const":1,"description":"Bad"},{"const":2,"description":"Broken"}]}
	},"required":["p","c","s"]}`)
}

func answers(items ...string) string {
	return `{"model":"gpt-6-luna","answers":[` + strings.Join(items, ",") + `],"usage":{}}`
}

func choice(probabilities string) string {
	return `{"type":"choice","name":"c","choice":"a","confidence":0.9,"probabilities":[` + probabilities + `]}`
}

func score(value, probabilities string) string {
	return `{"type":"score","name":"s","score":` + value + `,"confidence":0.5,"probabilities":[` + probabilities + `]}`
}

func TestResponseValidation(t *testing.T) {
	for _, test := range []struct {
		name, body, match string
	}{
		{"not JSON", `{"answers":`, "unexpected end of JSON input"},
		{"answers not a list", `{"answers":{}}`, "cannot unmarshal object"},
		{"missing name", answers(`{"type":"predicate","probability":0.9}`, choiceAnswer, scoreAnswer), "omitted name"},
		{"duplicate", answers(predicateAnswer, predicateAnswer, choiceAnswer), `duplicate answer "p"`},
		{"missing answer", answers(predicateAnswer, choiceAnswer), "do not match questions"},
		{"extra answer", answers(predicateAnswer, choiceAnswer, scoreAnswer, `{"type":"refusal","name":"x"}`), "do not match"},
		{"renamed answer", answers(predicateAnswer, choiceAnswer, strings.Replace(scoreAnswer, `"s"`, `"x"`, 1)), "do not match"},
		{"predicate type", answers(strings.Replace(predicateAnswer, "predicate", "score", 1), choiceAnswer, scoreAnswer), `"p": invalid predicate`},
		{"predicate missing", answers(`{"type":"predicate","name":"p"}`, choiceAnswer, scoreAnswer), "invalid predicate"},
		{"predicate range", answers(strings.Replace(predicateAnswer, "0.9", "1.5", 1), choiceAnswer, scoreAnswer), "invalid predicate"},
		{"choice type", answers(predicateAnswer, strings.Replace(choiceAnswer, `"type":"choice"`, `"type":"score"`, 1), scoreAnswer), `"c": invalid choice`},
		{"choice not text", answers(predicateAnswer, strings.Replace(choiceAnswer, `"choice":"a"`, `"choice":true`, 1), scoreAnswer), "invalid choice"},
		{"choice confidence missing", answers(predicateAnswer, strings.Replace(choiceAnswer, `"confidence":0.9,`, "", 1), scoreAnswer), "invalid choice"},
		{"choice confidence range", answers(predicateAnswer, strings.Replace(choiceAnswer, `"confidence":0.9`, `"confidence":-1`, 1), scoreAnswer), "invalid choice"},
		{"unknown choice", answers(predicateAnswer, strings.Replace(choiceAnswer, `"choice":"a"`, `"choice":"z"`, 1), scoreAnswer), `unknown choice "z"`},
		{"choice option not text", answers(predicateAnswer, choice(`{"value":true,"probability":0.9},{"value":"b","probability":0.1}`), scoreAnswer), "invalid choice probabilities"},
		{"choice option unknown", answers(predicateAnswer, choice(`{"value":"a","probability":0.9},{"value":"z","probability":0.1}`), scoreAnswer), "invalid choice probabilities"},
		{"choice option twice", answers(predicateAnswer, choice(`{"value":"a","probability":0},{"value":"a","probability":0.5},{"value":"b","probability":0.5}`), scoreAnswer), "invalid choice probabilities"},
		{"choice probability range", answers(predicateAnswer, choice(`{"value":"a","probability":1.1},{"value":"b","probability":-0.1}`), scoreAnswer), "invalid choice probabilities"},
		{"choice option missing", answers(predicateAnswer, choice(`{"value":"a","probability":1}`), scoreAnswer), "do not match options"},
		{"choice sum", answers(predicateAnswer, choice(`{"value":"a","probability":0.5},{"value":"b","probability":0.1}`), scoreAnswer), "do not match options"},
		{"choice all zero", answers(predicateAnswer, choice(`{"value":"a","probability":0},{"value":"b","probability":0}`), scoreAnswer), "do not match options"},
		{"score type", answers(predicateAnswer, choiceAnswer, strings.Replace(scoreAnswer, `"type":"score"`, `"type":"choice"`, 1)), `"s": invalid score`},
		{"score missing", answers(predicateAnswer, choiceAnswer, strings.Replace(scoreAnswer, `"score":1.5,`, "", 1)), "invalid score"},
		{"score above rubric", answers(predicateAnswer, choiceAnswer, score("2.5", "")), "invalid score"},
		{"score below rubric", answers(predicateAnswer, choiceAnswer, score("-0.5", "")), "invalid score"},
		{"score confidence", answers(predicateAnswer, choiceAnswer, strings.Replace(scoreAnswer, `"confidence":0.5`, `"confidence":2`, 1)), "invalid score"},
		{"level not whole", answers(predicateAnswer, choiceAnswer, score("1", `{"value":0.5,"probability":1}`)), "invalid score probabilities"},
		{"level as text", answers(predicateAnswer, choiceAnswer, score("1", `{"value":"1","probability":1}`)), "invalid score probabilities"},
		{"level out of rubric", answers(predicateAnswer, choiceAnswer, score("1", `{"value":3,"probability":1}`)), "invalid score probabilities"},
		{"level negative", answers(predicateAnswer, choiceAnswer, score("1", `{"value":-1,"probability":1}`)), "invalid score probabilities"},
		{"level twice", answers(predicateAnswer, choiceAnswer, score("1", `{"value":1,"probability":0.5},{"value":1,"probability":0.5}`)), "invalid score probabilities"},
		{"level probability", answers(predicateAnswer, choiceAnswer, score("1", `{"value":1,"probability":2}`)), "invalid score probabilities"},
		{"level missing", answers(predicateAnswer, choiceAnswer, score("1", `{"value":0,"probability":0},{"value":1,"probability":1}`)), "do not match score"},
		{"score sum", answers(predicateAnswer, choiceAnswer, score("0.5", `{"value":0,"probability":0.5},{"value":1,"probability":0.5},{"value":2,"probability":0.5}`)), "do not match score"},
		{"score mismatch", answers(predicateAnswer, choiceAnswer, score("1", `{"value":0,"probability":0},{"value":1,"probability":0.5},{"value":2,"probability":0.5}`)), "do not match score"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := newFake(t, test.body).model().Request(t.Context(), prompt("Down?"), validationParams())
			var unexpected *ai.UnexpectedModelBehaviorError
			if !errors.As(err, &unexpected) || !strings.Contains(err.Error(), "openai decisions: invalid response") ||
				!strings.Contains(err.Error(), test.match) {
				t.Fatalf("error=%v", err)
			}
		})
	}
}

func TestRoundedAnswers(t *testing.T) {
	body := answers(
		predicateAnswer,
		choice(`{"value":"a","probability":0.996},{"value":"b","probability":0.0}`),
		score("1", `{"value":0,"probability":0.33},{"value":1,"probability":0.33},{"value":2,"probability":0.33}`),
	)
	response, err := newFake(t, body).model().Request(t.Context(), prompt("Down?"), validationParams())
	if err != nil {
		t.Fatal(err)
	}
	assertJSON(t, response.ProviderDetails["probabilities"], `{
		"c":{"a":0.996,"b":0},
		"s":{"0":0.33,"1":0.33,"2":0.33}
	}`)
}

func TestRefusal(t *testing.T) {
	refusal := func(name string) string { return `{"type":"refusal","name":"` + name + `"}` }
	for _, test := range []struct {
		name, body, message string
		refused             []string
	}{
		{"one", answers(predicateAnswer, refusal("c"), scoreAnswer), "declined questions: c", []string{"c"}},
		{"every", answers(refusal("s"), refusal("p"), choiceAnswer), "declined questions: p, s", []string{"p", "s"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := newFake(t, test.body).model().Request(t.Context(), prompt("Down?"), validationParams())
			var filtered *ai.ContentFilterError
			if !errors.As(err, &filtered) || !strings.Contains(err.Error(), test.message) {
				t.Fatalf("error=%v", err)
			}
			response := filtered.Response()
			if response.FinishReason != ai.FinishReasonContentFilter || response.ModelName != "gpt-6-luna" ||
				response.ProviderName != "openai-decisions" || response.ProviderURL != "https://decisions.test/v1" ||
				response.ProviderResponseID != "req_1" {
				t.Fatalf("response=%+v", response)
			}
			assertJSON(t, response.ProviderDetails["refused_questions"], string(must(json.Marshal(test.refused))))
		})
	}
}

func TestStatusError(t *testing.T) {
	body := `{"error":{"message":"questions[0].choices: array too long, maximum length 255,"}}`
	model := openaidecisions.NewModel("gpt-6-luna", openaidecisions.WithHTTPClient(&http.Client{
		Transport: transport(func(*http.Request) (*http.Response, error) {
			return reply(http.StatusBadRequest, body, http.Header{"X-Request-Id": {"req_400"}}), nil
		}),
	}))
	_, err := model.Request(t.Context(), prompt("Down?"), urgentParams())
	var apiError *openaidecisions.APIError
	if !errors.As(err, &apiError) || apiError.StatusCode != http.StatusBadRequest || apiError.Body != body ||
		apiError.Headers.Get("X-Request-Id") != "req_400" || !apiError.IsModelAPIError() ||
		err.Error() != "OpenAI Decisions API returned status 400: "+body {
		t.Fatalf("error=%#v", err)
	}
}

func TestTransportErrors(t *testing.T) {
	failure := errors.New("connection reset")
	for _, test := range []struct {
		name      string
		roundTrip transport
		operation string
	}{
		{"request", func(*http.Request) (*http.Response, error) { return nil, failure }, "request"},
		{"read", func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(iotest.ErrReader(failure))}, nil
		}, "read response"},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := openaidecisions.NewModel("gpt-6-luna", openaidecisions.WithHTTPClient(&http.Client{Transport: test.roundTrip}))
			_, err := model.Request(t.Context(), prompt("Down?"), urgentParams())
			var transportError *ai.ModelTransportError
			if !errors.As(err, &transportError) || !errors.Is(err, failure) || transportError.Operation != test.operation ||
				transportError.ModelName != "gpt-6-luna" || transportError.ProviderName != "openai-decisions" {
				t.Fatalf("error=%#v", err)
			}
		})
	}

	ctx, cancel := context.WithCancel(t.Context())
	model := openaidecisions.NewModel("gpt-6-luna", openaidecisions.WithHTTPClient(&http.Client{
		Transport: transport(func(*http.Request) (*http.Response, error) {
			cancel()
			return nil, context.Canceled
		}),
	}))
	_, err := model.Request(ctx, prompt("Down?"), urgentParams())
	var transportError *ai.ModelTransportError
	if !errors.Is(err, context.Canceled) || errors.As(err, &transportError) {
		t.Fatalf("error=%#v", err)
	}
}

func TestRequestBuildErrors(t *testing.T) {
	fake := newFake(t)
	_, err := fake.model(openaidecisions.WithBaseURL("http://bad host/v1")).Request(t.Context(), prompt("Down?"), urgentParams())
	if err == nil || !strings.Contains(err.Error(), "openai decisions: parse") {
		t.Fatalf("error=%v", err)
	}
	params := urgentParams()
	params.Settings.ExtraBody = map[string]any{"bad": math.NaN()}
	_, err = fake.model().Request(t.Context(), prompt("Down?"), params)
	if err == nil || !strings.Contains(err.Error(), "openai decisions: encode request") {
		t.Fatalf("error=%v", err)
	}
	if len(fake.exchanges) != 0 {
		t.Fatal("invalid request reached the transport")
	}
}
