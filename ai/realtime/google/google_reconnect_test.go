package google_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	googlert "github.com/Kludex/pydantic-ai-go/ai/realtime/google"
	"google.golang.org/genai"
)

type connectorResult struct {
	session googlert.LiveSession
	err     error
}

type sequenceConnector struct {
	mutex   sync.Mutex
	results []connectorResult
	configs []*genai.LiveConnectConfig
}

func (connector *sequenceConnector) Connect(
	_ context.Context, _ string, config *genai.LiveConnectConfig,
) (googlert.LiveSession, error) {
	connector.mutex.Lock()
	defer connector.mutex.Unlock()
	copyConfig := *config
	connector.configs = append(connector.configs, &copyConfig)
	if len(connector.results) == 0 {
		return nil, errors.New("no session")
	}
	result := connector.results[0]
	connector.results = connector.results[1:]
	return result.session, result.err
}

func TestGoogleReconnectRestoresHandle(t *testing.T) {
	first := newFakeSession()
	second := newFakeSession()
	connector := &sequenceConnector{results: []connectorResult{{session: first}, {session: second}}}
	model := googlert.NewModel("model", googlert.WithConnector(connector))
	connection, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{
		Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 1, MaxDelay: time.Millisecond},
	}})
	if err != nil {
		t.Fatal(err)
	}
	first.receive <- &genai.LiveServerMessage{SessionResumptionUpdate: &genai.LiveServerSessionResumptionUpdate{
		NewHandle: "handle", Resumable: true,
	}}
	first.receive <- &genai.LiveServerMessage{GoAway: &genai.LiveServerGoAway{}}
	second.receive <- &genai.LiveServerMessage{ServerContent: &genai.LiveServerContent{TurnComplete: true}}
	var reconnected, completed bool
	for event, err := range connection.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		switch event := event.(type) {
		case realtime.SessionReconnected:
			reconnected = event.StateRestored
		case realtime.ResponseDone:
			completed = true
		}
		if completed {
			break
		}
	}
	if !reconnected || len(connector.configs) != 2 || connector.configs[1].SessionResumption.Handle != "handle" {
		t.Fatalf("reconnect did not restore handle: reconnected=%v configs=%+v", reconnected, connector.configs)
	}
	_ = connection.Close(t.Context())
}

func TestGoogleReconnectConsumerStops(t *testing.T) {
	first := newFakeSession()
	second := newFakeSession()
	connector := &sequenceConnector{results: []connectorResult{{session: first}, {session: second}}}
	connection, err := googlert.NewModel("model", googlert.WithConnector(connector)).Connect(
		t.Context(), realtime.ConnectParams{Settings: realtime.Settings{
			Reconnect: &realtime.ReconnectPolicy{
				MaxAttempts: 1, MaxReconnects: 1, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond, Jitter: true,
			},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	first.once.Do(func() {
		close(first.closed)
		close(first.receive)
	})
	for range connection.Events(t.Context()) {
		break
	}
	_ = connection.Close(t.Context())
}

func TestGoogleGoAwayConsumerStops(t *testing.T) {
	first := newFakeSession()
	second := newFakeSession()
	connector := &sequenceConnector{results: []connectorResult{{session: first}, {session: second}}}
	connection, err := googlert.NewModel("model", googlert.WithConnector(connector)).Connect(
		t.Context(), realtime.ConnectParams{Settings: realtime.Settings{Reconnect: &realtime.ReconnectPolicy{
			MaxAttempts: 1, MaxReconnects: 1, MaxDelay: time.Millisecond,
		}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	first.receive <- &genai.LiveServerMessage{GoAway: &genai.LiveServerGoAway{}}
	for range connection.Events(t.Context()) {
		break
	}
	_ = connection.Close(t.Context())
}

func TestGoogleGoAwayReconnectFailure(t *testing.T) {
	first := newFakeSession()
	connector := &sequenceConnector{results: []connectorResult{{session: first}, {err: errors.New("dial")}}}
	connection, err := googlert.NewModel("model", googlert.WithConnector(connector)).Connect(
		t.Context(), realtime.ConnectParams{Settings: realtime.Settings{Reconnect: &realtime.ReconnectPolicy{
			MaxAttempts: 1, MaxReconnects: 1, MaxDelay: time.Millisecond,
		}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	first.receive <- &genai.LiveServerMessage{GoAway: &genai.LiveServerGoAway{}}
	found := false
	for _, err := range connection.Events(t.Context()) {
		found = err != nil
	}
	if !found {
		t.Fatal("expected GoAway reconnect failure")
	}
	_ = connection.Close(t.Context())
}

func TestGoogleReconnectFailures(t *testing.T) {
	for name, results := range map[string][]connectorResult{
		"dial": {{session: newFakeSession()}, {err: errors.New("dial")}, {err: errors.New("dial again")}},
		"nil":  {{session: newFakeSession()}, {}},
	} {
		t.Run(name, func(t *testing.T) {
			first := results[0].session.(*fakeSession)
			connector := &sequenceConnector{results: results}
			connection, err := googlert.NewModel("model", googlert.WithConnector(connector)).Connect(
				t.Context(), realtime.ConnectParams{Settings: realtime.Settings{Reconnect: &realtime.ReconnectPolicy{
					MaxAttempts: 2, MaxReconnects: 1, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond,
				}}},
			)
			if err != nil {
				t.Fatal(err)
			}
			first.once.Do(func() {
				close(first.closed)
				close(first.receive)
			})
			found := false
			for _, err := range connection.Events(t.Context()) {
				found = err != nil
			}
			if !found {
				t.Fatal("expected reconnect failure")
			}
			_ = connection.Close(t.Context())
		})
	}
}

func TestGoogleReconnectLimitAndCancellation(t *testing.T) {
	first := newFakeSession()
	second := newFakeSession()
	connector := &sequenceConnector{results: []connectorResult{{session: first}, {session: second}}}
	connection, err := googlert.NewModel("model", googlert.WithConnector(connector)).Connect(
		t.Context(), realtime.ConnectParams{Settings: realtime.Settings{
			Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 1, MaxDelay: time.Millisecond},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	first.once.Do(func() { close(first.closed); close(first.receive) })
	second.once.Do(func() { close(second.closed); close(second.receive) })
	found := false
	for _, err := range connection.Events(t.Context()) {
		if err != nil {
			found = true
		}
	}
	if !found {
		t.Fatal("expected reconnect limit error")
	}
	_ = connection.Close(t.Context())

	first = newFakeSession()
	connector = &sequenceConnector{results: []connectorResult{{session: first}, {err: errors.New("dial")}}}
	connection, err = googlert.NewModel("model", googlert.WithConnector(connector)).Connect(
		t.Context(), realtime.ConnectParams{Settings: realtime.Settings{Reconnect: &realtime.ReconnectPolicy{
			MaxAttempts: 2, MaxReconnects: 1, BaseDelay: time.Second, MaxDelay: time.Millisecond,
		}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	first.once.Do(func() { close(first.closed); close(first.receive) })
	ctx, cancel := context.WithTimeout(t.Context(), time.Millisecond)
	defer cancel()
	for range connection.Events(ctx) {
	}
	_ = connection.Close(t.Context())
}

func TestGoogleDefaultReconnectAndClosedSend(t *testing.T) {
	live := newFakeSession()
	connection, err := googlert.NewModel("model", googlert.WithConnector(&fakeConnector{session: live})).Connect(
		t.Context(), realtime.ConnectParams{Settings: realtime.Settings{Reconnect: &realtime.ReconnectPolicy{}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := connection.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := connection.Send(t.Context(), realtime.TextInput{Text: "closed"}); err == nil {
		t.Fatal("expected closed send error")
	}
}
