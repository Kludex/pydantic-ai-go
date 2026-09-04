package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/smithy-go/middleware"
	smithyhttp "github.com/aws/smithy-go/transport/http"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// LegacyBedrockClient is the AWS Bedrock Runtime surface used by the legacy
// Anthropic InvokeModel transport.
type LegacyBedrockClient interface {
	// InvokeModel generates one complete Anthropic response.
	InvokeModel(
		ctx context.Context, input *bedrockruntime.InvokeModelInput, options ...func(*bedrockruntime.Options),
	) (*bedrockruntime.InvokeModelOutput, error)
	// CountTokens counts an InvokeModel-formatted Anthropic request.
	CountTokens(
		ctx context.Context, input *bedrockruntime.CountTokensInput, options ...func(*bedrockruntime.Options),
	) (*bedrockruntime.CountTokensOutput, error)
}

// LegacyBedrockEventStream is one InvokeModelWithResponseStream result.
type LegacyBedrockEventStream interface {
	// Events returns provider payload chunks until the channel closes.
	Events() <-chan types.ResponseStream
	// Close releases the stream and must permit repeated calls.
	Close() error
	// Err returns the terminal stream-reader error.
	Err() error
}

// LegacyBedrockStreamingClient is the optional streaming AWS surface.
type LegacyBedrockStreamingClient interface {
	// InvokeModelWithResponseStream starts one Anthropic event stream.
	InvokeModelWithResponseStream(
		ctx context.Context, input *bedrockruntime.InvokeModelWithResponseStreamInput,
		options ...func(*bedrockruntime.Options),
	) (LegacyBedrockEventStream, error)
}

// LegacyBedrockConfig configures the Anthropic InvokeModel transport.
type LegacyBedrockConfig struct {
	// Client is caller-owned and must be safe for concurrent requests.
	Client LegacyBedrockClient
	// ProviderURL records the Bedrock Runtime endpoint for telemetry.
	ProviderURL string
}

// NewLegacyBedrockModel creates an Anthropic model using Bedrock's legacy
// InvokeModel API. The caller retains ownership of client.
func NewLegacyBedrockModel(name string, config LegacyBedrockConfig, options ...Option) *Model {
	if legacyBedrockClientIsNil(config.Client) {
		panic("anthropic: legacy Bedrock client must not be nil")
	}
	model := NewModel(name, options...)
	if sdkClient, ok := config.Client.(*bedrockruntime.Client); ok {
		model.legacyBedrockClient = &legacyAWSClient{client: sdkClient}
	} else {
		model.legacyBedrockClient = config.Client
	}
	model.baseURL = config.ProviderURL
	return model
}

type legacyAWSRuntimeClient interface {
	LegacyBedrockClient
	InvokeModelWithResponseStream(
		context.Context, *bedrockruntime.InvokeModelWithResponseStreamInput, ...func(*bedrockruntime.Options),
	) (*bedrockruntime.InvokeModelWithResponseStreamOutput, error)
}

type legacyAWSClient struct{ client legacyAWSRuntimeClient }

func (client *legacyAWSClient) InvokeModel(
	ctx context.Context, input *bedrockruntime.InvokeModelInput, options ...func(*bedrockruntime.Options),
) (*bedrockruntime.InvokeModelOutput, error) {
	return client.client.InvokeModel(ctx, input, options...)
}

func (client *legacyAWSClient) CountTokens(
	ctx context.Context, input *bedrockruntime.CountTokensInput, options ...func(*bedrockruntime.Options),
) (*bedrockruntime.CountTokensOutput, error) {
	return client.client.CountTokens(ctx, input, options...)
}

func (client *legacyAWSClient) InvokeModelWithResponseStream(
	ctx context.Context, input *bedrockruntime.InvokeModelWithResponseStreamInput,
	options ...func(*bedrockruntime.Options),
) (LegacyBedrockEventStream, error) {
	output, err := client.client.InvokeModelWithResponseStream(ctx, input, options...)
	if err != nil {
		return nil, err
	}
	return output.GetStream(), nil
}

func (model *Model) requestLegacyBedrock(
	ctx context.Context, payload *messagesRequest, headers map[string]string,
) (*ai.ModelResponse, error) {
	body, err := legacyBedrockBody(payload)
	if err != nil {
		return nil, err
	}
	output, err := model.legacyBedrockClient.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId: aws.String(model.name), Body: body,
		ContentType: aws.String("application/json"), Accept: aws.String("application/json"),
	}, legacyBedrockOptions(headers))
	if err != nil {
		return nil, model.legacyBedrockError(ctx, "request", err)
	}
	if output == nil {
		return nil, fmt.Errorf("anthropic: legacy Bedrock response is nil")
	}
	response, err := parseResponse(output.Body)
	if response != nil {
		response.ProviderName = "anthropic"
		response.ProviderURL = model.baseURL
	}
	return response, err
}

func (model *Model) countLegacyBedrock(
	ctx context.Context, payload *messagesRequest, headers map[string]string,
) (ai.Usage, error) {
	body, err := legacyBedrockBody(payload)
	if err != nil {
		return ai.Usage{}, err
	}
	output, err := model.legacyBedrockClient.CountTokens(ctx, &bedrockruntime.CountTokensInput{
		ModelId: aws.String(model.name),
		Input:   &types.CountTokensInputMemberInvokeModel{Value: types.InvokeModelTokensRequest{Body: body}},
	}, legacyBedrockOptions(headers))
	if err != nil {
		return ai.Usage{}, model.legacyBedrockError(ctx, "token count request", err)
	}
	if output == nil || output.InputTokens == nil {
		return ai.Usage{}, fmt.Errorf("anthropic: legacy Bedrock token count response omitted inputTokens")
	}
	return ai.Usage{InputTokens: int(*output.InputTokens)}, nil
}

func legacyBedrockBody(payload *messagesRequest) ([]byte, error) {
	body, err := marshalRequest(payload, payload.ExtraBody)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal legacy Bedrock request: %w", err)
	}
	var fields map[string]any
	_ = json.Unmarshal(body, &fields)
	delete(fields, "model")
	fields["anthropic_version"] = "bedrock-2023-05-31"
	encoded, _ := json.Marshal(fields)
	return encoded, nil
}

func legacyBedrockOptions(headers map[string]string) func(*bedrockruntime.Options) {
	cloned := make(map[string]string, len(headers))
	for name, value := range headers {
		cloned[name] = value
	}
	return func(options *bedrockruntime.Options) {
		if len(cloned) == 0 {
			return
		}
		options.APIOptions = append(options.APIOptions, func(stack *middleware.Stack) error {
			return stack.Build.Add(middleware.BuildMiddlewareFunc(
				"PydanticAIAnthropicBedrockHeaders",
				func(
					ctx context.Context, input middleware.BuildInput, next middleware.BuildHandler,
				) (middleware.BuildOutput, middleware.Metadata, error) {
					request := input.Request.(*smithyhttp.Request)
					for name, value := range cloned {
						request.Header.Set(name, value)
					}
					return next.HandleBuild(ctx, input)
				},
			), middleware.After)
		})
	}
}

func (model *Model) legacyBedrockError(ctx context.Context, operation string, err error) error {
	var responseError *smithyhttp.ResponseError
	if errors.As(err, &responseError) {
		return &APIError{StatusCode: responseError.HTTPStatusCode(), Body: err.Error()}
	}
	return ai.NewModelTransportError(ctx, model, operation, err)
}

func legacyBedrockClientIsNil(client LegacyBedrockClient) bool {
	if client == nil {
		return true
	}
	value := reflect.ValueOf(client)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
