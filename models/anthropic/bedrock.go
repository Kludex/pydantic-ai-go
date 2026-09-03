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

	ai "github.com/Kludex/pydantic-ai-go"
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
	model.legacyBedrockClient = config.Client
	model.baseURL = config.ProviderURL
	return model
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
	delete(fields, "stream")
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
