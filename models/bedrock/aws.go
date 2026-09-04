package bedrock

import (
	"context"
	"errors"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/smithy-go/middleware"
	smithyhttp "github.com/aws/smithy-go/transport/http"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// APIError reports an HTTP error returned by Bedrock Runtime.
type APIError struct {
	// StatusCode is the Bedrock HTTP response status.
	StatusCode int
	// Err is the AWS SDK operation failure.
	Err error
}

// Error describes the failed Bedrock request.
func (err *APIError) Error() string {
	if err.Err == nil {
		return "bedrock: API request failed"
	}
	return "bedrock: " + err.Err.Error()
}

// Unwrap returns the AWS SDK operation failure.
func (err *APIError) Unwrap() error { return err.Err }

// IsModelAPIError marks the response as eligible for model fallback.
func (*APIError) IsModelAPIError() bool { return true }

func modelError(ctx context.Context, model *Model, operation string, err error) error {
	var responseError *smithyhttp.ResponseError
	if errors.As(err, &responseError) {
		return &APIError{StatusCode: responseError.HTTPStatusCode(), Err: err}
	}
	return ai.NewModelTransportError(ctx, model, operation, err)
}

func requestOptions(headers map[string]string) func(*bedrockruntime.Options) {
	cloned := make(map[string]string, len(headers))
	for name, value := range headers {
		cloned[name] = value
	}
	return func(options *bedrockruntime.Options) {
		if len(cloned) == 0 {
			return
		}
		options.APIOptions = append(options.APIOptions, func(stack *middleware.Stack) error {
			return stack.Finalize.Insert(middleware.FinalizeMiddlewareFunc(
				"PydanticAIBedrockHeaders",
				func(
					ctx context.Context, input middleware.FinalizeInput, next middleware.FinalizeHandler,
				) (middleware.FinalizeOutput, middleware.Metadata, error) {
					request := input.Request.(*smithyhttp.Request)
					for name, value := range cloned {
						request.Header.Set(name, value)
					}
					return next.HandleFinalize(ctx, input)
				},
			), "Signing", middleware.Before)
		})
	}
}

type awsRuntimeClient interface {
	Converse(
		context.Context, *bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error)
	ConverseStream(
		context.Context, *bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseStreamOutput, error)
	CountTokens(
		context.Context, *bedrockruntime.CountTokensInput, ...func(*bedrockruntime.Options),
	) (*bedrockruntime.CountTokensOutput, error)
}

type awsClient struct{ client awsRuntimeClient }

func (client *awsClient) Converse(
	ctx context.Context, input *bedrockruntime.ConverseInput, options ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	return client.client.Converse(ctx, input, options...)
}

func (client *awsClient) ConverseStream(
	ctx context.Context, input *bedrockruntime.ConverseStreamInput, options ...func(*bedrockruntime.Options),
) (EventStream, error) {
	output, err := client.client.ConverseStream(ctx, input, options...)
	if err != nil {
		return nil, err
	}
	return &awsEventStream{EventStream: output.GetStream(), metadata: output.ResultMetadata}, nil
}

type awsEventStream struct {
	EventStream
	metadata middleware.Metadata
}

func (stream *awsEventStream) ResultMetadata() middleware.Metadata { return stream.metadata }

func (client *awsClient) CountTokens(
	ctx context.Context, input *bedrockruntime.CountTokensInput, options ...func(*bedrockruntime.Options),
) (*bedrockruntime.CountTokensOutput, error) {
	return client.client.CountTokens(ctx, input, options...)
}

func awsProviderURL(config aws.Config) string {
	if config.BaseEndpoint != nil {
		return strings.TrimRight(*config.BaseEndpoint, "/")
	}
	if config.Region == "" {
		return ""
	}
	suffix := ".amazonaws.com"
	if strings.HasPrefix(config.Region, "cn-") {
		suffix = ".amazonaws.com.cn"
	}
	return "https://bedrock-runtime." + config.Region + suffix
}
