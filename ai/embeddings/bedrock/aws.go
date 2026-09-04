package bedrock

import (
	"context"
	"errors"
	"strconv"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/smithy-go/middleware"
	smithyhttp "github.com/aws/smithy-go/transport/http"
)

// WithAWSConfig uses an already loaded AWS SDK configuration.
func WithAWSConfig(config aws.Config) Option {
	config = config.Copy()
	if config.BaseEndpoint != nil {
		baseEndpoint := *config.BaseEndpoint
		config.BaseEndpoint = &baseEndpoint
	}
	return func(model *Model) {
		model.client = &awsClient{client: bedrockruntime.NewFromConfig(config)}
		model.setProviderURL(awsProviderURL(config))
	}
}

// WithAWSLoadOptions configures loading the default AWS SDK configuration.
// These options are ignored when a client or loaded configuration is supplied.
func WithAWSLoadOptions(options ...func(*awsconfig.LoadOptions) error) Option {
	cloned := append([]func(*awsconfig.LoadOptions) error(nil), options...)
	return func(model *Model) { model.loadOptions = cloned }
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

type awsRuntimeClient interface {
	InvokeModel(context.Context, *bedrockruntime.InvokeModelInput, ...func(*bedrockruntime.Options)) (
		*bedrockruntime.InvokeModelOutput, error,
	)
}

type awsClient struct{ client awsRuntimeClient }

func (client *awsClient) InvokeModel(ctx context.Context, request Request) (Response, error) {
	inputTokens := 0
	output, err := client.client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId: &request.ModelID, Body: request.Body, ContentType: aws.String("application/json"), Accept: aws.String("application/json"),
	}, func(options *bedrockruntime.Options) {
		options.APIOptions = append(options.APIOptions, func(stack *middleware.Stack) error {
			var buildErr error
			if len(request.Headers) > 0 {
				headers := middleware.BuildMiddlewareFunc(
					"PydanticAIBedrockHeaders",
					func(
						ctx context.Context, input middleware.BuildInput, next middleware.BuildHandler,
					) (middleware.BuildOutput, middleware.Metadata, error) {
						httpRequest := input.Request.(*smithyhttp.Request)
						for key, value := range request.Headers {
							httpRequest.Header.Set(key, value)
						}
						return next.HandleBuild(ctx, input)
					},
				)
				buildErr = stack.Build.Add(headers, middleware.After)
			}
			capture := middleware.DeserializeMiddlewareFunc(
				"PydanticAIBedrockInputTokens",
				func(
					ctx context.Context, input middleware.DeserializeInput, next middleware.DeserializeHandler,
				) (middleware.DeserializeOutput, middleware.Metadata, error) {
					return captureInputTokens(ctx, input, next, &inputTokens)
				},
			)
			return errors.Join(buildErr, stack.Deserialize.Add(capture, middleware.After))
		})
	})
	if err != nil {
		var responseError *smithyhttp.ResponseError
		if errors.As(err, &responseError) {
			return Response{}, &APIError{
				StatusCode: responseError.HTTPStatusCode(), Headers: responseError.HTTPResponse().Header.Clone(), Err: err,
			}
		}
		return Response{}, err
	}
	return Response{Body: append([]byte(nil), output.Body...), InputTokens: inputTokens}, nil
}

func captureInputTokens(
	ctx context.Context, input middleware.DeserializeInput, next middleware.DeserializeHandler, inputTokens *int,
) (middleware.DeserializeOutput, middleware.Metadata, error) {
	output, metadata, err := next.HandleDeserialize(ctx, input)
	response, ok := output.RawResponse.(*smithyhttp.Response)
	if ok && response != nil && response.Response != nil {
		value := response.Header.Get("x-amzn-bedrock-input-token-count")
		if parsed, parseErr := strconv.Atoi(value); parseErr == nil {
			*inputTokens = parsed
		}
	}
	return output, metadata, err
}
