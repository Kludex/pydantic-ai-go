package xai

import (
	"context"
	"fmt"
	"math"

	"github.com/Kludex/pydantic-ai-go/ai/images"
	"github.com/Kludex/pydantic-ai-go/ai/images/xai/internal/xaiapi"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

type inputReferences struct {
	imageURL     string
	imageFileID  string
	imageURLs    []string
	imageFileIDs []string
}

// Generate creates, edits, or batches images through xAI's GenerateImage RPC.
func (model *Model) Generate(
	ctx context.Context, prompt string, inputs []images.Input, settings images.Settings,
) (*images.Result, error) {
	settings = images.MergeSettings(model.settings, settings)
	prompt, inputs, settings, err := images.PrepareRequest(prompt, inputs, settings)
	if err != nil {
		return nil, err
	}
	provider, err := extractSettings(settings)
	if err != nil {
		return nil, err
	}
	count := 1
	if provider.n != nil {
		count = *provider.n
	}
	if count <= 0 || count > math.MaxInt32 {
		return nil, fmt.Errorf("xai images: image count must be between 1 and %d", math.MaxInt32)
	}
	ratio, resolution, conflicts, err := resolveGeometry(model.name, settings, provider)
	if err != nil {
		return nil, err
	}
	references, err := model.mapInputs(ctx, inputs)
	if err != nil {
		return nil, err
	}
	request := &xaiapi.GenerateImageRequest{
		Prompt: prompt, Model: model.name, N: int32Pointer(int32(count)), User: provider.user,
		Format: xaiapi.ImageFormat_IMG_FORMAT_BASE64,
	}
	if ratio != "" {
		value := imageAspectRatios[ratio]
		request.AspectRatio = &value
	}
	if resolution != "" {
		value := imageResolutions[resolution]
		request.Resolution = &value
	}
	request.Image, request.Images = mapReferences(references)
	client, err := model.imageClient()
	if err != nil {
		return nil, err
	}
	if model.apiKey != "" {
		ctx = metadata.AppendToOutgoingContext(ctx, "authorization", "Bearer "+model.apiKey)
	}
	response, err := client.GenerateImage(ctx, request)
	if err != nil {
		if ctx.Err() != nil {
			return nil, fmt.Errorf("xai images: GenerateImage: %w", ctx.Err())
		}
		return nil, newAPIError(ctx, model, err)
	}
	result, err := model.parseResponse(prompt, response, count)
	if err != nil {
		return nil, err
	}
	for _, conflict := range conflicts {
		result.Warnings = append(result.Warnings, "xai images: used provider-specific geometry instead of portable "+conflict)
	}
	if len(settings.ExtraHeaders) > 0 {
		result.Warnings = append(result.Warnings, "xai images: gRPC transport ignored extra headers")
	}
	if len(settings.ExtraBody) > 0 {
		result.Warnings = append(result.Warnings, "xai images: gRPC transport ignored extra body fields")
	}
	return result, nil
}

var imageAspectRatios = map[images.AspectRatio]xaiapi.ImageAspectRatio{
	images.AspectRatio1To1:    xaiapi.ImageAspectRatio_IMG_ASPECT_RATIO_1_1,
	images.AspectRatio3To4:    xaiapi.ImageAspectRatio_IMG_ASPECT_RATIO_3_4,
	images.AspectRatio4To3:    xaiapi.ImageAspectRatio_IMG_ASPECT_RATIO_4_3,
	images.AspectRatio9To16:   xaiapi.ImageAspectRatio_IMG_ASPECT_RATIO_9_16,
	images.AspectRatio16To9:   xaiapi.ImageAspectRatio_IMG_ASPECT_RATIO_16_9,
	images.AspectRatio2To3:    xaiapi.ImageAspectRatio_IMG_ASPECT_RATIO_2_3,
	images.AspectRatio3To2:    xaiapi.ImageAspectRatio_IMG_ASPECT_RATIO_3_2,
	images.AspectRatio9To19_5: xaiapi.ImageAspectRatio_IMG_ASPECT_RATIO_9_19_5,
	images.AspectRatio19_5To9: xaiapi.ImageAspectRatio_IMG_ASPECT_RATIO_19_5_9,
	images.AspectRatio9To20:   xaiapi.ImageAspectRatio_IMG_ASPECT_RATIO_9_20,
	images.AspectRatio20To9:   xaiapi.ImageAspectRatio_IMG_ASPECT_RATIO_20_9,
	images.AspectRatio1To2:    xaiapi.ImageAspectRatio_IMG_ASPECT_RATIO_1_2,
	images.AspectRatio2To1:    xaiapi.ImageAspectRatio_IMG_ASPECT_RATIO_2_1,
}

var imageResolutions = map[Resolution]xaiapi.ImageResolution{
	Resolution1K: xaiapi.ImageResolution_IMG_RESOLUTION_1K,
	Resolution2K: xaiapi.ImageResolution_IMG_RESOLUTION_2K,
}

func mapReferences(references inputReferences) (*xaiapi.ImageUrlContent, []*xaiapi.ImageUrlContent) {
	if references.imageURL != "" {
		return &xaiapi.ImageUrlContent{
			Source: &xaiapi.ImageUrlContent_ImageUrl{ImageUrl: references.imageURL},
			Detail: xaiapi.ImageDetail_DETAIL_AUTO,
		}, nil
	}
	if references.imageFileID != "" {
		return &xaiapi.ImageUrlContent{
			Source: &xaiapi.ImageUrlContent_FileId{FileId: references.imageFileID},
			Detail: xaiapi.ImageDetail_DETAIL_AUTO,
		}, nil
	}
	mapped := make([]*xaiapi.ImageUrlContent, 0, len(references.imageFileIDs)+len(references.imageURLs))
	for _, fileID := range references.imageFileIDs {
		mapped = append(mapped, &xaiapi.ImageUrlContent{
			Source: &xaiapi.ImageUrlContent_FileId{FileId: fileID}, Detail: xaiapi.ImageDetail_DETAIL_AUTO,
		})
	}
	for _, imageURL := range references.imageURLs {
		mapped = append(mapped, &xaiapi.ImageUrlContent{
			Source: &xaiapi.ImageUrlContent_ImageUrl{ImageUrl: imageURL}, Detail: xaiapi.ImageDetail_DETAIL_AUTO,
		})
	}
	return nil, mapped
}

func int32Pointer(value int32) *int32 { return &value }

// APIError is an error returned by xAI's image generation gRPC API.
type APIError struct {
	StatusCode int
	Code       codes.Code
	Body       string
	Err        error
}

// Error formats the provider status and detail.
func (err *APIError) Error() string {
	if err.StatusCode != 0 {
		return fmt.Sprintf("xai images: API returned status %d: %s", err.StatusCode, err.Body)
	}
	return fmt.Sprintf("xai images: gRPC returned %s: %s", err.Code, err.Body)
}

// Unwrap returns the underlying gRPC error.
func (err *APIError) Unwrap() error { return err.Err }

// IsModelAPIError marks provider API and transport failures as eligible for fallback decisions.
func (*APIError) IsModelAPIError() bool { return true }

func newAPIError(ctx context.Context, model *Model, err error) error {
	grpcStatus, ok := status.FromError(err)
	if !ok {
		return images.NewModelTransportError(ctx, model, "GenerateImage", err)
	}
	return &APIError{
		StatusCode: map[codes.Code]int{
			codes.InvalidArgument: 400, codes.Unauthenticated: 401, codes.PermissionDenied: 403,
			codes.NotFound: 404, codes.ResourceExhausted: 429, codes.Internal: 500,
			codes.Unavailable: 503, codes.DeadlineExceeded: 504,
		}[grpcStatus.Code()],
		Code: grpcStatus.Code(), Body: grpcStatus.Message(), Err: err,
	}
}
