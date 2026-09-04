package bedrock

import (
	"fmt"
	"mime"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	ai "github.com/Kludex/pydantic-ai-go"
)

func binaryContentBlock(content ai.BinaryContent, documentIndex int) (types.ContentBlock, error) {
	mediaType, _, _ := mime.ParseMediaType(strings.ToLower(content.MediaType))
	switch mediaType {
	case "image/png", "image/jpeg", "image/gif", "image/webp":
		return &types.ContentBlockMemberImage{Value: types.ImageBlock{
			Format: imageFormat(mediaType), Source: &types.ImageSourceMemberBytes{Value: append([]byte(nil), content.Data...)},
		}}, nil
	case "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/flac", "audio/ogg", "audio/aac", "audio/mp4", "audio/webm":
		return &types.ContentBlockMemberAudio{Value: types.AudioBlock{
			Format: audioFormat(mediaType), Source: &types.AudioSourceMemberBytes{Value: append([]byte(nil), content.Data...)},
		}}, nil
	case "video/mp4", "video/webm", "video/quicktime", "video/x-matroska", "video/mpeg":
		return &types.ContentBlockMemberVideo{Value: types.VideoBlock{
			Format: videoFormat(mediaType), Source: &types.VideoSourceMemberBytes{Value: append([]byte(nil), content.Data...)},
		}}, nil
	case "application/pdf", "text/csv", "text/plain", "text/markdown", "text/html",
		"application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
		"application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
		return &types.ContentBlockMemberDocument{Value: types.DocumentBlock{
			Name: aws.String(fmt.Sprintf("document-%d", documentIndex+1)), Format: documentFormat(mediaType),
			Source: &types.DocumentSourceMemberBytes{Value: append([]byte(nil), content.Data...)},
		}}, nil
	default:
		return nil, fmt.Errorf("bedrock: unsupported binary content media type %q", content.MediaType)
	}
}

func uploadedFileBlock(file ai.UploadedFile, documentIndex int) (types.ContentBlock, error) {
	if file.ProviderName != "bedrock" {
		return nil, fmt.Errorf("bedrock: uploaded file %q belongs to provider %q", file.FileID, file.ProviderName)
	}
	if !strings.HasPrefix(file.FileID, "s3://") {
		return nil, fmt.Errorf("bedrock: uploaded file %q must use an s3:// URI", file.FileID)
	}
	location := types.S3Location{Uri: aws.String(file.FileID)}
	mediaType, _, _ := mime.ParseMediaType(strings.ToLower(file.ResolvedMediaType()))
	switch {
	case strings.HasPrefix(mediaType, "image/"):
		format := imageFormat(mediaType)
		if format == "" {
			return nil, fmt.Errorf("bedrock: unsupported uploaded file media type %q", file.MediaType)
		}
		return &types.ContentBlockMemberImage{Value: types.ImageBlock{
			Format: format, Source: &types.ImageSourceMemberS3Location{Value: location},
		}}, nil
	case strings.HasPrefix(mediaType, "audio/"):
		format := audioFormat(mediaType)
		if format == "" {
			return nil, fmt.Errorf("bedrock: unsupported uploaded file media type %q", file.MediaType)
		}
		return &types.ContentBlockMemberAudio{Value: types.AudioBlock{
			Format: format, Source: &types.AudioSourceMemberS3Location{Value: location},
		}}, nil
	case strings.HasPrefix(mediaType, "video/"):
		format := videoFormat(mediaType)
		if format == "" {
			return nil, fmt.Errorf("bedrock: unsupported uploaded file media type %q", file.MediaType)
		}
		return &types.ContentBlockMemberVideo{Value: types.VideoBlock{
			Format: format, Source: &types.VideoSourceMemberS3Location{Value: location},
		}}, nil
	default:
		format := documentFormat(mediaType)
		if format == "" {
			return nil, fmt.Errorf("bedrock: unsupported uploaded file media type %q", file.MediaType)
		}
		return &types.ContentBlockMemberDocument{Value: types.DocumentBlock{
			Name: aws.String(fmt.Sprintf("document-%d", documentIndex+1)), Format: format,
			Source: &types.DocumentSourceMemberS3Location{Value: location},
		}}, nil
	}
}
