package bedrock

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/internal/download"
)

func userContentBlocks(
	ctx context.Context, prompt ai.UserPromptPart, priorMessages []types.Message,
) ([]types.ContentBlock, error) {
	if len(prompt.Contents) == 0 {
		return []types.ContentBlock{&types.ContentBlockMemberText{Value: prompt.Content}}, nil
	}
	blocks := make([]types.ContentBlock, 0, len(prompt.Contents))
	documentIndex := 0
	for _, content := range prompt.Contents {
		var block types.ContentBlock
		var err error
		switch value := content.(type) {
		case ai.TextContent:
			block = &types.ContentBlockMemberText{Value: value.Text}
		case ai.BinaryContent:
			block, err = binaryContentBlock(value, documentIndex)
		case ai.ImageURL:
			block, err = downloadBlock(ctx, value.URL, value.MediaType, value.ForceDownload, documentIndex)
		case ai.VideoURL:
			block, err = downloadBlock(ctx, value.URL, value.MediaType, value.ForceDownload, documentIndex)
		case ai.AudioURL:
			block, err = downloadBlock(ctx, value.URL, value.MediaType, value.ForceDownload, documentIndex)
		case ai.DocumentURL:
			block, err = downloadBlock(ctx, value.URL, value.MediaType, value.ForceDownload, documentIndex)
		case ai.UploadedFile:
			block, err = uploadedFileBlock(value, documentIndex)
		case ai.CachePoint:
			block, err = cachePointBlock(value)
			if err == nil {
				if len(blocks) == 0 {
					err = attachCachePoint(priorMessages, block)
					if err == nil {
						continue
					}
				} else {
					blocks, err = insertCachePoint(blocks, block, false)
					if err == nil {
						continue
					}
				}
			}
		}
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, block)
		if _, ok := block.(*types.ContentBlockMemberDocument); ok {
			documentIndex++
		}
	}
	if hasDocumentWithoutText(blocks) {
		blocks = append([]types.ContentBlock{&types.ContentBlockMemberText{Value: "See attached document(s)."}}, blocks...)
	}
	return blocks, nil
}

func downloadBlock(
	ctx context.Context, url string, mediaType string, mode ai.FileDownloadMode, documentIndex int,
) (types.ContentBlock, error) {
	if err := mode.Validate(); err != nil {
		return nil, err
	}
	result, err := download.Fetch(ctx, url, mode == ai.FileDownloadAllowLocal)
	if err != nil {
		return nil, err
	}
	if mediaType == "" {
		mediaType = result.MediaType
	}
	return binaryContentBlock(ai.BinaryContent{Data: result.Data, MediaType: mediaType}, documentIndex)
}
