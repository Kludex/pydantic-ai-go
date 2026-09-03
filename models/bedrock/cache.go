package bedrock

import (
	"fmt"
	"slices"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	ai "github.com/Kludex/pydantic-ai-go"
)

func cachePointBlock(point ai.CachePoint) (types.ContentBlock, error) {
	ttl, err := point.ResolvedTTL()
	if err != nil {
		return nil, err
	}
	cacheTTL := types.CacheTTLFiveMinutes
	if ttl == ai.CachePointTTL1Hour {
		cacheTTL = types.CacheTTLOneHour
	}
	return &types.ContentBlockMemberCachePoint{Value: types.CachePointBlock{
		Type: types.CachePointTypeDefault, Ttl: cacheTTL,
	}}, nil
}

func attachCachePoint(messages []types.Message, point types.ContentBlock) error {
	for index := len(messages) - 1; index >= 0; index-- {
		message := &messages[index]
		if message.Role != types.ConversationRoleUser || len(message.Content) == 0 {
			continue
		}
		content, _ := insertCachePoint(message.Content, point, true)
		message.Content = content
		return nil
	}
	return fmt.Errorf("bedrock: cache point requires preceding user content")
}

func insertCachePoint(
	content []types.ContentBlock, point types.ContentBlock, replace bool,
) ([]types.ContentBlock, error) {
	index := len(content)
	for index > 0 {
		if _, document := content[index-1].(*types.ContentBlockMemberDocument); !document {
			break
		}
		index--
	}
	if index == 0 {
		return nil, fmt.Errorf("bedrock: cache point requires preceding non-document content")
	}
	if _, previousIsPoint := content[index-1].(*types.ContentBlockMemberCachePoint); previousIsPoint {
		if !replace {
			return nil, fmt.Errorf("bedrock: cache points require content between them")
		}
		cloned := slices.Clone(content)
		cloned[index-1] = point
		return cloned, nil
	}
	content = slices.Clone(content)
	content = append(content, nil)
	copy(content[index+1:], content[index:])
	content[index] = point
	return content, nil
}

func limitCachePoints(messages []types.Message, maximum int) {
	remaining := maximum
	for messageIndex := len(messages) - 1; messageIndex >= 0; messageIndex-- {
		content := messages[messageIndex].Content
		for blockIndex := len(content) - 1; blockIndex >= 0; blockIndex-- {
			if _, point := content[blockIndex].(*types.ContentBlockMemberCachePoint); !point {
				continue
			}
			if remaining > 0 {
				remaining--
				continue
			}
			content = append(content[:blockIndex], content[blockIndex+1:]...)
		}
		messages[messageIndex].Content = content
	}
}

func hasDocumentWithoutText(content []types.ContentBlock) bool {
	hasDocument := false
	for _, block := range content {
		switch block.(type) {
		case *types.ContentBlockMemberText:
			return false
		case *types.ContentBlockMemberDocument:
			hasDocument = true
		}
	}
	return hasDocument
}
