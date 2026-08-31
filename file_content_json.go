package ai

import (
	"encoding/json"
	"fmt"
)

// MarshalJSON uses PydanticAI's persisted image URL shape.
func (image ImageURL) MarshalJSON() ([]byte, error) {
	wire, err := marshalFileURL(
		"image-url", image.URL, image.ResolvedMediaType, image.ResolvedIdentifier(), image.ForceDownload,
		image.VendorMetadata,
	)
	if err != nil {
		return nil, err
	}
	return json.Marshal(wire)
}

// UnmarshalJSON reads PydanticAI's persisted image URL shape.
func (image *ImageURL) UnmarshalJSON(data []byte) error {
	content, err := unmarshalFileContentJSON(data, "image-url")
	if err != nil {
		return err
	}
	*image = content.(ImageURL)
	return nil
}

// MarshalJSON uses PydanticAI's persisted video URL shape.
func (video VideoURL) MarshalJSON() ([]byte, error) {
	wire, err := marshalFileURL(
		"video-url", video.URL, video.ResolvedMediaType, video.ResolvedIdentifier(), video.ForceDownload,
		video.VendorMetadata,
	)
	if err != nil {
		return nil, err
	}
	return json.Marshal(wire)
}

// UnmarshalJSON reads PydanticAI's persisted video URL shape.
func (video *VideoURL) UnmarshalJSON(data []byte) error {
	content, err := unmarshalFileContentJSON(data, "video-url")
	if err != nil {
		return err
	}
	*video = content.(VideoURL)
	return nil
}

// MarshalJSON uses PydanticAI's persisted audio URL shape.
func (audio AudioURL) MarshalJSON() ([]byte, error) {
	wire, err := marshalFileURL(
		"audio-url", audio.URL, audio.ResolvedMediaType, audio.ResolvedIdentifier(), audio.ForceDownload,
		audio.VendorMetadata,
	)
	if err != nil {
		return nil, err
	}
	return json.Marshal(wire)
}

// UnmarshalJSON reads PydanticAI's persisted audio URL shape.
func (audio *AudioURL) UnmarshalJSON(data []byte) error {
	content, err := unmarshalFileContentJSON(data, "audio-url")
	if err != nil {
		return err
	}
	*audio = content.(AudioURL)
	return nil
}

// MarshalJSON uses PydanticAI's persisted document URL shape.
func (document DocumentURL) MarshalJSON() ([]byte, error) {
	wire, err := marshalFileURL(
		"document-url", document.URL, document.ResolvedMediaType, document.ResolvedIdentifier(),
		document.ForceDownload, document.VendorMetadata,
	)
	if err != nil {
		return nil, err
	}
	return json.Marshal(wire)
}

// UnmarshalJSON reads PydanticAI's persisted document URL shape.
func (document *DocumentURL) UnmarshalJSON(data []byte) error {
	content, err := unmarshalFileContentJSON(data, "document-url")
	if err != nil {
		return err
	}
	*document = content.(DocumentURL)
	return nil
}

// MarshalJSON uses PydanticAI's persisted uploaded-file shape.
func (file UploadedFile) MarshalJSON() ([]byte, error) {
	return json.Marshal(wireUserContent{
		Kind: "uploaded-file", FileID: file.FileID, ProviderName: file.ProviderName,
		MediaType: file.MediaType, Identifier: file.Identifier, VendorMetadata: file.VendorMetadata,
	})
}

// UnmarshalJSON reads PydanticAI's persisted uploaded-file shape.
func (file *UploadedFile) UnmarshalJSON(data []byte) error {
	content, err := unmarshalFileContentJSON(data, "uploaded-file")
	if err != nil {
		return err
	}
	*file = content.(UploadedFile)
	return nil
}

// MarshalJSON uses PydanticAI's persisted binary content shape.
func (content BinaryContent) MarshalJSON() ([]byte, error) {
	return json.Marshal(wireUserContent{
		Kind: "binary", Data: urlBase64Bytes(content.Data), MediaType: content.MediaType,
		Identifier: content.ResolvedIdentifier(), VendorMetadata: content.VendorMetadata,
	})
}

// UnmarshalJSON reads PydanticAI's persisted binary content shape.
func (content *BinaryContent) UnmarshalJSON(data []byte) error {
	value, err := unmarshalFileContentJSON(data, "binary")
	if err != nil {
		return err
	}
	*content = value.(BinaryContent)
	return nil
}

func unmarshalFileContentJSON(data []byte, expectedKind string) (UserContent, error) {
	var wire wireUserContent
	if err := json.Unmarshal(data, &wire); err != nil {
		return nil, err
	}
	if wire.Kind == "" {
		wire.Kind = expectedKind
	}
	if wire.Kind != expectedKind {
		return nil, fmt.Errorf("ai: expected %s content, got %s", expectedKind, wire.Kind)
	}
	return unmarshalUserContentItem(wire)
}
