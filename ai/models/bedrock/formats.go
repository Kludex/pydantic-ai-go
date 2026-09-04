package bedrock

import "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

func imageFormat(mediaType string) types.ImageFormat {
	formats := map[string]types.ImageFormat{
		"image/png": types.ImageFormatPng, "image/jpeg": types.ImageFormatJpeg,
		"image/gif": types.ImageFormatGif, "image/webp": types.ImageFormatWebp,
	}
	return formats[mediaType]
}

func documentFormat(mediaType string) types.DocumentFormat {
	formats := map[string]types.DocumentFormat{
		"application/pdf": types.DocumentFormatPdf, "text/csv": types.DocumentFormatCsv,
		"text/plain": types.DocumentFormatTxt, "text/markdown": types.DocumentFormatMd,
		"text/html": types.DocumentFormatHtml, "application/msword": types.DocumentFormatDoc,
		"application/vnd.openxmlformats-officedocument.wordprocessingml.document": types.DocumentFormatDocx,
		"application/vnd.ms-excel": types.DocumentFormatXls,
		"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": types.DocumentFormatXlsx,
	}
	return formats[mediaType]
}

func audioFormat(mediaType string) types.AudioFormat {
	formats := map[string]types.AudioFormat{
		"audio/mpeg": types.AudioFormatMp3, "audio/mp3": types.AudioFormatMp3,
		"audio/wav": types.AudioFormatWav, "audio/x-wav": types.AudioFormatWav,
		"audio/flac": types.AudioFormatFlac, "audio/ogg": types.AudioFormatOgg,
		"audio/aac": types.AudioFormatAac, "audio/mp4": types.AudioFormatMp4,
		"audio/webm": types.AudioFormatWebm,
	}
	return formats[mediaType]
}

func videoFormat(mediaType string) types.VideoFormat {
	formats := map[string]types.VideoFormat{
		"video/mp4": types.VideoFormatMp4, "video/webm": types.VideoFormatWebm,
		"video/quicktime": types.VideoFormatMov, "video/x-matroska": types.VideoFormatMkv,
		"video/mpeg": types.VideoFormatMpeg,
	}
	return formats[mediaType]
}

func stringValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func int32Value(value *int32) int {
	if value == nil {
		return 0
	}
	return int(*value)
}
