package ai

import (
	"encoding/base64"
	"encoding/json"
)

type urlBase64Bytes []byte

func (data urlBase64Bytes) MarshalJSON() ([]byte, error) {
	return json.Marshal(base64.URLEncoding.EncodeToString(data))
}

func (data *urlBase64Bytes) UnmarshalJSON(encoded []byte) error {
	var value string
	if err := json.Unmarshal(encoded, &value); err != nil {
		return err
	}
	decoded, err := base64.URLEncoding.DecodeString(value)
	if err != nil {
		decoded, err = base64.StdEncoding.DecodeString(value)
		if err != nil {
			return err
		}
	}
	*data = decoded
	return nil
}
