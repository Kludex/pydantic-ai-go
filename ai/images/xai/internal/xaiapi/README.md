# xAI image protocol bindings

These Go bindings are generated from the `image.proto` and `usage.proto` descriptors shipped by the official [`xai-sdk-python`](https://github.com/xai-org/xai-sdk-python) v1.18.0 release at commit `a2231b51520e590f1df3c497f99bd95cd893dc53`.

xAI does not publish an official Go SDK or Go protobuf module. The checked-in bindings keep the Go adapter on the same `xai_api.Image/GenerateImage` contract used by the official SDK. See `LICENSE` for the upstream Apache 2.0 license.

Regenerate the bindings from the repository root after updating the protocol files:

```console
$ protoc -I ai/images/xai/internal/xaiapi/proto \
    --go_out=. --go_opt=module=github.com/Kludex/pydantic-ai-go \
    --go-grpc_out=. --go-grpc_opt=module=github.com/Kludex/pydantic-ai-go \
    ai/images/xai/internal/xaiapi/proto/xai/api/v1/usage.proto \
    ai/images/xai/internal/xaiapi/proto/xai/api/v1/image.proto
```
