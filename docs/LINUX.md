# Linux backend

The Linux side is a llama.cpp-based engine with remote model-weight placement. It reads a local GGUF, uploads its
weights to one RPC device, and exposes completion and chat APIs. The included local CPU worker lets development and
integration tests run without an iPhone. The device target remains iPhone 15+ with iOS 26+.

## Build

On Fedora:

```bash
sudo dnf install git cmake gcc-c++ make python3
git submodule update --init --recursive
bash scripts/build-linux.sh
```

The script applies `patches/llama-linux.patch` and `patches/llama-remote-stream.patch` to the pinned submodule, then builds
`llama-server` and `ggml-rpc-server`. The first patch fixes Linux portability; the second adds bounded synchronous reads
for RPC tensors when mmap and tensor validation are disabled. Local CPU/GPU loading and model math are unchanged.

Rerunning the build is supported. An unmatched patch stops the build rather than overwriting other engine edits. The
submodule appears modified after the patches are applied; keep the pinned submodule revision when committing the parent
repository. Build parallelism defaults to at most four jobs; set `JOBS=2` to reduce build memory use.

`BACKEND=vulkan bash scripts/build-linux.sh` also builds local Vulkan support when Vulkan headers and a shader compiler
are installed. Pass `--server build/linux-vulkan/bin/llama-server` to `serve` or `doctor`. iEgpu still selects only the
remote device for weights; combined Iris Xe/remote GPU placement is outside the current backend.

## Commands

| Command | Purpose |
| --- | --- |
| `./iegpu worker` | Start a loopback CPU RPC worker with disk caching disabled |
| `./iegpu doctor --rpc 127.0.0.1:50052` | Verify the RPC protocol and single-device discovery; do not load a model |
| `./iegpu serve --rpc 127.0.0.1:50052 --model model.gguf` | Load local weights into the worker and serve inference |
| `./iegpu complete --prompt "Hello"` | Request text from the running local API |

`serve` defaults to 2,048 context tokens, one concurrent slot, and API port 8080. `--ctx` and `--port` change those settings.
The model path can contain spaces. `complete` supports `--tokens`, `--temperature`, `--json`, and `--port`; without
`--prompt`, it reads standard input. It prints the completed response rather than streaming tokens.

The worker defaults to port 50052 and at most four CPU threads; use `--port` and `--threads` to change them. It deliberately
exposes only CPU, so backend development does not depend on Iris Xe or a GPU driver. Run it in a separate terminal before
starting `serve`. Ctrl-C stops each command and cleans up its native child process.

Direct endpoints accept only `127.0.0.1:PORT`. For a worker on another trusted machine, use an authenticated local port
forward. The native RPC protocol has no authentication and must not be exposed publicly. It cannot report whether its
weight cache is enabled: use `iegpu worker` or configure your worker without `--cache` / with a null native cache path.
The automatic USB path additionally verifies cache policy through the phone control service.

## Loading and inference

The engine uses `--device RPC0`, 999 offloaded layers, an all-tensor override to `RPC0[endpoint]`, `--fit off`,
`--load-mode none`, and `--lazy-mode off`. The override includes input embeddings, which this engine otherwise keeps on
the host. iEgpu has no layer-splitting option: an allocation failure is an error, not a fallback to laptop weights.

The loader reads RPC tensor data in chunks of at most 8 MiB, uploads each chunk synchronously, and reuses its staging
buffer. This also works for tensors larger than the chunk limit. Native RPC serializes each transfer into an additional
buffer. The patch preserves the full-tensor validation path for other engine callers that enable tensor checking;
iEgpu does not request that mode.

Metadata and vocabulary remain on the laptop. Tokenization, sampling, HTTP handling, scheduling, and some model operations
still execute there. There is no full-model host weight buffer in the tested path, but total host RAM is greater than the
staging buffer size. The OS can cache file reads. Prompt-state RAM caching and context checkpoints are disabled; ordinary
in-session KV memory is still required by the model.

The loading log lists every model-weight buffer. Each should say `RPC0[127.0.0.1:PORT] model buffer size`. Streaming lines
include each tensor's byte size and the 8,388,608-byte staging limit. CPU compute or output buffers are expected; CPU
**model** buffers are not expected in this mode.

The API becomes ready after loading and warmup. Check `GET /health`, then use `/completion`, `/v1/completions`, or
`/v1/chat/completions`. Chat requires a suitable model template. `complete` uses the plain completion API and ignores
proxy environment variables for loopback requests. Native engine argument environment variables (`LLAMA_ARG_*`) are
removed when launching it so they cannot silently change placement or network settings.

## Future phone worker contract

The Linux host and worker must use the same pinned llama.cpp RPC revision. No GGUF parser, tokenizer, chat API, or local
model file is needed on the phone: its worker receives tensor buffers and executes backend graphs.

The existing USB adapter expects:

| Service | Requirement |
| --- | --- |
| GPU RPC, phone port 50052 | One Metal device using the pinned ggml RPC protocol; native cache path must be null |
| Control, phone port 50061 | On `mem` followed by newline, reply with one newline-terminated JSON object |
| Control flags | `rpc_cache_enabled: false` and `rpc_only: true`, using JSON booleans |
| Optional memory field | `avail_mb` reports the app's currently available memory in MiB |

The Linux launcher uses USB-only `idevice_id -l` and `iproxy -l`, with loopback forwards for both services. It verifies the
control flags before any weight transfer and fails closed when caching is enabled or unknown. For several USB devices,
select one using `--udid`. No Wi-Fi discovery or Personal Hotspot is needed.

Later, install USB dependencies on Fedora:

```bash
sudo dnf install usbmuxd libusbmuxd-utils libimobiledevice-utils
idevicepair pair
./iegpu doctor
./iegpu serve --model ~/Models/model.gguf
```

This path needs an installed, signed app running in the foreground on an unlocked phone. A cable alone cannot execute
GPU work on iOS. The existing app scaffold can be built on a Mac with Xcode 26+ using
`RPC_ONLY=1 IPA=1 bash scripts/build-iphone.sh`, or through the manual iPhone workflow. Both produce an unsigned IPA that
must be signed and installed. These phone steps remain unverified and are deferred from the Linux backend work.
[Existing installation notes](../INSTALL-IPHONE.md) describe signing and sideloading assumptions.

Weights, KV cache, and work buffers must fit the phone's app memory budget. Base and Pro phones have different memory and
transfer limits. Compatibility and performance need device measurements; the local CPU worker cannot establish them.
After app termination, locking, or cable loss, reconnect and restart; session recovery is not implemented.

## Tests and troubleshooting

```bash
python3 -m unittest discover -s tests/linux -v
bash scripts/check-listeners.sh
python3 tests/linux/smoke_rpc.py
```

The smoke test generates a GGUF with embeddings and output tensors larger than 8 MiB. It runs the real CLI against a
cache-free local CPU worker, verifies all weight buffers are remote, compares 16 greedy tokens with a local CPU baseline,
calls both the CLI client and OpenAI completion API, then disconnects the worker and verifies inference stops. It downloads
no model and deletes its fixture on exit. Inspect `build/smoke-rpc.log`, `smoke-cpu.log`, and `smoke-host.log` for failures.

- **RPC handshake fails:** start the worker first, check its port, and use binaries from the same pinned build. A worker
  exposing multiple devices is rejected; expose only the intended device.
- **Model loading fails:** check the GGUF, worker memory, and context size. Lower `--ctx` or use a smaller quantized model.
  The backend does not silently split weights onto the laptop.
- **API unavailable:** wait for loading to finish and check `/health`; `--port` on the client must match the server.
- **Chat template error:** use a GGUF with a supported template, or use plain completion.
- **Worker disconnected:** restart the worker and `serve`. The native engine can terminate on transport errors; there is
  no automatic CPU recovery.

Linux CI builds and runs the backend checks. The original Mac security suite requires Xcode and is separate from these
host tests. Physical-phone network exposure, signing, cache inspection, USB loading, and Metal correctness remain future
integration checks.
